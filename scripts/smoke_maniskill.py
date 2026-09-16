#!/usr/bin/env python3
"""Staged smoke test for the ManiSkill port and the manuscript evaluation.

Each stage prints PASS or FAIL and the run stops at the first failure, so a
break is reported where it happens instead of as a traceback three layers
down. Nothing here needs a trained checkpoint: an untrained agent exercises
exactly the same plumbing. Pass --checkpoint to smoke real weights instead.

  python scripts/smoke_maniskill.py --logdir /tmp/ms_smoke
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# Sets XLA_PYTHON_CLIENT_PREALLOCATE before JAX initialises, and puts the repo
# root on sys.path the same way the real entry point does.
from dreamerv3 import main_maniskill  # noqa: E402  (import order is load-bearing)

import elements  # noqa: E402

STAGES = []


def stage(name):
  def wrap(fn):
    STAGES.append((name, fn))
    return fn
  return wrap


def build_config(args):
  configs = main_maniskill.load_configs()
  config = elements.Config(configs['defaults'])
  for name in args.configs.split():
    config = config.update(configs[name])
  config = config.update(logdir=str(args.logdir), task=args.task)
  config = config.update({
      'env.maniskill.num_envs': 1,   # clip collection needs one sub-env
      'logger.outputs': ['jsonl'],
      'jax.prealloc': False,
  })
  return config


@stage('build agent (obs/act space discovery)')
def check_agent(state, args):
  from dreamerv3 import main_htp
  state['config'] = config = build_config(args)
  state['agent'] = agent = main_htp.make_agent(config)
  obs, act = agent.obs_space, agent.act_space
  shapes = {k: tuple(v.shape) for k, v in act.items()}
  print(f'      obs_space: {sorted(obs)}')
  print(f'      act_space: {shapes}')
  assert 'image' in obs, (
      f'no image key in obs_space: {sorted(obs)}. The evaluation decodes an '
      'image; set env.maniskill.obs_mode to an rgb mode.')
  assert obs['image'].shape[-1] == 3, (
      f'image has {obs["image"].shape[-1]} channels; the figure grids need 3. '
      'Set env.maniskill.num_frames=1.')
  assert not act['action'].discrete, 'expected a continuous action space'
  if args.checkpoint:
    loaded = elements.Checkpoint()
    loaded.agent = agent
    loaded.load(Path(args.checkpoint), keys=['agent'])
    print(f'      loaded weights from {args.checkpoint}')


@stage('build env and check the batched interface')
def check_env(state, args):
  from dreamerv3 import main_htp
  state['env'] = env = main_htp.make_env(
      state['config'], 0, use_seed=False, seed=260909, num_envs=1)
  print(f'      is_batched={env.is_batched} num_envs={env.num_envs} '
        f'horizon={env._max_episode_steps}')
  assert env.is_batched, 'ManiSkill wrapper should report is_batched'
  from corewm_eval.config import MIN_CLIP_LENGTH
  assert env._max_episode_steps >= MIN_CLIP_LENGTH, (
      f'task horizon {env._max_episode_steps} is below the protocol minimum '
      f'{MIN_CLIP_LENGTH}; raise env.maniskill.max_episode_steps.')


@stage('step through BatchedDriver')
def check_driver(state, args):
  import embodied
  from corewm_eval.manuscript_clips import make_driver
  env, agent = state['env'], state['agent']
  driver = make_driver(env)
  assert isinstance(driver, embodied.BatchedDriver), type(driver)
  seen = []
  driver.on_step(lambda tran, worker: seen.append(tran))
  driver.reset(agent.init_policy)
  start = time.time()
  driver(lambda *xs: agent.policy(*xs, mode='eval'), steps=5)
  print(f'      {len(seen)} transitions in {time.time() - start:.1f}s')
  assert len(seen) == 5, f'expected 5 callbacks, got {len(seen)}'
  first = seen[0]
  assert bool(first['is_first']), 'first transition should be a reset'
  assert not bool(first['log/action_executed']), (
      'the reset call must not count as an executed env action')
  assert all(bool(t['log/action_executed']) for t in seen[1:]), (
      'steps after the reset should count as executed env actions')
  for key in ('image', 'state', 'reward', 'is_last'):
    assert np.asarray(first[key]).ndim == np.asarray(
        seen[1][key]).ndim, key
  print(f'      per-transition image shape: {np.asarray(first["image"]).shape}')
  # Deliberately not closed: a driver does not own the env until close(), and
  # the next stage reuses this one. collect_clips() closes it at the end.


@stage('collect one clip')
def check_clip(state, args):
  from corewm_eval.config import MIN_CLIP_LENGTH
  from corewm_eval.manuscript_clips import collect_clips
  clips = collect_clips(state['agent'], state['env'], episodes=1)
  values, metrics = clips[0]
  state['clip'] = values
  length = len(values['is_last'])
  print(f'      {length} frames, keys {sorted(values)}')
  print(f'      task metrics: {metrics}')
  assert length >= MIN_CLIP_LENGTH, (length, MIN_CLIP_LENGTH)
  assert bool(values['is_first'][0]), 'clip must start at a reset'
  assert values['is_last'][:-1].sum() == 0, (
      'clip crosses an episode boundary')
  assert values['action'].ndim == 2, (
      f'expected [T, act_dim] actions, got {values["action"].shape}')
  state['env'] = None  # collect_clips closed it through the driver


@stage('read-only evaluation on the clip')
def check_evaluation(state, args):
  from corewm_eval.config import MANUSCRIPT_START, ROLLOUT_LENGTH
  from corewm_eval.extraction import (
      ExtractionSpec, evaluation_seed, previous_actions)
  from corewm_eval.manuscript_eval import evaluate_readonly
  from corewm_eval.smoke_test import _tree_hash
  config, agent, ep = state['config'], state['agent'], state['clip']
  dyn = config.agent.dyn[config.agent.dyn.typ]
  dims = tuple(map(int, config.agent.htp.proj.dims))
  spec = ExtractionSpec(int(dyn.deter), (int(dyn.stoch), int(dyn.classes)), dims)
  before = _tree_hash(agent.save()['params'])
  obs = {k: v[None] for k, v in ep.items() if k != 'action'}
  actions = {'action': ep['action'][None]}
  start = time.time()
  result = evaluate_readonly(
      agent, obs, {'action': previous_actions(ep['action'])[None]}, actions,
      spec, evaluation_seed(0, 0, MANUSCRIPT_START), MANUSCRIPT_START)
  print(f'      evaluated in {time.time() - start:.1f}s')
  state['result'] = result
  stop = MANUSCRIPT_START + ROLLOUT_LENGTH
  np.testing.assert_array_equal(
      result['used_actions']['action'],
      actions['action'][:, MANUSCRIPT_START:stop])
  assert result['full'].shape[1] == ROLLOUT_LENGTH, result['full'].shape
  assert len(result['prefixes']) == len(dims), len(result['prefixes'])
  assert len(result['one_blocks']) == len(dims)
  print(f'      full rollout {result["full"].shape}, '
        f'{len(result["prefixes"])} prefixes, z {result["one_z"].shape}')
  assert before == _tree_hash(agent.save()['params']), (
      'evaluation modified model parameters')


@stage('matrices and figures')
def check_matrices(state, args):
  from corewm_eval.config import HORIZONS, MANUSCRIPT_START, ROLLOUT_LENGTH
  from corewm_eval import manuscript_figures as figures
  from corewm_eval.metrics import (
      cka_matrix, full_error_row, marginal_block_gain, marginal_visual_change,
      off_diagonal_mean, performance_gap, prefix_error_matrix)
  ep, result = state['clip'], state['result']
  dims = tuple(map(int, state['config'].agent.htp.proj.dims))
  stop = MANUSCRIPT_START + ROLLOUT_LENGTH
  gt = ep['image'][MANUSCRIPT_START + 1:stop + 1].astype(np.float32) / 255
  full_images = result['full'][0]
  prefix_images = [p[0] for p in result['prefixes']]
  prefix = prefix_error_matrix(prefix_images, gt, HORIZONS)
  full = full_error_row(full_images, gt, HORIZONS)
  gap = performance_gap(prefix, full)
  gain = marginal_block_gain(prefix)
  change = marginal_visual_change(prefix_images, HORIZONS)
  z = result['one_z']
  cka = cka_matrix([z[:, lo:hi] for lo, hi in zip((0, *dims[:-1]), dims)])
  assert prefix.shape == (len(dims), len(HORIZONS)), prefix.shape
  assert gap.shape == prefix.shape and gain.shape == (len(dims) - 1, len(HORIZONS))
  assert change.shape == gain.shape and cka.shape == (len(dims), len(dims))
  np.set_printoptions(precision=4, suppress=True)
  print(f'      full MAE      {full}')
  print(f'      prefix MAE\n{prefix}')
  print(f'      CKA off-diag  {off_diagonal_mean(cka):.4f}')
  out = Path(args.logdir) / 'figures'
  out.mkdir(parents=True, exist_ok=True)
  figures.decoded_matrix(out / 'decoded_matrix.png', gt, full_images, prefix_images)
  figures.mae_heatmap(out / 'mae.png', full, prefix)
  figures.gap_heatmap(out / 'gap.png', gap)
  figures.gain_heatmap(out / 'gain.png', gain)
  figures.change_heatmap(out / 'visual_change.png', change)
  figures.pixel_error_vs_ground_truth(out / 'pixel_error_gt.png', gt, full_images, prefix_images)
  figures.pixel_change_between_prefixes(out / 'pixel_change_prefix.png', prefix_images)
  figures.cka_heatmap(out / 'cka.png', cka)
  written = sorted(p.name for p in out.glob('*.png'))
  print(f'      wrote {len(written)} figures to {out}: {written}')
  assert len(written) == 8, written


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--logdir', type=Path, default=Path('/tmp/ms_smoke'))
  p.add_argument('--task', default='maniskill_PickCube-v1')
  p.add_argument('--configs', default='maniskill_rgb_eval_ready size1m corewm_full',
                 help='space-separated preset names, applied in order')
  p.add_argument('--checkpoint', help='optional checkpoint to load weights from')
  args = p.parse_args()
  args.logdir.mkdir(parents=True, exist_ok=True)
  print(f'logdir  {args.logdir}')
  print(f'task    {args.task}')
  print(f'configs {args.configs}\n')
  state = {}
  for index, (name, fn) in enumerate(STAGES, 1):
    print(f'[{index}/{len(STAGES)}] {name}')
    try:
      fn(state, args)
    except Exception:
      traceback.print_exc()
      print(f'\nFAIL at stage {index}: {name}')
      return 1
    print(f'      PASS\n')
  print(f'All {len(STAGES)} stages passed.')
  return 0


if __name__ == '__main__':
  sys.exit(main())
