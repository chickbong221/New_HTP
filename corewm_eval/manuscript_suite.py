"""Evaluate trained checkpoints; all production computation belongs in Slurm.

Protocol constants live in corewm_eval/config.py and are not re-declared here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from .config import (
    HORIZONS, MANUSCRIPT_START, MIN_CLIP_LENGTH, ROLLOUT_LENGTH)

ROOT = Path('/home/vn-user0101/Dat/HTS-Dreamer/production_runs/corewm_atari100k_v1')
# Rollout starts per checkpoint. Each is one clip from the seed-0 policy.
EPISODES = 4


def dump(path, data):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(data, indent=2, allow_nan=False))


def digest(path):
  h = hashlib.sha256()
  with path.open('rb') as f:
    for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
  return h.hexdigest()


def source(game, seed):
  """Atari100k checkpoint layout produced by run.exact_env_action_budget."""
  if game == 'alien' and seed == 0:
    candidates = list((ROOT / 'training/wave1/full/alien/seed_0').glob('**/ckpt/env_action_steps_000100000'))
  else:
    candidates = list((ROOT / f'training/stage1_full/{game}/seed_{seed}').glob('*/ckpt/env_action_steps_000100000'))
  if len(candidates) != 1:
    raise RuntimeError(f'Ambiguous checkpoint {game}/{seed}: {candidates}')
  return candidates[0]


def resolve_checkpoint(path):
  """Return the directory that actually holds the checkpoint entries.

  elements.Checkpoint writes each save into a timestamped subfolder of its
  directory and records the current one in `latest`, so <logdir>/ckpt is the
  natural thing to pass but is one level above what load() wants. An Atari
  milestone path already points at the entries and is returned unchanged.
  """
  path = Path(path)
  if (path / 'agent.pkl').exists():
    return path
  latest = path / 'latest'
  if latest.is_file():
    candidate = path / latest.read_text().strip()
    if (candidate / 'agent.pkl').exists():
      return candidate
  saves = sorted(p for p in path.glob('*') if (p / 'agent.pkl').is_file())
  if len(saves) == 1:
    return saves[0]
  if saves:
    # Never guess which save a published number came from.
    raise RuntimeError(
        f'{path} holds {len(saves)} checkpoints and no usable `latest`. Pass '
        f'one explicitly: {[str(p) for p in saves]}')
  raise RuntimeError(f'No agent.pkl in {path} or any of its subdirectories')


def config_path(checkpoint):
  """Find the run config for a checkpoint directory.

  An Atari milestone checkpoint sits at <logdir>/ckpt/env_action_steps_*,
  while a run without exact milestones (every ManiSkill run) checkpoints to
  <logdir>/ckpt directly, so the depth below the logdir is not fixed.
  """
  for folder in (checkpoint, *checkpoint.parents):
    candidate = folder / 'config.yaml'
    if candidate.exists():
      return candidate
  raise RuntimeError(f'No config.yaml at or above {checkpoint}')


def one_step_positions(available):
  """Absolute timesteps shown in the qualitative block-only figure."""
  wanted = (0, MANUSCRIPT_START // 2, MANUSCRIPT_START,
            MANUSCRIPT_START + ROLLOUT_LENGTH)
  return [p for p in wanted if p < int(available)]


def worker(out, game, seed, checkpoint=None, config_file=None):
  import elements
  from dreamerv3 import main_htp
  from .smoke_test import _load_config, _tree_hash
  from .extraction import (
      ExtractionSpec, assert_rollout_actions, evaluation_seed,
      previous_actions)
  from .manuscript_clips import collect_clips
  from .manuscript_eval import evaluate_readonly
  from .metrics import (
      cka_matrix, full_error_row, marginal_block_gain, marginal_visual_change,
      off_diagonal_mean, performance_gap, prefix_error_matrix)
  from . import manuscript_figures as figures
  cp = resolve_checkpoint(checkpoint) if checkpoint else source(game, seed)
  print(f'Checkpoint: {cp}')
  # A checkpoint kept outside its log tree has no config.yaml above it to find.
  config = _load_config(Path(config_file) if config_file else config_path(cp))
  config = config.update(logdir=str(out / game / f'seed_{seed}/runtime'))
  agent = main_htp.make_agent(config)
  # make_agent returns freshly initialised weights. Nothing downstream fails
  # if the checkpoint does not reach them -- the evaluation just decodes an
  # untrained model into plausible-looking noise -- so compare against the
  # initialisation rather than trusting the load.
  initialised = _tree_hash(agent.save()['params'])
  loaded = elements.Checkpoint(); loaded.agent = agent; loaded.load(cp, keys=['agent'])
  before = _tree_hash(agent.save()['params']); updates = int(agent.n_updates)
  if before == initialised:
    raise RuntimeError(
        f'Loading {cp} left every parameter at its initial value, so the '
        'checkpoint never reached the agent. The figures would show an '
        'untrained decoder.')
  print(f'Loaded {updates} optimizer updates from the checkpoint')
  if updates == 0:
    raise RuntimeError(
        f'{cp} restored 0 optimizer updates: it holds an untrained agent.')
  sha = digest(cp / 'agent.pkl')
  suite = str(config.task).split('_', 1)[0]
  dyn = config.agent.dyn[config.agent.dyn.typ]
  dims = tuple(map(int, config.agent.htp.proj.dims))
  spec = ExtractionSpec(int(dyn.deter), (int(dyn.stoch), int(dyn.classes)), dims)
  destination = out / game / f'seed_{seed}'; destination.mkdir(parents=True, exist_ok=True)
  data = out / game / 'clips'; data.mkdir(parents=True, exist_ok=True)
  # Seed-0 source policy creates the predeclared early-episode clips shared by
  # every checkpoint of this game. Each begins at a genuine environment reset.
  if seed == 0:
    # A vector env must be narrowed to one sub-env: the per-worker callbacks
    # of a wider one would interleave into a single clip.
    overrides = {'num_envs': 1} if suite == 'maniskill' else {}
    env = main_htp.make_env(config, 0, use_seed=False, seed=260909, **overrides)
    clips = collect_clips(agent, env, EPISODES)
    for episode, (values, metrics) in enumerate(clips):
      np.savez_compressed(
          data / f'episode_{episode}.npz',
          **{k: values[k] for k in sorted(values)})
      dump(data / f'episode_{episode}_task.json', metrics)
  all_z, prefix_rows, full_rows, gap_rows, gain_rows, change_rows = [], [], [], [], [], []
  for episode in range(EPISODES):
    with np.load(data / f'episode_{episode}.npz') as f: ep = dict(f)
    obs = {k: v[None] for k, v in ep.items() if k != 'action'}
    actions = {'action': ep['action'][None]}
    result = evaluate_readonly(
        agent, obs, {'action': previous_actions(ep['action'])[None]}, actions,
        spec, evaluation_seed(0, episode, MANUSCRIPT_START), MANUSCRIPT_START)
    stop = MANUSCRIPT_START + ROLLOUT_LENGTH
    assert_rollout_actions(
        result['used_actions']['action'],
        actions['action'][:, MANUSCRIPT_START:stop])
    # Horizon h is scored against frame T + h, so the target window starts at
    # T + 1 and every representation at h sees the same ground truth.
    gt = ep['image'][MANUSCRIPT_START + 1:stop + 1].astype(np.float32) / 255
    full_images = result['full'][0]
    prefix_images = [p[0] for p in result['prefixes']]
    prefix = prefix_error_matrix(prefix_images, gt, HORIZONS)
    full = full_error_row(full_images, gt, HORIZONS)
    prefix_rows.append(prefix); full_rows.append(full)
    gap_rows.append(performance_gap(prefix, full))
    gain_rows.append(marginal_block_gain(prefix))
    change_rows.append(marginal_visual_change(prefix_images, HORIZONS))
    all_z.append(result['one_z'])
    if episode == 0:
      figures.decoded_matrix(
          destination / 'decoded_matrix.png', gt, full_images, prefix_images,
          posterior=result['posterior'][0])
      figures.pixel_error_vs_ground_truth(
          destination / 'pixel_error_gt.png', gt, full_images, prefix_images)
      figures.pixel_change_between_prefixes(
          destination / 'pixel_change_prefix.png', prefix_images)
      positions = one_step_positions(len(result['one_full']))
      figures.one_step_blocks(
          destination / 'one_step_blocks.png',
          ep['image'][np.array(positions) + 1] / 255,
          result['one_full'][positions],
          [b[positions] for b in result['one_blocks']], positions)
  z = np.concatenate(all_z)
  cka = cka_matrix([z[:, lo:hi] for lo, hi in zip((0, *dims[:-1]), dims)])
  prefix_mae = np.mean(prefix_rows, axis=0); full_mae = np.mean(full_rows, axis=0)
  gap = np.mean(gap_rows, axis=0); gain = np.mean(gain_rows, axis=0)
  change = np.mean(change_rows, axis=0)
  figures.mae_heatmap(destination / 'mae.png', full_mae, prefix_mae)
  figures.gap_heatmap(destination / 'gap.png', gap)
  figures.gain_heatmap(destination / 'gain.png', gain)
  figures.change_heatmap(destination / 'visual_change.png', change)
  figures.cka_heatmap(destination / 'cka.png', cka)
  assert before == _tree_hash(agent.save()['params']) and updates == int(agent.n_updates)
  assert sha == digest(cp / 'agent.pkl')
  task = [json.loads(p.read_text())
          for p in sorted(data.glob('*_task.json'))]
  dump(destination / 'result.json', {
      'game': game, 'seed': seed, 'task': str(config.task), 'suite': suite,
      'checkpoint': str(cp), 'checkpoint_sha256': sha,
      'protocol': {
          'start': MANUSCRIPT_START, 'rollout_length': ROLLOUT_LENGTH,
          'horizons': list(HORIZONS), 'min_clip_length': MIN_CLIP_LENGTH,
          'number_of_prefixes': len(dims)},
      'horizons': list(HORIZONS),
      'prefix_dimensions': list(dims),
      'full_mae': full_mae.tolist(),
      'prefix_mae': prefix_mae.tolist(),
      'performance_gap': gap.tolist(),
      'marginal_block_gain': gain.tolist(),
      'marginal_visual_change': change.tolist(),
      'cka': cka.tolist(),
      'cka_off_diagonal_mean': off_diagonal_mean(cka),
      'number_of_rollout_starts': EPISODES,
      'one_step_samples': len(z),
      'clip_sha256': {p.name: digest(p) for p in sorted(data.glob('*.npz'))},
      'parameters_unchanged': True,
      'per_episode': {
          'full_mae': np.asarray(full_rows).tolist(),
          'prefix_mae': np.asarray(prefix_rows).tolist(),
          'performance_gap': np.asarray(gap_rows).tolist(),
          'marginal_block_gain': np.asarray(gain_rows).tolist(),
          'marginal_visual_change': np.asarray(change_rows).tolist()},
      'task_metrics': task})


# --------------------------------------------------------------------------
#  Reporting
# --------------------------------------------------------------------------

FIGURES = ('decoded_matrix', 'mae', 'gap', 'gain', 'visual_change', 'cka',
           'one_step_blocks', 'pixel_error_gt', 'pixel_change_prefix')


def _table(rows, values, horizons=HORIZONS):
  head = '| | ' + ' | '.join(f'h={h}' for h in horizons) + ' |'
  rule = '|---|' + '---:|' * len(horizons)
  body = ['| ' + label + ' | ' + ' | '.join(f'{v:.4f}' for v in row) + ' |'
          for label, row in zip(rows, np.asarray(values))]
  return [head, rule, *body]


def _load_results(out, name, seeds):
  paths = [out / name / f'seed_{seed}/result.json' for seed in seeds]
  if not all(p.exists() for p in paths):
    return None
  return [json.loads(p.read_text()) for p in paths]


def _mean(results, key):
  return np.mean([np.asarray(r[key]) for r in results], axis=0)


def _section(out, name, results):
  """Matrices averaged over seeds, plus the seed-0 figures."""
  from .manuscript_figures import added_block_labels, prefix_labels
  dims = results[0]['prefix_dimensions']
  prefix = _mean(results, 'prefix_mae'); full = _mean(results, 'full_mae')
  lines = [f'### {name}', '',
           f'Prefix dimensions: {dims}. Seeds averaged: {len(results)}.', '',
           '**Pixel MAE** (rows Full, P1..P%d)' % len(dims), '',
           *_table(['Full'] + prefix_labels(len(dims)),
                   np.vstack([full[None, :], prefix])), '',
           '**Performance gap** (prefix MAE - full MAE; positive is worse)', '',
           *_table(prefix_labels(len(dims)), _mean(results, 'performance_gap')), '',
           '**Marginal block gain** (E_{l-1} - E_l; positive means the added '
           'block helps)', '',
           *_table(added_block_labels(len(dims) - 1),
                   _mean(results, 'marginal_block_gain')), '',
           '**Marginal visual change** (|I_l - I_{l-1}|)', '',
           *_table(added_block_labels(len(dims) - 1),
                   _mean(results, 'marginal_visual_change')), '',
           'Mean off-diagonal CKA: '
           f'{np.mean([r["cka_off_diagonal_mean"] for r in results]):.4f}', '']
  task = [m for r in results for m in r.get('task_metrics', [])]
  if task:
    keys = sorted({k for m in task for k in m if k != 'episode_completed'})
    lines += ['**Task metrics over the evaluation clips** '
              f'(n={len(task)} episodes)', '',
              '| Metric | Mean |', '|---|---:|',
              *[f'| {k} | {np.mean([m[k] for m in task if k in m]):.4f} |'
                for k in keys], '']
  lines += [f'![{name} {fig}]({name}/seed_0/{fig}.png)' for fig in FIGURES]
  return lines + ['']


def _preamble():
  return [
      f'Protocol: posterior warm-up to t={MANUSCRIPT_START}, then a '
      f'{ROLLOUT_LENGTH}-action open-loop RSSM rollout on the ground-truth '
      f'action sequence. Reported horizons {list(HORIZONS)}; horizon h is '
      'scored against ground-truth frame T+h, and every representation at a '
      'horizon is compared against the same frame.', '',
      'All prefixes share one imagined latent trajectory; a reconstruction is '
      'never fed back into the dynamics, and the dedicated prefix-dynamics '
      'predictors are not invoked. Prefix images come from the cumulative '
      'reconstruction sum decoded through the shared observation decoder. '
      'Block-only images are the difference of two cumulative '
      'reconstructions, which equals D_l(z_l) exactly and avoids the bias a '
      'masked head would introduce.', '',
      'CKA is computed on non-overlapping blocks of z from independent '
      'one-step predictions, and is a secondary redundancy diagnostic rather '
      'than the primary evidence. Prefix L is not assumed to equal the full '
      'RSSM feature; the performance gap measures that difference directly.',
      '']


def report(out):
  """Atari100k report: HNS policy table plus the manuscript matrices."""
  from .full_vs_dreamerv3_curves import load_full_policy_curves
  from .hns_reference import load_reference
  frame, sources = load_full_policy_curves(ROOT / 'evaluations')
  refs = load_reference('baselines.yaml', 'atari57_gamer')
  lines = ['# CoRe-WM: 26 games x 5 seeds', '',
           'Policy: checkpoint at 100k actions, 100 evaluation episodes per '
           'seed; trapezoidal AUC over 10k-100k divided by 90k. HNS is not '
           'clipped; mean/median are taken after averaging 5 seeds per game.',
           '', *_preamble(),
           '| Game | Return mean +- SD (5 seeds) | HNS % | AUC HNS % | '
           'CKA off-diag | Full MAE @16 | P1 gap @16 | P5 gap @16 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
  hnss = []
  for game in sorted(refs):
    random, human = refs[game]; sub = frame[frame.game == game]
    final = sub[sub.agent_steps == 100000]['return'].to_numpy()
    hns = (final.mean() - random) / (human - random); hnss.append(hns)
    auc = []
    for seed in range(5):
      curve = sub[sub.seed == seed].sort_values('agent_steps')
      auc.append(np.trapz((curve['return'].to_numpy()-random)/(human-random), curve.agent_steps.to_numpy()) / 90000)
    results = _load_results(out, game, range(5))
    if results:
      gap = _mean(results, 'performance_gap')
      extra = ' | '.join(f'{v:.4f}' for v in (
          np.mean([r['cka_off_diagonal_mean'] for r in results]),
          _mean(results, 'full_mae')[-1], gap[0][-1], gap[-1][-1]))
    else:
      extra = 'pending | pending | pending | pending'
    lines.append(f'| {game} | {final.mean():.2f} +- {final.std(ddof=1):.2f} | {100*hns:.2f} | {100*np.mean(auc):.2f} | {extra} |')
  lines += ['', f'Mean HNS: {100*np.mean(hnss):.2f}%; median HNS: {100*np.median(hnss):.2f}%; games above human: {sum(x>1 for x in hnss)}/26.', '',
            '## Per-game representation results', '']
  for game in sorted(refs):
    results = _load_results(out, game, range(5))
    if results:
      lines += _section(out, game, results)
  (out / 'README.md').write_text('\n'.join(lines) + '\n')
  dump(out / 'policy_sources.json', sources)


def report_maniskill(out):
  """ManiSkill report: no HNS reference; task metrics stay separate."""
  names = sorted(p.name for p in out.iterdir()
                 if p.is_dir() and (p / 'seed_0/result.json').exists())
  lines = ['# CoRe-WM: ManiSkill block and prefix evaluation', '',
           *_preamble(),
           'ManiSkill task metrics (success rate, return, episode length, '
           'success_once, success_at_end) are reported separately below, per '
           'target, and are measured over the evaluation clips only -- they '
           'are a diagnostic, not a full policy evaluation.', '',
           '| Target | Task | CKA off-diag | Full MAE @16 | P1 gap @16 | '
           'P5 gap @16 |', '|---|---|---:|---:|---:|---:|']
  for name in names:
    results = _load_results(out, name, [0])
    gap = _mean(results, 'performance_gap')
    lines.append(
        f'| {name} | {results[0]["task"]} | '
        f'{np.mean([r["cka_off_diagonal_mean"] for r in results]):.4f} | '
        f'{_mean(results, "full_mae")[-1]:.4f} | '
        f'{gap[0][-1]:.4f} | {gap[-1][-1]:.4f} |')
  lines += ['', '## Per-target representation results', '']
  for name in names:
    lines += _section(out, name, _load_results(out, name, [0]))
  (out / 'README.md').write_text('\n'.join(lines) + '\n')


def main():
  p = argparse.ArgumentParser()
  p.add_argument('out', type=Path)
  p.add_argument('--game', help='output subdirectory; the Atari game name, '
                                'or any label for an explicit --checkpoint')
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--checkpoint', help='explicit checkpoint directory; skips '
                                      'the Atari production layout lookup')
  p.add_argument('--config', help='run config.yaml; needed when the '
                                  'checkpoint lives outside its log tree')
  p.add_argument('--report', action='store_true')
  p.add_argument('--maniskill', action='store_true',
                 help='write the ManiSkill report instead of the Atari one')
  args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
  if args.report:
    report_maniskill(args.out) if args.maniskill else report(args.out)
  elif args.game:
    worker(args.out, args.game, args.seed, args.checkpoint, args.config)
  else:
    # Only the unattended 130-checkpoint sweep is Slurm-only; a single
    # checkpoint is also run as a Slurm subprocess from here, and is allowed
    # to run directly for development.
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Slurm required')
    from .config import ATARI100K_GAMES
    from concurrent.futures import ThreadPoolExecutor
    report(args.out)
    cpus = sorted(os.sched_getaffinity(0))
    devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if len(cpus) < 32 or len(devices) != 2:
      raise RuntimeError('Expected two GPUs and 32 CPUs')
    def slot(index):
      env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices[index//2])
      for game in ATARI100K_GAMES[index::4]:
        for seed in range(5):
          target = args.out / game / f'seed_{seed}/result.json'
          if target.exists(): continue
          print(f'START {game} seed={seed} slot={index}', flush=True)
          with (args.out / f'{game}_{seed}.log').open('w') as log:
            subprocess.run(['taskset', '-c', ','.join(map(str, cpus[8*index:8*index+8])),
                sys.executable, '-u', '-m', 'corewm_eval.manuscript_suite', str(args.out),
                '--game', game, '--seed', str(seed)], env=env,
                stdout=log, stderr=subprocess.STDOUT, check=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
      list(pool.map(slot, range(4)))
    report(args.out)
    dump(args.out / 'status.json', {'status': 'COMPLETE', 'checkpoints': 130})


if __name__ == '__main__': main()
