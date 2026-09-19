#!/usr/bin/env python3
"""Find out why the manuscript evaluation decodes noise from a good checkpoint.

The symptom this exists for: `corewm_eval.manuscript_suite` produced a decoded
matrix that is structureless colour noise at every horizon, with a pixel MAE
flat at ~0.167 across h=1..16 and across Full and P1..P5, while the same run's
W&B `report/openloop/image` shows the model reconstructing and imagining the
same task sharply. A flat MAE means the decoded frame carries no information
about the latent at all, so the question is not "how good is the world model"
but "does the evaluation harness actually run the trained weights through the
trained decoder".

Four checks, cheapest first; each prints its own verdict and none depends on
the ones after it.

  1. load     -- does the checkpoint reach the agent, per module prefix?
  2. keys     -- does the read-only pure call ask ninjax for any parameter the
                 checkpoint does not carry? `nj.pure(..., create=False,
                 ignore=True)`, which is what `evaluate_readonly` uses, does
                 not raise on a missing parameter: the module builds a fresh
                 random one and the returned state still has the original key
                 set, so the existing `set(returned) != set(params)` assertion
                 cannot see it. Running the same function with create=True and
                 ignore=False leaves every such parameter in the returned
                 state, which is what this compares against.
  3. control  -- the same evaluation with the freshly initialised parameters
                 instead of the checkpoint's. If the two MAEs agree, the
                 loaded weights are provably not reaching the decode.
  4. report   -- the agent's own `report()` path, the one W&B draws, on the
                 very same clip and with the same 16 observed + 16 imagined
                 split. Sharp here and noise in check 3 localises the fault to
                 the read-only harness rather than the checkpoint.

Usage on the cluster, reusing the clips the suite already wrote:

  python scripts/diagnose_maniskill_eval.py out/diagnosis \
      --checkpoint $CKPT_PATH --config ${CKPT_PATH}_config.yaml \
      --clips $EVAL_DIR/PullCubeTool-v1/clips

Without --clips it collects one clip itself, which needs the ManiSkill env.
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

# Sets XLA_PYTHON_CLIENT_PREALLOCATE before JAX initialises.
from dreamerv3 import main_maniskill  # noqa: E402,F401


def _rule(text):
  print('\n=== ' + text + ' ' + '=' * max(0, 66 - len(text)), flush=True)


def _prefix(key):
  return key.split('/', 1)[0]


def check_load(agent, checkpoint):
  """Compare the checkpoint against the initialisation, module by module."""
  import elements
  before = {k: np.asarray(v).copy() for k, v in agent.save()['params'].items()}
  loaded = elements.Checkpoint()
  loaded.agent = agent
  loaded.load(checkpoint, keys=['agent'])
  after = agent.save()['params']
  print(f'optimizer updates restored: {int(agent.n_updates)}')
  changed_any = False
  for group in sorted({_prefix(k) for k in before}):
    keys = [k for k in before if _prefix(k) == group]
    changed = [k for k in keys
               if not np.array_equal(before[k], np.asarray(after[k]))]
    changed_any = changed_any or bool(changed)
    print(f'  {group:12s} {len(changed):4d}/{len(keys):4d} '
          f'tensors differ from init')
  if not changed_any:
    print('VERDICT: the checkpoint did not reach a single parameter.')
  elif int(agent.n_updates) == 0:
    print('VERDICT: parameters moved but the update counter is 0, so the '
          'checkpoint holds an untrained agent.')
  else:
    print('VERDICT: the checkpoint loaded.')
  return before


def check_keys(agent, obs, prevact, actions, spec, seed, start):
  """List parameters the evaluation asks for that the checkpoint lacks.

  `evaluate_readonly` runs with create=False and ignore=True. Under those
  flags a lookup that misses does not raise: the module constructs the
  parameter from its initialiser and the discarded creation never appears in
  the returned state. The same call with create=True and ignore=False leaves
  those keys behind, so a non-empty difference names exactly the weights the
  production evaluation invents fresh on every call.
  """
  import jax
  import ninjax as nj
  from corewm_eval.manuscript_eval import _evaluate
  params = agent.save()['params']
  pure = nj.pure(lambda: _evaluate(
      agent.model, obs, prevact, actions, spec, start))
  with jax._src.config.explicit_device_put_scope():
    returned, _ = pure(dict(params), seed=np.asarray(seed, np.uint32),
                       create=True, modify=True, ignore=False)
  missing = sorted(set(returned) - set(params))
  print(f'parameters in the checkpoint: {len(params)}')
  print('parameters the evaluation touched but the checkpoint lacks: '
        f'{len(missing)}')
  for key in missing[:40]:
    print(f'  {key}')
  if len(missing) > 40:
    print(f'  ... and {len(missing) - 40} more')
  if missing:
    print('VERDICT: every key above is silently re-initialised at random on '
          'each production evaluation call.')
  else:
    print('VERDICT: the evaluation runs entirely on checkpoint parameters.')
  return missing


def _pure_evaluate(agent, params, obs, prevact, actions, spec, seed, start):
  """The production read-only call, but on parameters chosen by the caller."""
  import jax
  import ninjax as nj
  from corewm_eval.manuscript_eval import _evaluate
  pure = nj.pure(lambda: _evaluate(
      agent.model, obs, prevact, actions, spec, start))
  with jax._src.config.explicit_device_put_scope():
    _, result = pure(params, seed=np.asarray(seed, np.uint32),
                     create=False, modify=True, ignore=True)
  with jax._src.config.explicit_device_get_scope():
    return jax.tree.map(np.asarray, result)


def check_control(agent, initial_params, obs, prevact, actions, spec, seed,
                  start, gt, out):
  """Evaluate twice: with the checkpoint, and with the initialisation.

  A trained decoder driven by an off-distribution latent still paints the
  training manifold -- wood floor, grey wall, robot-shaped blobs. Fine pixel
  noise is what an untrained decoder produces. If the two rows below agree,
  the checkpoint is not what the decode ran on, whatever check 1 reported.
  """
  from corewm_eval.config import ROLLOUT_LENGTH
  from corewm_eval import manuscript_figures as figures
  rows = {}
  for name, params in (('checkpoint', agent.save()['params']),
                       ('initialisation', initial_params)):
    result = _pure_evaluate(
        agent, dict(params), obs, prevact, actions, spec, seed, start)
    full, post = result['full'][0], result['posterior'][0]
    rows[name] = {
        'posterior_mae': float(np.abs(post - gt).mean()),
        'open_loop_mae': float(np.abs(full - gt).mean()),
        'pred_std': float(np.std(full)),
    }
    figures.decoded_matrix(
        out / f'decoded_{name}.png', gt, full,
        [p[0] for p in result['prefixes']], posterior=post)
  constant = float(np.abs(0.5 - gt).mean())
  print(f'{"source":16s} {"posterior MAE":>14s} {"open-loop MAE":>14s} '
        f'{"pred std":>10s}')
  for name, row in rows.items():
    print(f'{name:16s} {row["posterior_mae"]:14.4f} '
          f'{row["open_loop_mae"]:14.4f} {row["pred_std"]:10.4f}')
  print(f'{"constant 0.5":16s} {constant:14.4f} {constant:14.4f}')
  print(f'(rollout length {ROLLOUT_LENGTH}; figures written to {out})')
  gap = abs(rows['checkpoint']['open_loop_mae']
            - rows['initialisation']['open_loop_mae'])
  if gap < 0.01:
    print('VERDICT: the checkpoint and the initialisation decode equally '
          'badly, so the loaded weights never reach this code path.')
  elif rows['checkpoint']['posterior_mae'] < 0.03:
    print('VERDICT: the decode path is healthy, the posterior column is '
          'sharp, and what is left is genuine open-loop prediction error.')
  else:
    print('VERDICT: the checkpoint decodes better than the initialisation but '
          'the posterior is still blurred; suspect the observation fed to the '
          'encoder rather than the parameters.')
  return rows


def check_report(agent, clip, length, out):
  """Run the agent's own report path on the same clip.

  This is the code that draws the W&B openloop panel: first half observed,
  second half imagined, true over prediction over error. The agent was built
  with replay_context disabled, so its carry starts at zeros exactly like the
  manuscript protocol does.
  """
  from corewm_eval import manuscript_figures as figures
  data = agent._zeros(agent.spaces, (1, length))
  for key in agent.obs_space:
    if key in clip:
      data[key] = np.asarray(clip[key][:length])[None]
  data['action'] = np.asarray(clip['action'][:length])[None]
  data['seed'] = agent._seeds(0, agent.train_mirrored)
  carry = agent.init_report(1)
  _, metrics = agent.report(carry, data)
  grid = metrics.get('openloop/image')
  if grid is None:
    print('VERDICT: report() returned no openloop panel. Is agent.report '
          'enabled in this config?')
    return
  grid = np.asarray(grid)
  plt = figures._pyplot()
  fig, axes = plt.subplots(len(grid), 1, figsize=(6, 1.2 * len(grid)))
  for axis, frame in zip(np.atleast_1d(axes), grid):
    axis.imshow(np.asarray(frame, np.uint8))
    axis.axis('off')
  fig.tight_layout()
  fig.savefig(out / 'report_openloop.png', dpi=140)
  plt.close(fig)
  print(f'wrote {out / "report_openloop.png"} ({len(grid)} frames; within '
        'each frame the rows are true / prediction / error)')
  print('VERDICT: compare it against decoded_checkpoint.png from check 3. '
        'Sharp here and noise there puts the fault in the read-only harness; '
        'sharp in both means neither the checkpoint nor either decode path is '
        'what produced the published figures, and the clip is what differs.')


def load_clip(args, agent, config):
  if args.clips:
    path = Path(args.clips) / 'episode_0.npz'
    with np.load(path) as f:
      clip = dict(f)
    print(f'clip: {path} ({len(clip["image"])} frames)')
    return clip
  from dreamerv3 import main_htp
  from corewm_eval.manuscript_clips import collect_clips
  env = main_htp.make_env(config, 0, use_seed=False, seed=260909, num_envs=1)
  (values, metrics), = collect_clips(agent, env, 1)
  print(f'clip: collected {len(values["image"])} frames, {metrics}')
  return values


def main():
  p = argparse.ArgumentParser()
  p.add_argument('out', type=Path)
  p.add_argument('--checkpoint', required=True)
  p.add_argument('--config', required=True)
  p.add_argument('--clips', help='clips folder from a suite run, to reuse the '
                                 'exact episodes the bad figures came from')
  p.add_argument('--skip-report', action='store_true')
  args = p.parse_args()
  args.out.mkdir(parents=True, exist_ok=True)

  from dreamerv3 import main_htp
  from corewm_eval.config import (
      EVALUATION_SEED, MANUSCRIPT_START, ROLLOUT_LENGTH)
  from corewm_eval.extraction import (
      ExtractionSpec, evaluation_seed, previous_actions)
  from corewm_eval.manuscript_suite import resolve_checkpoint
  from corewm_eval.smoke_test import _load_config

  checkpoint = resolve_checkpoint(args.checkpoint)
  config = _load_config(Path(args.config))
  config = config.update(logdir=str(args.out / 'runtime'))
  # check_report needs a zeros carry and plain obs tensors; with a replay
  # context the report path would instead spend the first K frames restoring a
  # carry from entries this clip does not record. Parameter shapes do not
  # depend on it, so the checkpoint still loads.
  config = config.update(replay_context=0)
  agent = main_htp.make_agent(config)

  _rule('1. checkpoint load')
  initial_params = check_load(agent, checkpoint)

  clip = load_clip(args, agent, config)
  needed = MANUSCRIPT_START + ROLLOUT_LENGTH + 1
  if len(clip['image']) < needed:
    raise SystemExit(f'clip has {len(clip["image"])} frames, need {needed}')
  obs = {k: v[None] for k, v in clip.items() if k != 'action'}
  actions = {'action': clip['action'][None]}
  prevact = {'action': previous_actions(clip['action'])[None]}
  dyn = config.agent.dyn[config.agent.dyn.typ]
  spec = ExtractionSpec(
      int(dyn.deter), (int(dyn.stoch), int(dyn.classes)),
      tuple(map(int, config.agent.htp.proj.dims)))
  seed = evaluation_seed(EVALUATION_SEED, 0, MANUSCRIPT_START)
  stop = MANUSCRIPT_START + ROLLOUT_LENGTH
  gt = clip['image'][MANUSCRIPT_START + 1:stop + 1].astype(np.float32) / 255

  _rule('2. parameters the evaluation asks for but the checkpoint lacks')
  check_keys(agent, obs, prevact, actions, spec, seed, MANUSCRIPT_START)

  _rule('3. checkpoint versus initialisation on the same clip')
  check_control(agent, initial_params, obs, prevact, actions, spec, seed,
                MANUSCRIPT_START, gt, args.out)

  if not args.skip_report:
    _rule('4. the report path W&B draws, on the same clip')
    # report() always splits its window in half, so it cannot mirror a
    # protocol with no warm-up. Give it the whole clip and read it as a
    # qualitative check on the decode rather than as the same measurement.
    check_report(agent, clip, len(clip['image']) // 2 * 2, args.out)


if __name__ == '__main__':
  main()
