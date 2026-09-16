"""Clip collection for the manuscript block/prefix evaluation.

Suite-agnostic. Atari drives the scalar Driver; ManiSkill drives BatchedDriver
over one GPU sub-env, because the scalar Driver cannot step a vector env at
all -- it hands ManiSkill a 0-d reset mask and np.where() rejects it.

The recorded key set is read off the agent's obs_space rather than hard-coded.
The encoder indexes observations by its own spaces, so an env that carries an
extra encoder input (ManiSkill's proprioceptive `state`) has to be recorded
too or the evaluator raises KeyError at replay time.
"""

import numpy as np

from .config import CLIP_LENGTH, MIN_CLIP_LENGTH

# Driver bookkeeping, not task outcomes: never reported as a task metric.
BOOKKEEPING_LOGS = ('log/action_executed', 'log/raw_ale_frames')


def recorded_keys(agent):
  """Observation keys the evaluator has to replay, plus the action."""
  keys = {k for k in agent.obs_space if not k.startswith('log/')}
  return tuple(sorted(keys | {'action'}))


def make_driver(env):
  """BatchedDriver for a vector env, the scalar Driver otherwise."""
  import embodied
  if getattr(env, 'is_batched', False):
    if int(env.num_envs) != 1:
      raise ValueError(
          f'Clip collection needs num_envs=1, got {env.num_envs}. With more '
          'sub-envs the per-worker callbacks interleave and a single clip '
          'would mix trajectories from different environments.')
    return embodied.BatchedDriver(env, num_envs=1)
  return embodied.Driver([lambda: env], parallel=False)


def cut_at_first_terminal(values):
  """Truncate a recording just after its first is_last.

  Drivers keep stepping into the next episode, so this is what keeps a clip
  from silently joining two episodes. The terminal observation is retained.
  """
  terminals = np.flatnonzero(values['is_last'])
  cut = int(terminals[0]) + 1 if len(terminals) else len(values['is_last'])
  return {k: v[:cut] for k, v in values.items()}


def task_metrics(values, logs):
  """Task outcome of one clip, as far as the clip itself observed it.

  The first recorded frame is the reset frame (is_first, zero reward), so the
  episode is one transition shorter than the clip.
  """
  completed = bool(values['is_last'][-1])
  out = {
      'return': float(np.sum(values['reward'][1:])),
      'episode_length': int(len(values['is_last']) - 1),
      'episode_completed': completed,
  }
  if completed and logs:
    for key, value in logs[-1].items():
      if key not in BOOKKEEPING_LOGS:
        out[key[len('log/'):]] = float(value)
  return out


def collect_clips(agent, env, episodes, steps=CLIP_LENGTH,
                  minimum=MIN_CLIP_LENGTH, mode='eval'):
  """Drive `env` with the loaded policy and return `episodes` clips.

  Each clip begins at a genuine environment reset and is cut at the first
  is_last, so no clip crosses an episode boundary. Returns a list of
  (values, task_metrics) pairs; `values` maps each recorded key to an array
  whose leading axis is time.
  """
  if steps < minimum:
    raise ValueError(f'steps={steps} cannot yield {minimum} frames')
  keys = recorded_keys(agent)
  driver = make_driver(env)
  frames, logs = [], []

  def record(tran, worker):
    if worker != 0:
      return
    frames.append({k: np.asarray(tran[k]).copy() for k in keys})
    logs.append({k: float(np.asarray(v)) for k, v in tran.items()
                 if k.startswith('log/') and np.asarray(v).ndim == 0})

  driver.on_step(record)
  policy = lambda *xs: agent.policy(*xs, mode=mode)
  clips = []
  try:
    for episode in range(episodes):
      frames.clear()
      logs.clear()
      driver.reset(agent.init_policy)
      driver(policy, steps=steps)
      values = {k: np.stack([frame[k] for frame in frames]) for k in keys}
      if not bool(values['is_first'][0]):
        raise RuntimeError(
            f'Clip {episode} does not start at an environment reset')
      values = cut_at_first_terminal(values)
      cut = len(values['is_last'])
      if cut < minimum:
        raise RuntimeError(
            f'Clip {episode} has {cut} frames, need {minimum}. The episode '
            'horizon is shorter than the evaluation protocol; raise '
            'env.maniskill.max_episode_steps or pick a task with a longer '
            'registered horizon.')
      clips.append((values, task_metrics(values, logs[:cut])))
  finally:
    driver.close()
  return clips
