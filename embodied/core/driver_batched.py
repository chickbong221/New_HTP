import elements
import numpy as np

from .driver import Driver


class BatchedDriver(Driver):
  """Driver for a single GPU-vectorized env that holds N sub-envs internally.

  Used by ManiSkill, where one embodied.Env instance wraps
  ManiSkillVectorEnv(num_envs=N). The env is stepped once with [N, ...]
  actions; callbacks are then fired N times (once per sub-env) so that
  replay.add() and logfn() receive single-env transitions exactly as they do
  from the normal Driver. The step counter advances by N to reflect true env
  throughput.

  reset(), on_step(), __call__() and _mask() are inherited unchanged.
  """

  def __init__(self, env, num_envs, **kwargs):
    self.batched = True
    self.parallel = False
    self._env = env
    self.act_space = env.act_space
    self.length = int(num_envs)
    self.kwargs = kwargs
    self.callbacks = []
    self.acts = None
    self.carry = None
    self.reset()

  def close(self):
    self._env.close()

  def _step(self, policy, step, episode):
    assert all(len(x) == self.length for x in self.acts.values())
    assert all(isinstance(v, np.ndarray) for v in self.acts.values())

    # Same accounting as Driver._step(): a paper interaction is an agent action
    # that advances the environment, and a reset-control call is not one even
    # though it is routed through the env.step() API. ManiSkill takes the reset
    # branch for the whole vector as soon as any sub-env asks for it, so this
    # is read off the outgoing reset mask rather than per sub-env physics.
    action_executed = ~np.asarray(self.acts['reset'], dtype=bool)

    # Single call returns batched obs [N, ...]
    obs = self._env.step(self.acts)
    obs = {k: np.asarray(v) for k, v in obs.items()}

    logs = {k: v for k, v in obs.items() if k.startswith('log/')}
    obs = {k: v for k, v in obs.items() if not k.startswith('log/')}

    assert all(len(x) == self.length for x in obs.values()), obs

    self.carry, acts, outs = policy(self.carry, obs, **self.kwargs)
    assert all(k not in acts for k in outs), (
        list(outs.keys()), list(acts.keys()))

    if obs['is_last'].any():
      mask = ~obs['is_last']
      acts = {k: self._mask(v, mask) for k, v in acts.items()}

    # Next step: envs whose episode ended get reset=True
    self.acts = {**acts, 'reset': obs['is_last'].copy()}

    # Fire one callback per sub-env so replay sees single transitions
    trans = {
        **obs, **acts, **outs, **logs,
        'log/action_executed': action_executed,
    }
    for i in range(self.length):
      trn = elements.tree.map(lambda x: x[i], trans)
      [fn(trn, i, **self.kwargs) for fn in self.callbacks]

    # Advance step counter by N (not 1) to reflect parallel throughput
    step += self.length
    episode += int(obs['is_last'].sum())
    return step, episode
