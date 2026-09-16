"""Read-only block and prefix evaluation using standard RSSM dynamics."""
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from .config import MANUSCRIPT_START, ROLLOUT_LENGTH
from .extraction import _flat_to_feat


def residual_blocks(cumulative):
  """Recover D_l(z_l), without introducing other heads' zero-input biases."""
  return tuple(x if i == 0 else x - cumulative[i - 1]
               for i, x in enumerate(cumulative))


def _evaluate(model, obs, prevact, actions, spec, start):
  batch = obs['is_first'].shape[0]
  _, _, tokens = model.enc(model.enc.initial(batch), obs, obs['is_first'], training=False)
  _, posterior, _ = model.dyn.observe(model.dyn.initial(batch), tokens, prevact,
                                     obs['is_first'], training=False)
  # Flatten posterior times into independent batch entries: exactly one step
  # from each posterior, with outgoing a_t and target observation o_{t+1}.
  states = {k: posterior[k][:, :-1].reshape((-1, *posterior[k].shape[2:]))
            for k in ('deter', 'stoch')}
  acts = {k: v[:, :-1].reshape((-1, *v.shape[2:])) for k, v in actions.items()}
  _, (one, _) = model.dyn.imagine(states, acts, 1, training=False, single=True)
  one = jax.tree.map(lambda x: x[:, None], one)
  def decode(h):
    _, _, dist = model.dec(model.dec.initial(h.shape[0]), _flat_to_feat(h, spec),
                           jnp.zeros(h.shape[:2], bool), training=False)
    return dist['image'].pred()
  one_h = model.feat2h(one)
  one_z = model.htp_proj(one_h)
  cumulative = model.htp_recon.reconstruct(one_z, one_h)
  isolated = residual_blocks(cumulative)
  first = {k: posterior[k][:, start] for k in ('deter', 'stoch')}
  rollout_actions = {
      k: v[:, start:start + ROLLOUT_LENGTH] for k, v in actions.items()}
  _, future, used = model.dyn.imagine(
      first, rollout_actions, ROLLOUT_LENGTH, training=False)
  future_h = model.feat2h(future)
  future_z = model.htp_proj(future_h)
  prefix_h = model.htp_recon.reconstruct(future_z, future_h)
  return {'one_z': one_z[:, 0], 'one_full': decode(one_h)[:, 0],
          'one_blocks': tuple(decode(x)[:, 0] for x in isolated),
          'full': decode(future_h), 'prefixes': tuple(decode(x) for x in prefix_h),
          'used_actions': used}


def evaluate_readonly(agent, obs, prevact, actions, spec, seed,
                      start=MANUSCRIPT_START):
  params = agent.save()['params']
  pure = nj.pure(lambda: _evaluate(agent.model, obs, prevact, actions, spec, start))
  with jax._src.config.explicit_device_put_scope():
    returned, result = pure(params, seed=np.asarray(seed, np.uint32),
                            create=False, modify=True, ignore=True)
  if set(returned) != set(params):
    raise AssertionError('Evaluation changed parameter keys')
  with jax._src.config.explicit_device_get_scope():
    return jax.tree.map(np.asarray, result)
