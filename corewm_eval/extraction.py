"""Read-only posterior representation extraction from real Dreamer modules."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np


@dataclass(frozen=True)
class ExtractionSpec:
  deter_dim: int
  stoch_shape: tuple
  prefix_dims: tuple


def evaluation_seed(global_seed, episode_id, start_timestep=0, stream=0):
  """Order-independent uint32[2] RNG key derivation."""
  sequence = np.random.SeedSequence([
      int(global_seed), int(episode_id), int(start_timestep), int(stream)])
  return sequence.generate_state(2, dtype=np.uint32)


def _flat_to_feat(h, spec):
  # A spec that disagrees with the checkpoint splits h at the wrong offset and
  # hands the decoder two halves of the wrong tensors, which decodes to noise
  # rather than failing. Check it instead: the dims come from the run config,
  # and a preset that resized the RSSM without the spec following would land
  # here.
  expected = spec.deter_dim + int(np.prod(spec.stoch_shape))
  if h.shape[-1] != expected:
    raise ValueError(
        f'Latent is {h.shape[-1]} wide but the extraction spec describes '
        f'{expected} (deter {spec.deter_dim} + stoch '
        f'{spec.stoch_shape}). The spec does not match this checkpoint.')
  return {
      'deter': h[..., :spec.deter_dim],
      'stoch': h[..., spec.deter_dim:].reshape(
          (*h.shape[:-1], *spec.stoch_shape)),
  }


def _extract_impl(model, obs, prevact, spec):
  reset = obs['is_first']
  batch = reset.shape[0]
  enc_carry = model.enc.initial(batch)
  dyn_carry = model.dyn.initial(batch)
  dec_carry = model.dec.initial(batch)
  enc_carry, _, tokens = model.enc(
      enc_carry, obs, reset, training=False)
  dyn_carry, posterior, feat = model.dyn.observe(
      dyn_carry, tokens, prevact, reset, training=False)
  h = model.feat2h(feat)
  z = model.htp_proj(h)
  cumulative = model.htp_recon.reconstruct(z, h)
  _, _, decoded_original = model.dec(
      dec_carry, {'deter': feat['deter'], 'stoch': feat['stoch']},
      reset, training=False)
  recovered = _flat_to_feat(h, spec)
  _, _, decoded_recovered = model.dec(
      dec_carry, recovered, reset, training=False)
  decoded_prefixes = []
  for hhat in cumulative:
    _, _, decoded = model.dec(
        dec_carry, _flat_to_feat(hhat, spec), reset, training=False)
    decoded_prefixes.append({k: v.pred() for k, v in decoded.items()})
  dims = tuple(spec.prefix_dims)
  lows = (0, *dims[:-1])
  return {
      'posterior': posterior,
      'h': h,
      'z': z,
      'prefixes': tuple(z[..., :d] for d in dims),
      'blocks': tuple(z[..., lo:hi] for lo, hi in zip(lows, dims)),
      'cumulative_reconstructions': cumulative,
      'decoded_original': {k: v.pred() for k, v in decoded_original.items()},
      'decoded_recovered': {k: v.pred() for k, v in decoded_recovered.items()},
      'decoded_prefixes': tuple(decoded_prefixes),
  }


def extract_readonly(agent, obs, prevact, spec, seed, params=None):
  """Evaluate without allowing Ninjax state creation or modification."""
  pure = nj.pure(lambda obs, prevact: _extract_impl(
      agent.model, obs, prevact, spec))
  params = agent.save()['params'] if params is None else params
  before_keys = tuple(sorted(params))
  returned, outputs = pure(
      params, obs, prevact, seed=np.asarray(seed, np.uint32),
      # Ninjax scan needs modify=True for its access pre-pass. ignore=True
      # guarantees that any attempted state update is discarded.
      create=False, modify=True, ignore=True)
  if tuple(sorted(returned)) != before_keys:
    raise AssertionError('Parameter key set changed during extraction')
  return jax.tree.map(np.asarray, outputs)


def _open_loop_impl(model, start_state, actions, spec):
  length = jax.tree.leaves(actions)[0].shape[1]
  _, feat, used_actions = model.dyn.imagine(
      start_state, actions, length, training=False)
  h = model.feat2h(feat)
  z = model.htp_proj(h)
  cumulative = model.htp_recon.reconstruct(z, h)
  reset = jnp.zeros(h.shape[:2], bool)
  carry = model.dec.initial(h.shape[0])
  _, _, decoded_full = model.dec(
      carry, {'deter': feat['deter'], 'stoch': feat['stoch']},
      reset, training=False)
  decoded_prefixes = []
  for hhat in cumulative:
    _, _, decoded = model.dec(
        carry, _flat_to_feat(hhat, spec), reset, training=False)
    decoded_prefixes.append({k: v.pred() for k, v in decoded.items()})
  return {
      'h_tilde': h,
      'z_tilde': z,
      'cumulative_reconstructions': cumulative,
      'decoded_full': {k: v.pred() for k, v in decoded_full.items()},
      'decoded_prefixes': tuple(decoded_prefixes),
      'actions_used': used_actions,
  }


def open_loop_readonly(agent, start_state, actions, spec, seed, params=None):
  """Shared-backbone rollout; reconstructed prefixes are never transitioned."""
  params = agent.save()['params'] if params is None else params
  pure = nj.pure(lambda state, acts: _open_loop_impl(
      agent.model, state, acts, spec))
  returned, outputs = pure(
      params, start_state, actions, seed=np.asarray(seed, np.uint32),
      create=False, modify=True, ignore=True)
  if tuple(sorted(returned)) != tuple(sorted(params)):
    raise AssertionError('Parameter key set changed during rollout')
  return jax.tree.map(np.asarray, outputs)


def assert_rollout_actions(used, wanted):
  """Check that an imagination consumed exactly the intended action slice.

  RSSM.imagine runs its inputs through nn.cast, which casts floating-point
  arrays to jax.compute_dtype (bfloat16 by default) and leaves integers alone.
  A discrete action therefore returns byte-identical, while a continuous one
  returns rounded to roughly three significant digits -- about 0.4% relative
  error, which is not a mismatch. A misaligned slice differs by order 1, so
  comparing in float32 with a tolerance between the two still pins alignment.
  """
  used, wanted = np.asarray(used), np.asarray(wanted)
  message = 'Imagination did not consume the intended action slice'
  if used.shape != wanted.shape:
    raise AssertionError(f'{message}: {used.shape} vs {wanted.shape}')
  if used.dtype == wanted.dtype:
    np.testing.assert_array_equal(used, wanted, err_msg=message)
    return
  np.testing.assert_allclose(
      used.astype(np.float32), wanted.astype(np.float32),
      rtol=1e-2, atol=1e-2, err_msg=message)


def previous_actions(actions, cardinality=None):
  """Shift an action sequence one step along TIME, which is axis 0.

  Callers pass the recorded actions of a single clip, so axis 0 is time and
  any trailing axis is the action itself. Shifting along the last axis is the
  same operation only when the action is a scalar per step (Atari); for a
  continuous action vector (ManiSkill, shape [T, act_dim]) it would rotate
  coordinates inside the vector instead of stepping back in time.
  """
  actions = np.asarray(actions)
  if actions.ndim < 1:
    raise ValueError(f'Expected a time axis, got shape {actions.shape}')
  previous = np.zeros_like(actions)
  previous[1:] = actions[:-1]
  if cardinality is not None and (
      (previous < 0).any() or (previous >= cardinality).any()):
    raise ValueError('Action outside cardinality')
  return previous
