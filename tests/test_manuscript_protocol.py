"""Verification plan for the manuscript block/prefix evaluation protocol."""

import inspect

import numpy as np
import pytest

from corewm_eval import manuscript_clips, manuscript_eval
from corewm_eval.config import (
    CLIP_LENGTH, HORIZONS, MANUSCRIPT_START, MIN_CLIP_LENGTH, NUM_PREFIXES,
    ROLLOUT_LENGTH)
from corewm_eval.extraction import assert_rollout_actions, previous_actions
from corewm_eval.metrics import (
    full_error_row, horizon_indices, marginal_block_gain,
    marginal_visual_change, off_diagonal_mean, performance_gap,
    prefix_error_matrix)

SHAPE = (6, 5, 3)   # small HxWxC so the fixtures stay cheap


def _images(rng, length=ROLLOUT_LENGTH):
  return rng.random((length, *SHAPE))


@pytest.fixture
def rollout():
  rng = np.random.default_rng(0)
  ground_truth = _images(rng)
  full = _images(rng)
  prefixes = [_images(rng) for _ in range(NUM_PREFIXES)]
  return ground_truth, full, prefixes


# -- protocol constants ----------------------------------------------------

def test_exactly_five_horizons_with_a_matching_rollout_length():
  assert len(HORIZONS) == 5
  assert HORIZONS == (1, 2, 4, 8, 16)
  assert ROLLOUT_LENGTH == max(HORIZONS) == 16
  assert MANUSCRIPT_START == 16
  assert MIN_CLIP_LENGTH == MANUSCRIPT_START + ROLLOUT_LENGTH + 1 == 33
  assert CLIP_LENGTH >= MIN_CLIP_LENGTH


def test_horizon_h_maps_to_array_index_h_minus_one():
  assert horizon_indices(HORIZONS) == [0, 1, 3, 7, 15]
  with pytest.raises(ValueError):
    horizon_indices((0, 1))


def test_no_evaluation_module_redeclares_the_horizon_list():
  from corewm_eval import manuscript_figures, manuscript_suite
  for module in (manuscript_eval, manuscript_clips, manuscript_figures,
                 manuscript_suite):
    source = inspect.getsource(module)
    assert 'HORIZONS = ' not in source, module.__name__
    assert '32, 64' not in source, module.__name__


# -- matrix shapes ---------------------------------------------------------

def test_matrix_shapes(rollout):
  ground_truth, full, prefixes = rollout
  prefix_mae = prefix_error_matrix(prefixes, ground_truth, HORIZONS)
  full_mae = full_error_row(full, ground_truth, HORIZONS)
  assert prefix_mae.shape == (5, 5)
  assert full_mae.shape == (5,)
  assert performance_gap(prefix_mae, full_mae).shape == (5, 5)
  assert marginal_block_gain(prefix_mae).shape == (4, 5)
  assert marginal_visual_change(prefixes, HORIZONS).shape == (4, 5)


# -- matrix identities -----------------------------------------------------

def test_the_full_row_would_have_exactly_zero_gap(rollout):
  ground_truth, full, _ = rollout
  full_mae = full_error_row(full, ground_truth, HORIZONS)
  gap = performance_gap(full_mae[None, :], full_mae)
  np.testing.assert_array_equal(gap, np.zeros((1, len(HORIZONS))))


def test_marginal_gain_is_the_difference_of_consecutive_prefix_errors(rollout):
  ground_truth, _, prefixes = rollout
  prefix_mae = prefix_error_matrix(prefixes, ground_truth, HORIZONS)
  np.testing.assert_allclose(
      marginal_block_gain(prefix_mae), prefix_mae[:-1] - prefix_mae[1:])


def test_marginal_gain_has_no_row_for_the_first_block(rollout):
  ground_truth, _, prefixes = rollout
  gain = marginal_block_gain(
      prefix_error_matrix(prefixes, ground_truth, HORIZONS))
  assert len(gain) == NUM_PREFIXES - 1


def test_last_prefix_is_not_assumed_to_equal_the_full_feature(rollout):
  ground_truth, full, prefixes = rollout
  prefix_mae = prefix_error_matrix(prefixes, ground_truth, HORIZONS)
  full_mae = full_error_row(full, ground_truth, HORIZONS)
  gap = performance_gap(prefix_mae, full_mae)
  # The last prefix is measured against the full feature rather than being
  # substituted for it, so its gap row is free to be non-zero.
  assert np.abs(gap[-1]).max() > 0


def test_off_diagonal_mean_ignores_the_unit_diagonal():
  matrix = np.array([[1.0, 0.2, 0.4], [0.2, 1.0, 0.6], [0.4, 0.6, 1.0]])
  assert off_diagonal_mean(matrix) == pytest.approx((0.2 + 0.4 + 0.6) / 3)


# -- ground-truth alignment ------------------------------------------------

def test_each_horizon_is_scored_against_ground_truth_frame_start_plus_h():
  # Frame i carries the constant value i, so a misalignment of even one step
  # shows up as a MAE of exactly that offset.
  clip = np.stack([np.full(SHAPE, float(i)) for i in range(64)])
  target = clip[MANUSCRIPT_START + 1:MANUSCRIPT_START + ROLLOUT_LENGTH + 1]
  for horizon in HORIZONS:
    assert target[horizon - 1][0, 0, 0] == MANUSCRIPT_START + horizon
  aligned = prefix_error_matrix([target], target, HORIZONS)
  np.testing.assert_array_equal(aligned, np.zeros((1, len(HORIZONS))))
  shifted = prefix_error_matrix([target + 1], target, HORIZONS)
  np.testing.assert_allclose(shifted, np.ones((1, len(HORIZONS))))


def test_every_representation_at_a_horizon_shares_one_ground_truth_frame():
  clip = np.stack([np.full(SHAPE, float(i)) for i in range(64)])
  target = clip[MANUSCRIPT_START + 1:MANUSCRIPT_START + ROLLOUT_LENGTH + 1]
  prefixes = [target + offset for offset in range(NUM_PREFIXES)]
  matrix = prefix_error_matrix(prefixes, target, HORIZONS)
  # Row l is offset l from the same frame at every horizon.
  np.testing.assert_allclose(
      matrix, np.tile(np.arange(NUM_PREFIXES, dtype=float)[:, None],
                      (1, len(HORIZONS))))


def test_rollout_uses_exactly_the_actions_that_follow_the_warm_up():
  source = inspect.getsource(manuscript_eval)
  assert 'start:start + ROLLOUT_LENGTH' in source
  assert '64' not in source


# -- episode boundaries ----------------------------------------------------

def test_clip_is_cut_at_the_first_terminal():
  is_last = np.zeros(10, bool)
  is_last[[4, 8]] = True
  values = {'is_last': is_last, 'image': np.arange(10)}
  cut = manuscript_clips.cut_at_first_terminal(values)
  assert len(cut['is_last']) == 5
  assert bool(cut['is_last'][-1])
  np.testing.assert_array_equal(cut['image'], np.arange(5))


def test_clip_without_a_terminal_is_kept_whole():
  values = {'is_last': np.zeros(7, bool), 'image': np.arange(7)}
  cut = manuscript_clips.cut_at_first_terminal(values)
  assert len(cut['is_last']) == 7


def test_recorded_keys_cover_every_encoder_input_plus_the_action():
  class Agent:
    obs_space = {'image': None, 'state': None, 'reward': None,
                 'is_first': None, 'is_last': None, 'is_terminal': None,
                 'log/success_once': None}
  keys = manuscript_clips.recorded_keys(Agent())
  assert 'state' in keys and 'image' in keys and 'action' in keys
  assert not any(k.startswith('log/') for k in keys)


# -- action semantics ------------------------------------------------------

def test_previous_actions_shifts_along_time_for_scalar_actions():
  actions = np.array([3, 1, 4, 1, 5])
  np.testing.assert_array_equal(
      previous_actions(actions), [0, 3, 1, 4, 1])


def test_previous_actions_shifts_along_time_for_continuous_actions():
  actions = np.arange(12, dtype=np.float32).reshape(3, 4)
  expected = np.zeros_like(actions)
  expected[1:] = actions[:-1]
  np.testing.assert_array_equal(previous_actions(actions), expected)


def _bfloat16_round(values):
  """Quantize onto the bfloat16 grid: 1 implicit + 7 stored mantissa bits."""
  values = np.asarray(values, np.float32)
  scale = np.where(values == 0, 1.0, np.exp2(np.floor(np.log2(
      np.abs(np.where(values == 0, 1.0, values)))) - 7))
  return (np.round(values / scale) * scale).astype(np.float32)


def test_rollout_action_check_survives_the_rssm_compute_dtype():
  # RSSM.imagine casts float actions to jax.compute_dtype (bfloat16), so a
  # continuous action comes back rounded. That must not read as a mismatch.
  rng = np.random.default_rng(0)
  wanted = rng.uniform(-1.3, 1.3, size=(1, ROLLOUT_LENGTH, 8)).astype(np.float32)
  assert_rollout_actions(_bfloat16_round(wanted).astype(np.float16), wanted)


def test_rollout_action_check_is_exact_for_discrete_actions():
  wanted = np.arange(ROLLOUT_LENGTH, dtype=np.int32)[None]
  assert_rollout_actions(wanted.copy(), wanted)
  with pytest.raises(AssertionError):
    assert_rollout_actions(wanted + 1, wanted)


def test_rollout_action_check_still_catches_a_misaligned_slice():
  rng = np.random.default_rng(1)
  clip = rng.uniform(-1, 1, size=(1, 64, 8)).astype(np.float32)
  wanted = clip[:, MANUSCRIPT_START:MANUSCRIPT_START + ROLLOUT_LENGTH]
  shifted = clip[:, MANUSCRIPT_START + 1:MANUSCRIPT_START + 1 + ROLLOUT_LENGTH]
  with pytest.raises(AssertionError):
    assert_rollout_actions(shifted.astype(np.float16), wanted)


def test_rollout_action_check_rejects_a_wrong_length():
  wanted = np.zeros((1, ROLLOUT_LENGTH, 8), np.float32)
  with pytest.raises(AssertionError):
    assert_rollout_actions(wanted[:, :-1], wanted)


# -- read-only guarantees --------------------------------------------------

def test_prefix_dynamics_predictor_is_never_invoked():
  source = inspect.getsource(manuscript_eval)
  assert 'htp_pdyn' not in source
  assert 'htp_recon.reconstruct' in source


def test_evaluation_cannot_create_or_keep_parameter_updates():
  source = inspect.getsource(manuscript_eval.evaluate_readonly)
  assert 'create=False' in source and 'ignore=True' in source
  assert 'Evaluation changed parameter keys' in source
