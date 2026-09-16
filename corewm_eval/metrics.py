import numpy as np


def assert_nonnegative(name, values, tolerance=1e-8):
  values = np.asarray(values, np.float64)
  if not np.isfinite(values).all():
    raise AssertionError(f'{name} contains NaN/Inf')
  minimum = float(values.min()) if values.size else 0.0
  if minimum < -float(tolerance):
    raise AssertionError(f'{name} must be non-negative; minimum={minimum}')
  return values


def _finite(name, value):
  value = np.asarray(value)
  if not np.isfinite(value).all():
    raise ValueError(f'{name} contains NaN or Inf')
  return value


def multivariate_r2(target, prediction):
  target = _finite('target', target).astype(np.float64)
  prediction = _finite('prediction', prediction).astype(np.float64)
  if target.shape != prediction.shape:
    raise ValueError((target.shape, prediction.shape))
  residual = np.square(target - prediction).sum()
  centered = target - target.mean(axis=0, keepdims=True)
  denominator = np.square(centered).sum()
  if denominator <= 0:
    raise ValueError('R2 target is numerically degenerate')
  return float(1.0 - residual / denominator)


def progressive_sum(residuals):
  if not residuals:
    raise ValueError('At least one residual is required')
  out, running = [], np.zeros_like(np.asarray(residuals[0]))
  for residual in residuals:
    running = running + np.asarray(residual)
    out.append(running.copy())
  return tuple(out)


def reconstruction_errors(h, cumulative_recons):
  h = _finite('h', h).astype(np.float64)
  errors = []
  for recon in cumulative_recons:
    recon = _finite('reconstruction', recon).astype(np.float64)
    if recon.shape != h.shape:
      raise ValueError((h.shape, recon.shape))
    errors.append(float(np.square(h - recon).mean(axis=-1).mean()))
  return np.asarray(errors)


def reconstruction_metrics(h, cumulative_recons, block_dimensions):
  errors = reconstruction_errors(h, cumulative_recons)
  block_dimensions = np.asarray(block_dimensions, dtype=np.int64)
  if len(errors) != len(block_dimensions):
    raise ValueError((len(errors), len(block_dimensions)))
  gains = np.full(len(errors), np.nan)
  gains[1:] = errors[:-1] - errors[1:]
  gains_per_dim = gains / block_dimensions
  return errors, gains, gains_per_dim


def horizon_indices(horizons):
  """Array index of each reported horizon in a rollout: index = h - 1."""
  indices = [int(h) - 1 for h in horizons]
  if any(index < 0 for index in indices):
    raise ValueError(f'Horizons must be >= 1: {tuple(horizons)}')
  return indices


def pixel_mae(prediction, target):
  """Per-frame mean |prediction - target|, reducing every non-leading axis."""
  prediction = _finite('prediction', prediction).astype(np.float64)
  target = _finite('target', target).astype(np.float64)
  if prediction.shape != target.shape:
    raise ValueError((prediction.shape, target.shape))
  if prediction.ndim < 2:
    raise ValueError(f'Expected a leading frame axis, got {prediction.shape}')
  return np.abs(prediction - target).mean(axis=tuple(range(1, prediction.ndim)))


def prefix_error_matrix(prefix_images, ground_truth, horizons):
  """[num_prefixes, num_horizons] pixel MAE, E_l(h).

  Every representation at a horizon is scored against the same ground-truth
  frame, which is why the horizon selection happens here rather than in each
  caller.
  """
  index = horizon_indices(horizons)
  target = np.asarray(ground_truth)[index]
  return np.stack([
      pixel_mae(np.asarray(image)[index], target) for image in prefix_images])


def full_error_row(full_images, ground_truth, horizons):
  """[num_horizons] pixel MAE of the full RSSM feature, E_full(h)."""
  index = horizon_indices(horizons)
  return pixel_mae(
      np.asarray(full_images)[index], np.asarray(ground_truth)[index])


def performance_gap(prefix_mae, full_mae):
  """G_l(h) = E_l(h) - E_full(h); positive means worse than the full RSSM."""
  prefix_mae = _finite('prefix_mae', prefix_mae).astype(np.float64)
  full_mae = _finite('full_mae', full_mae).astype(np.float64)
  if prefix_mae.ndim != 2 or full_mae.ndim != 1:
    raise ValueError((prefix_mae.shape, full_mae.shape))
  if prefix_mae.shape[1] != full_mae.shape[0]:
    raise ValueError((prefix_mae.shape, full_mae.shape))
  return prefix_mae - full_mae[None, :]


def marginal_block_gain(prefix_mae):
  """Delta_l(h) = E_{l-1}(h) - E_l(h) for blocks 2..L; positive means better.

  Block 1 has no row because there is no Prefix 0 to improve on.
  """
  prefix_mae = _finite('prefix_mae', prefix_mae).astype(np.float64)
  if prefix_mae.ndim != 2 or len(prefix_mae) < 2:
    raise ValueError(prefix_mae.shape)
  return prefix_mae[:-1] - prefix_mae[1:]


def marginal_visual_change(prefix_images, horizons):
  """C_l(h) = mean |I_l(h) - I_{l-1}(h)| for blocks 2..L.

  Says how much the decoded image moved when a block was added; pair it with
  marginal_block_gain, which says whether the move helped.
  """
  index = horizon_indices(horizons)
  return np.stack([
      pixel_mae(np.asarray(current)[index], np.asarray(previous)[index])
      for previous, current in zip(prefix_images[:-1], prefix_images[1:])])


def off_diagonal_mean(matrix):
  """Mean of the strict upper triangle of a symmetric matrix."""
  matrix = _finite('matrix', matrix).astype(np.float64)
  if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
    raise ValueError(matrix.shape)
  if len(matrix) < 2:
    raise ValueError('Off-diagonal mean needs at least two blocks')
  return float(matrix[np.triu_indices(len(matrix), 1)].mean())


def linear_cka(x, y):
  x = _finite('x', x).astype(np.float64)
  y = _finite('y', y).astype(np.float64)
  if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
    raise ValueError((x.shape, y.shape))
  x = x - x.mean(axis=0, keepdims=True)
  y = y - y.mean(axis=0, keepdims=True)
  cross = x.T @ y
  numerator = np.square(cross).sum()
  norm_x = np.sqrt(np.square(x.T @ x).sum())
  norm_y = np.sqrt(np.square(y.T @ y).sum())
  denominator = norm_x * norm_y
  if denominator <= np.finfo(np.float64).tiny:
    raise ValueError('CKA denominator is numerically degenerate')
  return float(numerator / denominator)


def cka_matrix(blocks):
  matrix = np.empty((len(blocks), len(blocks)), np.float64)
  for i, x in enumerate(blocks):
    for j, y in enumerate(blocks):
      matrix[i, j] = linear_cka(x, y)
  if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
    raise AssertionError('CKA matrix is not symmetric')
  return matrix
