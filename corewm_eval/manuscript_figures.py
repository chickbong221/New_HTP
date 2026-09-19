"""Figures for the manuscript block/prefix evaluation.

Every function takes arrays already reduced to the reported horizons, or does
the horizon selection itself through metrics.horizon_indices, so no figure
re-derives the protocol.
"""

import numpy as np

from .config import HORIZONS
from .metrics import horizon_indices


def _pyplot():
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  return plt


def _channels_to_rgb(image):
  """Collapse a decoded frame to something imshow accepts.

  A ManiSkill task whose robot carries a wrist camera returns one image with
  every camera concatenated on the channel axis -- PegInsertionSide-v1 is
  base_camera + hand_camera, so 6 channels -- and frame stacking multiplies
  channels the same way. Tile those views side by side rather than keeping
  one, which would silently hide a camera in every figure while the MAE
  numbers still averaged over both.
  """
  image = np.asarray(image, np.float64)
  if image.ndim == 2:
    return image
  channels = image.shape[-1]
  if channels == 1:
    return image[..., 0]
  if channels == 3:
    return image
  if channels % 3 == 0:
    views = channels // 3
    height, width, _ = image.shape
    image = image.reshape(height, width, views, 3).transpose(0, 2, 1, 3)
    return image.reshape(height, views * width, 3)
  return image[..., :3]


def prefix_labels(count):
  return [f'P{i + 1}' for i in range(count)]


def added_block_labels(count):
  """Rows of the marginal matrices: block 1 has no row to add against."""
  return [f'Add B{i + 2}' for i in range(count)]


def decoded_matrix(path, ground_truth, full, prefixes, horizons=HORIZONS,
                   posterior=None):
  """Rows are horizons; columns are GT, [Posterior,] Full, P1..PL.

  The optional posterior column is the same frame decoded with the real
  observation fed in, so a reader can tell open-loop prediction error apart
  from a decode that is broken outright.
  """
  index = horizon_indices(horizons)
  columns = ['GT'] + (['Posterior'] if posterior is not None else []) + [
      'Full'] + prefix_labels(len(prefixes))
  panels = [np.asarray(ground_truth)[index]]
  if posterior is not None:
    panels.append(np.asarray(posterior)[index])
  panels += [np.asarray(full)[index],
             *[np.asarray(p)[index] for p in prefixes]]
  plt = _pyplot()
  fig, axes = plt.subplots(
      len(index), len(columns),
      figsize=(1.55 * len(columns), 1.7 * len(index)), squeeze=False)
  for row, horizon in enumerate(horizons):
    for col, panel in enumerate(panels):
      ax = axes[row][col]
      ax.imshow(np.clip(_channels_to_rgb(panel[row]), 0, 1))
      ax.set_xticks([])
      ax.set_yticks([])
      if row == 0:
        ax.set_title(columns[col])
      if col == 0:
        ax.set_ylabel(f'h={horizon}')
  fig.tight_layout()
  fig.savefig(path, dpi=140)
  plt.close(fig)


def _annotated_heatmap(path, matrix, rows, horizons, title, diverging,
                       limit=None):
  matrix = np.asarray(matrix, np.float64)
  plt = _pyplot()
  if diverging:
    # Centered at zero so sign is readable at a glance. Both ends of coolwarm
    # are saturated, so only cells near zero get a light background.
    span = float(limit if limit is not None else np.abs(matrix).max()) or 1.0
    kwargs = dict(cmap='coolwarm', vmin=-span, vmax=span)
    light_cell = lambda value: abs(value) < 0.6 * span
  else:
    # One fixed sequential scale anchored at zero. viridis runs dark -> light.
    span = float(limit if limit is not None else matrix.max()) or 1.0
    kwargs = dict(cmap='viridis', vmin=0.0, vmax=span)
    light_cell = lambda value: value > 0.6 * span
  fig, ax = plt.subplots(
      figsize=(1.15 * len(horizons) + 2.4, 0.62 * len(rows) + 1.9))
  image = ax.imshow(matrix, aspect='auto', **kwargs)
  fig.colorbar(image, ax=ax)
  ax.set(title=title, xlabel='Horizon (actions)',
         xticks=range(len(horizons)),
         xticklabels=[str(h) for h in horizons],
         yticks=range(len(rows)), yticklabels=list(rows))
  for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
      value = matrix[i, j]
      ax.text(j, i, f'{value:.4f}', ha='center', va='center', fontsize=7,
              color='black' if light_cell(value) else 'white')
  fig.tight_layout()
  fig.savefig(path, dpi=140)
  plt.close(fig)


def mae_heatmap(path, full_mae, prefix_mae, horizons=HORIZONS, limit=None):
  """Rows Full, P1..PL on one fixed sequential scale starting at zero."""
  matrix = np.vstack([np.asarray(full_mae)[None, :], np.asarray(prefix_mae)])
  rows = ['Full'] + prefix_labels(len(np.asarray(prefix_mae)))
  _annotated_heatmap(path, matrix, rows, horizons, 'Pixel MAE [0,1]',
                     diverging=False, limit=limit)


def gap_heatmap(path, gap, horizons=HORIZONS, limit=None):
  """Rows P1..PL; prefix MAE minus full MAE, diverging about zero."""
  gap = np.asarray(gap)
  _annotated_heatmap(path, gap, prefix_labels(len(gap)), horizons,
                     'Performance gap (prefix MAE - full MAE)',
                     diverging=True, limit=limit)


def gain_heatmap(path, gain, horizons=HORIZONS, limit=None):
  """Rows Add B2..BL; previous-prefix MAE minus current, diverging."""
  gain = np.asarray(gain)
  _annotated_heatmap(path, gain, added_block_labels(len(gain)), horizons,
                     'Marginal block gain (E_{l-1} - E_l)',
                     diverging=True, limit=limit)


def change_heatmap(path, change, horizons=HORIZONS, limit=None):
  """Rows Add B2..BL; how much the decoded image moved, sequential."""
  change = np.asarray(change)
  _annotated_heatmap(path, change, added_block_labels(len(change)), horizons,
                     'Marginal visual change |I_l - I_{l-1}|',
                     diverging=False, limit=limit)


def _error_grid(path, panels, rows, horizons, title):
  plt = _pyplot()
  panels = [np.asarray(p, np.float64) for p in panels]
  span = max(float(p.max()) for p in panels) or 1.0
  fig, axes = plt.subplots(
      len(rows), len(horizons),
      figsize=(1.55 * len(horizons), 1.7 * len(rows)), squeeze=False)
  image = None
  for row, panel in enumerate(panels):
    for col, horizon in enumerate(horizons):
      ax = axes[row][col]
      image = ax.imshow(panel[col], cmap='magma', vmin=0.0, vmax=span)
      ax.set_xticks([])
      ax.set_yticks([])
      if row == 0:
        ax.set_title(f'h={horizon}')
      if col == 0:
        ax.set_ylabel(rows[row])
  fig.suptitle(title)
  fig.colorbar(image, ax=axes, shrink=0.8)
  fig.savefig(path, dpi=140, bbox_inches='tight')
  plt.close(fig)


def _abs_error_maps(reference, images, horizons):
  """Per-pixel |image - reference| at each horizon, averaged over channels."""
  index = horizon_indices(horizons)
  reference = np.asarray(reference, np.float64)[index]
  return [np.abs(np.asarray(image, np.float64)[index] - reference).mean(-1)
          for image in images]


def pixel_error_vs_ground_truth(path, ground_truth, full, prefixes,
                                horizons=HORIZONS):
  """|I_l - I_GT| heatmaps, rows Full and P1..PL."""
  panels = _abs_error_maps(ground_truth, [full, *prefixes], horizons)
  rows = ['Full'] + prefix_labels(len(prefixes))
  _error_grid(path, panels, rows, horizons, '|I - I_GT|')


def pixel_change_between_prefixes(path, prefixes, horizons=HORIZONS):
  """|I_l - I_{l-1}| heatmaps, rows Add B2..BL."""
  index = horizon_indices(horizons)
  panels = [
      np.abs(np.asarray(current, np.float64)[index]
             - np.asarray(previous, np.float64)[index]).mean(-1)
      for previous, current in zip(prefixes[:-1], prefixes[1:])]
  _error_grid(path, panels, added_block_labels(len(panels)), horizons,
              '|I_l - I_{l-1}|')


def cka_heatmap(path, cka):
  """Block-vs-block CKA with the numeric values written into the cells."""
  cka = np.asarray(cka, np.float64)
  labels = [f'B{i + 1}' for i in range(len(cka))]
  plt = _pyplot()
  fig, ax = plt.subplots(figsize=(0.8 * len(cka) + 2.6, 0.8 * len(cka) + 2.0))
  image = ax.imshow(cka, vmin=0, vmax=1, cmap='viridis')
  fig.colorbar(image, ax=ax)
  ax.set(xlabel='Block', ylabel='Block', xticks=range(len(cka)),
         yticks=range(len(cka)), xticklabels=labels, yticklabels=labels)
  for i in range(len(cka)):
    for j in range(len(cka)):
      ax.text(j, i, f'{cka[i, j]:.3f}', ha='center', va='center', fontsize=7,
              color='black' if cka[i, j] > 0.6 else 'white')
  fig.tight_layout()
  fig.savefig(path, dpi=140)
  plt.close(fig)


def one_step_blocks(path, ground_truth, one_step_full, blocks, positions):
  """Qualitative block-only decodes at a few absolute timesteps."""
  plt = _pyplot()
  rows = ['GT', 'One step'] + [f'Block {i + 1}' for i in range(len(blocks))]
  panels = [np.asarray(ground_truth), np.asarray(one_step_full),
            *[np.asarray(b) for b in blocks]]
  fig, axes = plt.subplots(
      len(rows), len(positions),
      figsize=(1.55 * len(positions), 1.7 * len(rows)), squeeze=False)
  for row, panel in enumerate(panels):
    for col, position in enumerate(positions):
      ax = axes[row][col]
      ax.imshow(np.clip(_channels_to_rgb(panel[col]), 0, 1))
      ax.set_xticks([])
      ax.set_yticks([])
      if row == 0:
        ax.set_title(f't={position}')
      if col == 0:
        ax.set_ylabel(rows[row])
  fig.tight_layout()
  fig.savefig(path, dpi=140)
  plt.close(fig)
