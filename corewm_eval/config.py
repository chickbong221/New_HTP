import numpy as np

EVALUATION_SEED = 0
TRAINING_SEEDS = (0, 1, 2, 3, 4)
ATARI100K_GAMES = (
    'alien', 'amidar', 'assault', 'asterix', 'bank_heist', 'battle_zone',
    'boxing', 'breakout', 'chopper_command', 'crazy_climber', 'demon_attack',
    'freeway', 'frostbite', 'gopher', 'hero', 'jamesbond', 'kangaroo',
    'krull', 'kung_fu_master', 'ms_pacman', 'pong', 'private_eye', 'qbert',
    'road_runner', 'seaquest', 'up_n_down')
# --------------------------------------------------------------------------
# Manuscript block/prefix evaluation protocol -- the single authoritative
# definition. manuscript_eval, manuscript_clips, manuscript_figures and
# manuscript_suite all import from here; none of them re-declares a horizon
# list. Changing a value here changes the whole protocol.
#
# Note: smoke_test.py keeps its own 64-step open-loop horizons. That is the
# Phase-2 integration diagnostic (E_h / E_prefix / E_backbone / E_total), a
# different measurement with an already-published artifact, not this protocol.
# --------------------------------------------------------------------------
MANUSCRIPT_START = 16            # T: posterior index the imagination starts at
HORIZONS = (1, 2, 4, 8, 16)      # reported horizons h; array index is h - 1
ROLLOUT_LENGTH = max(HORIZONS)   # imagined steps taken from the posterior at T
NUM_PREFIXES = 5                 # cumulative prefixes P1..P5
# T warm-up transitions + ROLLOUT_LENGTH imagined transitions + initial frame.
MIN_CLIP_LENGTH = MANUSCRIPT_START + ROLLOUT_LENGTH + 1
# Frames requested per clip. The surplus over MIN_CLIP_LENGTH is the same
# slack the previous protocol used (96 collected for 81 required).
CLIP_LENGTH = MIN_CLIP_LENGTH + 15

LONG_HORIZONS = (16, 32, 64)     # probe pipeline only; not the manuscript protocol
ALPHA_GRID = np.logspace(-6, 6, 13)
PERFORMANCE_THRESHOLDS = np.linspace(0.0, 8.0, 161)
FINAL_CHECKPOINTS = tuple(range(10_000, 100_001, 10_000))
QUALITATIVE_GAMES = (
    'alien', 'bank_heist', 'frostbite', 'kangaroo', 'ms_pacman', 'seaquest')
SPLIT_COUNTS = {'train': 60, 'validation': 20, 'test': 20}

WANDB_ENTITY = 'ttdat170703-ho-chi-minh-city-university-of-technology'
WANDB_GAME_ALIASES = {'jamesbond': 'james_bond'}


def wandb_project_for_game(game):
  """Return the existing team project assigned to one canonical Atari game."""
  if game not in ATARI100K_GAMES:
    raise ValueError(f'Not a canonical Atari100K game: {game!r}')
  return f'dreamv3-{WANDB_GAME_ALIASES.get(game, game)}'
