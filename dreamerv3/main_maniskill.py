"""ManiSkill entry point.

Kept separate from dreamerv3/main.py and dreamerv3/main_htp.py so that the
existing entry points, their config tree and their paper-artifact
resolved_config_hash are untouched by the ManiSkill merge. Run with:

    python -m dreamerv3.main_maniskill --configs maniskill_rgb \
        --task maniskill_PickCube-v1 --logdir ~/logdir/ms_rgb

Compose the HTP presets on top, e.g.

    python -m dreamerv3.main_maniskill \
        --configs maniskill_rgb_eval_ready corewm_full \
        --logdir ~/logdir/ms_corewm_full

Set HTP_AGENT to pick the agent class (default 'agent_htp:Agent_HTP'), e.g.
HTP_AGENT=agent:Agent for the unmodified DreamerV3 agent.
"""

import importlib
import os
import pathlib
import sys
from functools import partial as bind

# Must be set before JAX initialises to share the GPU with ManiSkill cleanly.
# jax.prealloc=false in the preset does this too (embodied/jax/internal.py),
# this is the belt-and-braces version from dreamerv3-maniskill-hab.
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

folder = pathlib.Path(__file__).parent
sys.path.insert(0, str(folder.parent))
sys.path.insert(1, str(folder.parent.parent))
__package__ = folder.name

import elements
import embodied
import portal
import ruamel.yaml as yaml

from . import main_htp as base
from embodied.run.train_maniskill import train as train_maniskill


# Which agent class to build, as 'module:ClassName' relative to this package.
AGENT_SPEC = os.environ.get('HTP_AGENT', 'agent_htp:Agent_HTP')


def _resolve_agent():
  module, cls = AGENT_SPEC.split(':')
  module = importlib.import_module(f'.{module}', package=__package__)
  return getattr(module, cls)


def _merge(dst, src):
  """Recursive dict merge; src wins on conflicts."""
  out = dict(dst)
  for key, value in src.items():
    if isinstance(value, dict) and isinstance(out.get(key), dict):
      out[key] = _merge(out[key], value)
    else:
      out[key] = value
  return out


def load_configs():
  """configs.yaml with configs_maniskill.yaml merged on top.

  configs.yaml itself is never modified, so `python -m dreamerv3.main` and
  `python -m dreamerv3.main_htp` keep their exact previous config tree, and
  therefore the resolved_config_hash recorded in the published Atari
  checkpoint manifests.
  """
  load = lambda name: yaml.YAML(typ='safe').load(
      elements.Path(folder / name).read())
  configs = load('configs.yaml')
  extra = load('configs_maniskill.yaml')
  configs['defaults'] = _merge(configs['defaults'], extra.pop('defaults'))
  configs.update(extra)
  return configs


def main(argv=None):
  Agent = _resolve_agent()
  [elements.print(line) for line in Agent.banner]

  configs = load_configs()
  parsed, other = elements.Flags(configs=['defaults']).parse_known(argv)
  config = elements.Config(configs['defaults'])
  for name in parsed.configs:
    config = config.update(configs[name])
  config = elements.Flags(config).parse(other)
  config = config.update(logdir=os.path.expanduser(
      config.logdir.format(timestamp=elements.timestamp())))
  if os.environ.get('PAPER_DETERMINISTIC_UUID', '0') == '1':
    elements.UUID.reset(debug=True)

  suite = config.task.split('_', 1)[0]
  if suite != 'maniskill':
    raise NotImplementedError(
        f'dreamerv3.main_maniskill only runs maniskill tasks, got '
        f'{config.task!r}. Use dreamerv3.main_htp for everything else.')
  if config.script != 'train':
    raise NotImplementedError(
        f'ManiSkill only supports script=train, got {config.script}')

  if 'JOB_COMPLETION_INDEX' in os.environ:
    config = config.update(replica=int(os.environ['JOB_COMPLETION_INDEX']))
  print('Replica:', config.replica, '/', config.replicas)

  logdir = elements.Path(config.logdir)
  print('Logdir:', logdir)
  print('Run script:', config.script)
  logdir.mkdir()
  config.save(logdir / 'config.yaml')

  def init():
    elements.timer.global_timer.enabled = config.logger.timer

  portal.setup(
      errfile=config.errfile and logdir / 'error',
      clientkw=dict(logging_color='cyan'),
      serverkw=dict(logging_color='cyan'),
      initfns=[init],
      ipv6=config.ipv6,
  )

  args = elements.Config(
      **config.run,
      replica=config.replica,
      replicas=config.replicas,
      logdir=config.logdir,
      task=config.task,
      seed=config.seed,
      env=config.env,
      agent=config.agent,
      logger=config.logger,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      report_length=config.report_length,
      consec_train=config.consec_train,
      consec_report=config.consec_report,
      replay_context=config.replay_context,
  )
  # One env call, no subprocess: the vector width lives in
  # env.maniskill.num_envs, not run.envs.
  args = args.update(envs=1, debug=False)

  train_maniskill(
      bind(make_agent, config),
      bind(base.make_replay, config, 'replay'),
      bind(base.make_env, config),
      bind(base.make_stream, config),
      bind(make_logger, config),
      args)


def make_agent(config):
  Agent = _resolve_agent()
  # Build with num_envs=1 just for obs/act space discovery. This avoids
  # spinning up the full vector env twice.
  env = base.make_env(config, 0, num_envs=1)
  notlog = lambda k: not k.startswith('log/')
  obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
  act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
  env.close()
  if config.random_agent:
    return embodied.RandomAgent(obs_space, act_space)
  return Agent(obs_space, act_space, elements.Config(
      **config.agent,
      logdir=config.logdir,
      seed=config.seed,
      jax=config.jax,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      replay_context=config.replay_context,
      report_length=config.report_length,
      replica=config.replica,
      replicas=config.replicas,
  ))


def wandb_run_name(config):
  """dreamerv3-<task>-<obs_mode>-<seed>, unless the config names the run."""
  if config.logger.wandb_name:
    return str(config.logger.wandb_name)
  return '-'.join([
      'dreamerv3', config.task.split('_', 1)[1],
      str(config.env.maniskill.obs_mode), str(config.seed)])


def make_logger(config):
  # main_htp.make_logger reads its W&B settings from WANDB_* environment
  # variables, which is why configs_maniskill.yaml has to hand them over this
  # way rather than main_htp reading the config directly -- main_htp is on the
  # Atari path and stays untouched. setdefault keeps an already-exported
  # variable authoritative for sweeps and lets the config fill in the rest.
  for key, value in [
      ('WANDB_PROJECT', config.logger.wandb_project),
      ('WANDB_ENTITY', config.logger.wandb_entity),
      ('WANDB_GROUP', config.logger.wandb_group),
      ('WANDB_RUN_NAME', wandb_run_name(config)),
  ]:
    if value:
      os.environ.setdefault(key, str(value))
  # Keep ManiSkill metrics visible to terminal / normal logger outputs.
  extra = '|episode/score|epstats/log/|fps/|train/success'
  return base.make_logger(
      config.update({'logger.filter': config.logger.filter + extra}))


if __name__ == '__main__':
  main()
