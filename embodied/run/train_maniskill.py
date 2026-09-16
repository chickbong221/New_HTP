"""Training loop for ManiSkill's GPU-vectorized envs.

This is embodied/run/train.py (New_HTP's paper-artifact version) with the
ManiSkill additions grafted in:

  - a BatchedDriver over one env object holding N GPU sub-envs
  - TD-MPC2-style per-vector-episode success logging
  - a separate eval env (is_eval=True, its own vector width) + eval videos

embodied/run/train.py is untouched; nothing here runs unless the caller is
dreamerv3.main_maniskill.

Exact action milestones (run.exact_env_action_budget) are rejected below: the
vector env advances the canonical action counter by num_envs per driver call
and would step over a milestone, so such a checkpoint cannot be labeled exact.
"""

import collections
from functools import partial as bind
from time import time

import elements
import embodied
import numpy as np

from ..core.driver_batched import BatchedDriver
from .paper_artifacts import PaperArtifactWriter
from .train import TraceableRatio


def train(make_agent, make_replay, make_env, make_stream, make_logger, args):

  agent = make_agent()
  replay = make_replay()
  logger = make_logger()

  logdir = elements.Path(args.logdir)
  step = logger.step
  if bool(getattr(args, 'exact_env_action_budget', False)):
    raise RuntimeError(
        'run.exact_env_action_budget is not supported for ManiSkill: the '
        'vector env advances the action counter by env.maniskill.num_envs per '
        'driver call and can step over a milestone, so the resulting '
        'checkpoint must not be labeled exact. Use embodied/run/train.py with '
        'run.envs=1 for exact-budget runs.')
  env_action_steps = elements.Counter()
  reset_callbacks = elements.Counter()
  replay_insertions = elements.Counter()
  raw_ale_frames = elements.Counter()
  usage = elements.Usage(**args.usage)
  train_agg = elements.Agg()
  epstats = elements.Agg()
  episodes = collections.defaultdict(elements.Agg)
  policy_fps = elements.FPS()
  train_fps = elements.FPS()
  paper = PaperArtifactWriter(logdir, args)
  paper.bind_runtime_counters(
      env_action_steps, reset_callbacks, replay_insertions, raw_ale_frames,
      optimizer_updates=lambda: int(agent.n_updates))

  batch_steps = args.batch_size * args.batch_length
  should_train = TraceableRatio(args.train_ratio / batch_steps)
  should_log = embodied.LocalClock(args.log_every)
  should_report = embodied.LocalClock(args.report_every)
  # run.save_every is wall-clock seconds, not steps: embodied.LocalClock
  # compares time.time() and ignores the step it is handed. Three regimes:
  #   < 0  never checkpoint, matching the reference's maniskill presets
  #   = 0  no periodic saves; only the single final save at the end of the run
  #   > 0  save every N seconds, plus the final save
  should_save = args.save_every >= 0 and embodied.LocalClock(args.save_every)

  start_time = time()
  train_vector_episode_done = [False]
  train_episode_idx = [0]
  eval_stat_next = [True]
  eval_video_next = [False]

  def _arg(name, default):
    if hasattr(args, 'get'):
      return args.get(name, default)
    return getattr(args, name, default)

  # TD-MPC2 defaults: num_eval_envs=4, eval_episodes_per_env=2.
  # Total eval episodes per call = eval_envs * eval_eps; per-episode metrics
  # are averaged in `all_metrics` below, so any positive eval_eps is supported.
  eval_freq = int(float(_arg('eval_freq', 50000)))
  eval_video_freq = int(float(_arg('eval_video_freq', 0)))
  eval_num_envs = int(_arg('eval_envs', 4))
  eval_episodes_per_env = max(int(_arg('eval_eps', 1)), 1)

  # ManiSkill episode-level metrics that TD-MPC2 logs once per completed
  # training vector episode batch as mean over cfg.num_envs.
  maniskill_metric_keys = (
      'log/success_once',
      'log/success_at_end',
      'log/fail_once',
  )
  maniskill_batch = {
      'count': 0,
      'values': collections.defaultdict(list),
  }

  def _to_numpy(value):
    try:
      import torch
      if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    except Exception:
      pass
    return np.asarray(value)

  def _flush_logger_before_direct_wandb():
    # Direct wandb.log(step=...) must not overtake queued Dreamer summaries.
    # Flush first so W&B sees monotonically increasing steps.
    try:
      logger.write()
    except Exception as exc:
      print(f'Could not flush Dreamer logger before direct W&B log: {exc}')

  def _wandb_log_direct(payload, step_value):
    if not payload:
      return
    try:
      import wandb
      if wandb.run is not None:
        wandb.log(payload, step=int(step_value))
    except Exception as exc:
      print(f'Could not log directly to W&B: {exc}')

  def log_maniskill_vector_batch():
    values = maniskill_batch['values']

    payload = {}
    for key, vals in values.items():
      if vals:
        # log/success_once -> train/success_once
        payload[f'train/{key[len("log/"):]}'] = float(np.mean(vals))

    if payload:
      # Do not call wandb.log() directly for train metrics. Dreamer's logger may
      # still hold queued summaries for earlier callback steps. Logging through
      # the same logger stream prevents W&B step-order warnings.
      logger.add(payload)
      logger.write()

    values.clear()
    maniskill_batch['count'] = 0
    train_vector_episode_done[0] = True

  def _tile_images(images, nrows):
    try:
      from mani_skill.utils.visualization.misc import tile_images
      return tile_images(images, nrows=nrows)
    except Exception:
      images = np.asarray(images)
      n = len(images)
      nrows = max(1, int(nrows))
      ncols = int(np.ceil(n / nrows))
      h, w, c = images.shape[1:]
      canvas = np.zeros((nrows * h, ncols * w, c), dtype=images.dtype)
      for idx, image in enumerate(images):
        r, col = divmod(idx, ncols)
        canvas[r * h:(r + 1) * h, col * w:(col + 1) * w] = image
      return canvas

  def _log_wandb_eval_video(video_frames, step_value, fps=15,
                            key='videos/eval_video'):
    """TD-MPC2-style W&B video logging under videos/eval_video.

    video_frames is a list of arrays [num_eval_envs, height, width, 3], one per
    eval timestep. We tile vector-env frames per timestep, convert to W&B's
    [T, C, H, W] layout, and log at the current training step.
    """
    if not video_frames:
      return
    try:
      import wandb
      if wandb.run is None:
        return
      frames = [_to_numpy(frame) for frame in video_frames]
      frames = [frame[None] if frame.ndim == 3 else frame for frame in frames]
      frames = np.stack(frames, axis=0)  # [T, N, H, W, 3]
      nrows = max(1, int(np.sqrt(frames.shape[1])))
      tiled = [_tile_images(rgbs, nrows=nrows) for rgbs in frames]
      tiled = np.stack(tiled, axis=0)  # [T, H, W, 3]
      _wandb_log_direct({
          key: wandb.Video(
              tiled.transpose(0, 3, 1, 2), fps=fps, format='mp4')},
          step_value)
    except Exception as exc:
      print(f'Could not log eval video to W&B: {exc}')

  def _report_video_to_rgb(video):
    """Collapse an openloop report grid [T, H, W, C] down to 3 channels.

    The reference does this inside dreamerv3/agent.py (to_report_rgb) before
    the batch grid is formed. agent.py is left untouched here, so the same
    reduction is applied to the finished grid instead: multi-camera / stacked
    views are tiled along the width axis. With a single camera and
    num_frames=1 (C == 3) this returns the input unchanged.
    """
    channels = video.shape[-1]
    if channels == 3:
      return video
    if channels == 1:
      return np.repeat(video, 3, axis=-1)
    if channels % 3 == 0:
      views = channels // 3
      T_, H_, W_, _ = video.shape
      video = video.reshape((T_, H_, W_, views, 3))
      video = video.transpose((0, 1, 3, 2, 4))
      return video.reshape((T_, H_, views * W_, 3))
    return video[..., :3]

  @elements.timer.section('logfn')
  def logfn(tran, worker):
    episode = episodes[worker]
    tran['is_first'] and episode.reset()
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')
    for key, value in tran.items():
      if value.dtype == np.uint8 and value.ndim == 3 and args.log_policy_video:
        if worker == 0:
          episode.add(f'policy_{key}', value, agg='stack')
      elif key.startswith('log/'):
        # TD-MPC2 logs ManiSkill success metrics immediately once per completed
        # vector episode batch. Do not also average them through Dreamer's
        # epstats path.
        if getattr(driver, 'batched', False) and key in maniskill_metric_keys:
          continue
        assert value.ndim == 0, (key, value.shape, value.dtype)
        episode.add(key + '/avg', value, agg='avg')
        episode.add(key + '/max', value, agg='max')
        episode.add(key + '/sum', value, agg='sum')

    vector_batch_complete = False
    if getattr(driver, 'batched', False) and tran['is_last']:
      for key in maniskill_metric_keys:
        if key not in tran:
          continue
        value = np.asarray(tran[key])
        assert value.ndim == 0, (key, value.shape, value.dtype)
        maniskill_batch['values'][key].append(float(value))

      maniskill_batch['count'] += 1
      vector_batch_complete = maniskill_batch['count'] >= driver.length

    if tran['is_last']:
      result = episode.result()
      score = result.pop('score')
      length = result.pop('length')
      logger.add({'score': score, 'length': length}, prefix='episode')
      paper.write_episode(step, score, length)
      rew = result.pop('rewards')
      if len(rew) > 1:
        result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
      epstats.add(result)

    # Log after episode summaries for the final worker have been queued, so
    # logger.write() flushes the whole completed vector batch in order.
    if vector_batch_complete:
      log_maniskill_vector_batch()

  # One env object holding env.num_envs GPU sub-envs. args.envs is unused here;
  # the vector width comes from env.maniskill.num_envs.
  train_env = make_env(0)
  driver = BatchedDriver(train_env, num_envs=train_env.num_envs)
  driver.on_step(lambda tran, _: step.increment())
  def actioncountfn(tran, worker):
    if bool(tran['log/action_executed']):
      env_action_steps.increment()
    else:
      reset_callbacks.increment()
    raw_ale_frames.increment(int(tran.get('log/raw_ale_frames', 0)))
  driver.on_step(actioncountfn)
  driver.on_step(lambda tran, _: policy_fps.step())
  def replayfn(tran, worker):
    replay.add(tran, worker)
    replay_insertions.increment()
  driver.on_step(replayfn)
  driver.on_step(logfn)
  driver.on_step(lambda tran, worker: paper.write_action_trace(
      step, tran, worker, int(getattr(agent, 'n_updates', 0))))

  # TD-MPC2 creates a separate eval env once at startup and reuses it. It uses
  # cfg.num_eval_envs, not cfg.num_envs. Here eval_num_envs comes from
  # run.eval_envs, which should be 4 for TD-MPC2 default comparability.
  # run.eval_envs=0 skips the eval env entirely (saves GPU memory).
  eval_env = None
  if eval_num_envs > 0:
    eval_env = make_env(
        0,
        num_envs=eval_num_envs,
        is_eval=True,
        eval_reconfiguration_frequency=int(_arg(
            'eval_reconfiguration_frequency', 1)),
    )

  def _zero_actions(env, num_envs):
    acts = {
        k: np.zeros((num_envs,) + v.shape, v.dtype)
        for k, v in env.act_space.items()}
    acts['reset'] = np.ones(num_envs, bool)
    return acts

  def _mask(value, mask):
    while mask.ndim < value.ndim:
      mask = mask[..., None]
    return value * mask.astype(value.dtype)

  def _mean_episode_metrics(metrics):
    out = {}
    for key, value in metrics.items():
      arr = _to_numpy(value).astype(np.float32)
      if arr.size:
        out[key] = float(np.mean(arr))
    return out

  @elements.timer.section('evalfn')
  def evalfn(do_video=False):
    if eval_env is None:
      return

    # Ensure direct eval/* and videos/eval_video logs do not advance W&B past
    # queued Dreamer summaries from earlier steps.
    _flush_logger_before_direct_wandb()

    eval_policy = lambda *xs: agent.policy(*xs, mode='eval')
    horizon = int(getattr(eval_env, '_max_episode_steps', 10000))
    all_metrics = collections.defaultdict(list)
    video_frames = []

    for eval_episode_idx in range(eval_episodes_per_env):
      carry = agent.init_policy(eval_num_envs)
      acts = _zero_actions(eval_env, eval_num_envs)
      fallback_return = np.zeros(eval_num_envs, np.float32)
      fallback_len = np.zeros(eval_num_envs, np.float32)
      final_metrics = None

      for t in range(horizon + 2):
        obs = eval_env.step(acts)
        obs = {k: np.asarray(v) for k, v in obs.items()}

        if do_video and hasattr(eval_env, 'render'):
          try:
            frame = np.asarray(eval_env.render())
            video_frames.append(frame[:4].copy())
          except Exception as exc:
            if t == 0:
              print(f'Could not render eval video frames: {exc}')

        not_first = ~np.asarray(obs['is_first'], dtype=bool)
        fallback_return += obs['reward'].astype(np.float32) * not_first
        fallback_len += not_first.astype(np.float32)

        done = np.asarray(obs['is_last'], dtype=bool)
        if bool(done[0]):
          # Prefer full ManiSkill final_info['episode'], matching TD-MPC2's
          # final_info_metrics(). Fall back to values available in obs.
          if hasattr(eval_env, 'last_episode_metrics'):
            final_metrics = eval_env.last_episode_metrics()
          if not final_metrics:
            final_metrics = {
                'return': fallback_return,
                'episode_len': fallback_len,
                'reward': fallback_return / np.maximum(fallback_len, 1.0),
            }
            for name in ('success_once', 'success_at_end',
                         'fail_once', 'fail_at_end'):
              key = f'log/{name}'
              if key in obs:
                final_metrics[name] = obs[key]
          break

        # log/ keys are intentionally excluded from policy input.
        obs_for_policy = {k: v for k, v in obs.items()
                          if not k.startswith('log/')}
        carry, acts, outs = eval_policy(carry, obs_for_policy)
        assert all(k not in acts for k in outs), (
            list(outs.keys()), list(acts.keys()))

        if done.any():
          mask = ~done
          acts = {k: _mask(v, mask) for k, v in acts.items()}
        acts = {**acts, 'reset': done.copy()}
      else:
        raise RuntimeError(
            f'Eval episode did not finish within {horizon + 2} steps.')

      for key, value in _mean_episode_metrics(final_metrics).items():
        all_metrics[key].append(value)

    eval_metrics = {
        key: float(np.mean(values))
        for key, values in all_metrics.items()}
    eval_metrics.update({
        'step': int(step),
        'episode': int(train_episode_idx[0]),
        'total_time': time() - start_time,
    })

    # TD-MPC2 Logger.log(eval_metrics, 'eval') writes eval/<key> and uses
    # W&B step=d['step'].
    _wandb_log_direct({f'eval/{k}': v for k, v in eval_metrics.items()}, step)

    if do_video:
      _log_wandb_eval_video(video_frames, step, fps=15,
                            key='videos/eval_video')

    shown = ', '.join(
        f'{k}: {v:.4g}' for k, v in eval_metrics.items()
        if isinstance(v, (int, float, np.integer, np.floating)))
    suffix = ' [+video]' if do_video else ''
    print(f'Eval metrics{suffix} | {shown}')

  stream_train = iter(agent.stream(make_stream(replay, 'train')))
  stream_report = iter(agent.report_stream(make_stream(replay, 'report')))

  carry_train = [agent.init_train(args.batch_size)]
  carry_report = agent.init_report(args.batch_size)
  updates_before_event = collections.defaultdict(int)

  def trainfn(tran, worker):
    updates_before_event[worker] = int(getattr(agent, 'n_updates', 0))
    if len(replay) < args.batch_size * args.batch_length:
      paper.write_update_event(
          step, requested_updates=0, executed_updates=0,
          optimizer_updates_cumulative=int(getattr(agent, 'n_updates', 0)),
          is_prefill=True, is_compile_only=False,
          scheduler_accumulator_before=should_train.prev,
          scheduler_accumulator_after=should_train.prev)
      return
    requested, sched_before, sched_after = should_train(step)
    executed = 0
    for _ in range(requested):
      with elements.timer.section('stream_next'):
        batch = next(stream_train)
      before = int(getattr(agent, 'n_updates', 0))
      paper.write_batch_trace(step, executed, batch, before)
      carry_train[0], outs, mets = agent.train(carry_train[0], batch)
      executed += 1
      paper.write_batch_trace(
          step, executed - 1, batch, before,
          optimizer_updates_after=int(getattr(agent, 'n_updates', 0)),
          metrics={f'train/{k}': v for k, v in mets.items()})
      train_fps.step(batch_steps)
      if 'replay' in outs:
        replay.update(outs['replay'])
      train_agg.add(mets, prefix='train')
    paper.write_update_event(
        step, requested_updates=requested, executed_updates=executed,
        optimizer_updates_cumulative=int(getattr(agent, 'n_updates', 0)),
        is_prefill=False, is_compile_only=False,
        scheduler_accumulator_before=sched_before,
        scheduler_accumulator_after=sched_after)
  driver.on_step(trainfn)

  def actiontracefn(tran, worker):
    paper.write_env_action_event(
        step, tran, worker, env_action_steps, reset_callbacks,
        replay_insertions,
        updates_before_event[worker], int(getattr(agent, 'n_updates', 0)))
  driver.on_step(actiontracefn)

  cp = elements.Checkpoint(logdir / 'ckpt')
  cp.step = step
  cp.agent = agent
  # The reference omits the replay buffer from ManiSkill checkpoints: a
  # 300k-transition RGB buffer makes every save stall the run.
  if args.from_checkpoint:
    elements.checkpoint.load(args.from_checkpoint, dict(
        agent=bind(agent.load, regex=args.from_checkpoint_regex)))
  # Only resume from the run's own ckpt dir if it already exists; do not
  # auto-save a fresh checkpoint at start.
  if (logdir / 'ckpt').exists():
    cp.load()

  print('Start training loop')
  policy = lambda *xs: agent.policy(*xs, mode='train')
  driver.reset(agent.init_policy)

  # evaluate once at step 0 before the first train rollout
  if eval_env is not None and eval_stat_next[0]:
    do_video = eval_video_freq > 0
    eval_video_next[0] = False
    evalfn(do_video=do_video)
    eval_stat_next[0] = False

  while step < args.steps:

    train_vector_episode_done[0] = False
    driver(policy, steps=10)

    if eval_env is not None and eval_freq > 0:
      if int(step) > 0 and int(step) % eval_freq < driver.length:
        eval_stat_next[0] = True
      if (eval_video_freq > 0 and int(step) > 0
          and int(step) % eval_video_freq < driver.length):
        eval_video_next[0] = True
      if train_vector_episode_done[0] and eval_stat_next[0]:
        evalfn(do_video=eval_video_next[0])
        eval_stat_next[0] = False
        eval_video_next[0] = False

    if train_vector_episode_done[0]:
      train_episode_idx[0] += 1

    if should_report(step) and len(replay):
      agg = elements.Agg()
      for _ in range(args.consec_report * args.report_batches):
        carry_report, mets = agent.report(carry_report, next(stream_report))
        agg.add(mets)
      report_mets = agg.result()
      video_mets = {
          k: v for k, v in report_mets.items()
          if isinstance(v, np.ndarray) and v.ndim == 4 and v.dtype == np.uint8}
      scalar_mets = {
          k: v for k, v in report_mets.items() if k not in video_mets}
      logger.add(scalar_mets, prefix='report')
      if video_mets:
        try:
          import wandb
          if wandb.run is not None:
            _flush_logger_before_direct_wandb()
            payload = {}
            for k, v in video_mets.items():
              # v shape: [T, H, W, C] uint8 — wandb.Video expects [T, C, H, W]
              v = _report_video_to_rgb(v)
              payload[f'report/{k}'] = wandb.Video(
                  v.transpose(0, 3, 1, 2),
                  fps=int(_arg('report_fps', 4)),
                  format='mp4')
            _wandb_log_direct(payload, step)
        except Exception as exc:
          print(f'Could not log report videos to W&B as mp4: {exc}')

    if should_log(step):
      train_stats = train_agg.result()
      ep_stats = epstats.result()
      replay_stats = replay.stats()
      usage_stats = usage.stats()
      fps_stats = {
          'fps/policy': policy_fps.result(),
          'fps/train': train_fps.result(),
      }
      timer_stats = {'timer': elements.timer.stats()['summary']}
      logger.add(train_stats)
      logger.add(ep_stats, prefix='epstats')
      logger.add(replay_stats, prefix='replay')
      logger.add(usage_stats, prefix='usage')
      logger.add(fps_stats)
      logger.add(timer_stats)
      paper_stats = {}
      paper_stats.update(train_stats)
      paper_stats.update({f'epstats/{k}': v for k, v in ep_stats.items()})
      paper_stats.update({f'replay/{k}': v for k, v in replay_stats.items()})
      paper_stats.update({f'usage/{k}': v for k, v in usage_stats.items()})
      paper_stats.update(fps_stats)
      paper_stats.update(timer_stats)
      paper.write_train_metrics(step, paper_stats)
      logger.write()

    if should_save and should_save(step):
      cp.save()

  if args.save_every >= 0:
    cp.save()
  paper.finalize(step, checkpoint_path=logdir / 'ckpt')
  if eval_env is not None:
    eval_env.close()
  driver.close()
  logger.close()
