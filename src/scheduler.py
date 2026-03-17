from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from .config_loader import PipelineConfig


def build_trigger(config: PipelineConfig) -> CronTrigger:
    return CronTrigger(
        hour=config.schedule.hour,
        minute=config.schedule.minute,
        timezone=ZoneInfo(config.schedule.timezone),
    )


def next_run_time(config: PipelineConfig) -> datetime:
    now = datetime.now(ZoneInfo(config.schedule.timezone))
    trigger = build_trigger(config)
    next_fire_time = trigger.get_next_fire_time(None, now)
    assert next_fire_time is not None
    return next_fire_time


def build_scheduler(config: PipelineConfig, job_callable) -> BlockingScheduler:
    scheduler = BlockingScheduler(timezone=ZoneInfo(config.schedule.timezone))
    scheduler.add_job(
        job_callable,
        trigger=build_trigger(config),
        id="phase1-daily-pipeline",
        replace_existing=True,
    )
    return scheduler
