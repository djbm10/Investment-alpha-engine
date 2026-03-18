from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import yaml
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from .config_loader import PipelineConfig, load_config
from .learning.bayesian_optimizer import run_bayesian_update
from .learning.mistake_analyzer import run_mistake_analysis
from .pipeline import run_daily_pipeline


@dataclass(frozen=True)
class SchedulerJobConfig:
    time: str
    timezone: str
    day: str | int | None = None


@dataclass(frozen=True)
class SchedulerConfig:
    daily_pipeline: SchedulerJobConfig
    weekly_mistake_analysis: SchedulerJobConfig
    monthly_bayesian_update: SchedulerJobConfig


class PipelineScheduler:
    def __init__(
        self,
        config_path: str | Path,
        *,
        schedule_config_path: str | Path = "config/schedule.yaml",
        mode: str | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path)
        self.mode = mode
        self.schedule_config_path = Path(schedule_config_path)
        self.schedule_config = load_schedule_config(self.schedule_config_path)
        self.scheduler = BlockingScheduler(timezone=_timezone(self.schedule_config.daily_pipeline.timezone))

    def setup_jobs(self) -> BlockingScheduler:
        self.scheduler.add_job(
            self._run_daily_pipeline,
            trigger=build_trigger(self.schedule_config.daily_pipeline, day_of_week="mon-fri"),
            id="daily-pipeline",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self._run_weekly_mistake_analysis,
            trigger=build_trigger(self.schedule_config.weekly_mistake_analysis, day_of_week=str(self.schedule_config.weekly_mistake_analysis.day)),
            id="weekly-mistake-analysis",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self._run_monthly_bayesian_update,
            trigger=build_trigger(self.schedule_config.monthly_bayesian_update, day=str(self.schedule_config.monthly_bayesian_update.day)),
            id="monthly-bayesian-update",
            replace_existing=True,
        )
        return self.scheduler

    def run(self) -> None:
        self.setup_jobs()
        self.scheduler.start()

    def _run_daily_pipeline(self) -> None:
        run_daily_pipeline(self.config_path, mode=self.mode)

    def _run_weekly_mistake_analysis(self) -> None:
        evaluation_date = self._scheduler_now(self.schedule_config.weekly_mistake_analysis.timezone).date()
        start_date = (pd.Timestamp(evaluation_date) - pd.Timedelta(days=6)).date().isoformat()
        end_date = evaluation_date.isoformat()
        run_mistake_analysis(self.config_path, start_date, end_date)

    def _run_monthly_bayesian_update(self) -> None:
        evaluation_date = self._scheduler_now(self.schedule_config.monthly_bayesian_update.timezone).date().isoformat()
        run_bayesian_update(self.config_path, evaluation_date)

    def _scheduler_now(self, timezone_name: str) -> datetime:
        return datetime.now(_timezone(timezone_name))


def load_schedule_config(schedule_config_path: str | Path = "config/schedule.yaml") -> SchedulerConfig:
    payload = yaml.safe_load(Path(schedule_config_path).read_text(encoding="utf-8")) or {}
    schedule = payload.get("schedule", {})
    daily = schedule.get("daily_pipeline", {})
    weekly = schedule.get("weekly_mistake_analysis", {})
    monthly = schedule.get("monthly_bayesian_update", {})
    timezone_name = str(daily.get("timezone", "US/Eastern"))
    return SchedulerConfig(
        daily_pipeline=SchedulerJobConfig(
            time=str(daily.get("time", "16:30")),
            timezone=timezone_name,
        ),
        weekly_mistake_analysis=SchedulerJobConfig(
            time=str(weekly.get("time", "17:00")),
            timezone=str(weekly.get("timezone", timezone_name)),
            day=str(weekly.get("day", "friday")),
        ),
        monthly_bayesian_update=SchedulerJobConfig(
            time=str(monthly.get("time", "17:00")),
            timezone=str(monthly.get("timezone", timezone_name)),
            day=int(monthly.get("day", 1)),
        ),
    )


def build_trigger(job_config: SchedulerJobConfig, **cron_kwargs: str) -> CronTrigger:
    hour, minute = [int(part) for part in job_config.time.split(":", maxsplit=1)]
    normalized_kwargs = {
        key: _normalize_cron_value(value)
        for key, value in cron_kwargs.items()
    }
    return CronTrigger(
        hour=hour,
        minute=minute,
        timezone=_timezone(job_config.timezone),
        **normalized_kwargs,
    )


def next_run_time(config: PipelineConfig | None = None, schedule_config_path: str | Path = "config/schedule.yaml") -> datetime:
    del config
    schedule_config = load_schedule_config(schedule_config_path)
    now = datetime.now(_timezone(schedule_config.daily_pipeline.timezone))
    trigger = build_trigger(schedule_config.daily_pipeline, day_of_week="mon-fri")
    next_fire_time = trigger.get_next_fire_time(None, now)
    assert next_fire_time is not None
    return next_fire_time


def _timezone(timezone_name: str) -> ZoneInfo:
    aliases = {
        "US/Eastern": "America/New_York",
    }
    candidate = aliases.get(timezone_name, timezone_name)
    try:
        return ZoneInfo(candidate)
    except ZoneInfoNotFoundError:
        if candidate != "America/New_York":
            return ZoneInfo("America/New_York")
        raise


def _normalize_cron_value(value: str) -> str:
    weekday_aliases = {
        "monday": "mon",
        "tuesday": "tue",
        "wednesday": "wed",
        "thursday": "thu",
        "friday": "fri",
        "saturday": "sat",
        "sunday": "sun",
    }
    return weekday_aliases.get(value.lower(), value)
