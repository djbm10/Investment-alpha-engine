from src.scheduler import PipelineScheduler, load_schedule_config, next_run_time


def test_load_schedule_config_reads_yaml_values() -> None:
    config = load_schedule_config("config/schedule.yaml")

    assert config.daily_pipeline.time == "16:30"
    assert config.daily_pipeline.timezone == "US/Eastern"
    assert config.weekly_mistake_analysis.day == "friday"
    assert config.monthly_bayesian_update.day == 1


def test_pipeline_scheduler_registers_expected_jobs() -> None:
    scheduler = PipelineScheduler("config/phase8.yaml", mode="paper")
    blocking_scheduler = scheduler.setup_jobs()
    job_ids = {job.id for job in blocking_scheduler.get_jobs()}

    assert job_ids == {"daily-pipeline", "weekly-mistake-analysis", "monthly-bayesian-update"}


def test_next_run_time_returns_datetime() -> None:
    upcoming = next_run_time(schedule_config_path="config/schedule.yaml")

    assert upcoming.tzinfo is not None
