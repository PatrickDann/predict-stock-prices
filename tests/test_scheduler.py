"""Scheduler job registration (no jobs are executed)."""

from apscheduler.triggers.cron import CronTrigger

from market_intel.scheduler import build_scheduler


def _noop_session_factory():  # pragma: no cover - never called
    raise AssertionError("jobs must not run during registration tests")


def test_registers_all_jobs():
    sched = build_scheduler(
        _noop_session_factory,
        market_tickers=["AAPL"],
        fred_series=["GDP", "CPIAUCSL"],
        fred_api_key="KEY",
        gdelt_queries=["oil supply"],
    )
    jobs = {j.id: j for j in sched.get_jobs()}
    assert set(jobs) == {"market-csv-daily", "fred-daily", "gdelt-news"}
    assert all(isinstance(j.trigger, CronTrigger) for j in jobs.values())


def test_fred_skipped_without_api_key():
    sched = build_scheduler(
        _noop_session_factory,
        market_tickers=["AAPL"],
        fred_series=["GDP"],
        fred_api_key=None,
    )
    assert {j.id for j in sched.get_jobs()} == {"market-csv-daily"}


def test_no_config_no_jobs():
    sched = build_scheduler(_noop_session_factory)
    assert sched.get_jobs() == []
