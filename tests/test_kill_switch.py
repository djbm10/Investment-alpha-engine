import pandas as pd

from src.config_loader import KillSwitchConfig
from src.learning.kill_switch import StrategyKillSwitch


def test_kill_switch_reduces_quarantines_and_reactivates() -> None:
    config = KillSwitchConfig(
        reduction_threshold=-0.5,
        quarantine_threshold=-0.5,
        reactivation_threshold=0.0,
        reactivation_days=40,
        reduction_lookback_days=60,
        quarantine_lookback_days=120,
        reactivation_lookback_days=40,
    )
    kill_switch = StrategyKillSwitch(config)

    dates = pd.bdate_range("2024-01-01", periods=260)
    returns = pd.Series(
        ([0.001] * 60) + ([-0.003] * 120) + ([0.002] * 80),
        index=dates,
    )

    reduced_eval = kill_switch.evaluate("A", returns, dates[140])
    quarantined_eval = kill_switch.evaluate("A", returns, dates[190])
    reactivated_eval = kill_switch.evaluate("A", returns, dates[-1])

    assert reduced_eval.status in {"REDUCED", "QUARANTINED"}
    assert quarantined_eval.status == "QUARANTINED"
    assert reactivated_eval.status == "REACTIVATE"
