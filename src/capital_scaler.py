from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from .config_loader import PipelineConfig, load_config


class CapitalScaler:
    def __init__(self, config: PipelineConfig, live_start_date: str | date | pd.Timestamp) -> None:
        self.config = config
        self.live_start_date = pd.Timestamp(live_start_date).normalize()
        self.schedule = config.deployment.scaling_schedule

    def get_scale_factor(self, current_date: str | date | pd.Timestamp) -> float:
        current = pd.Timestamp(current_date).normalize()
        elapsed_days = max(int((current - self.live_start_date).days), 0)
        live_week = (elapsed_days // 7) + 1
        if live_week <= 4:
            return float(self.schedule.weeks_1_4)
        if live_week <= 12:
            return float(self.schedule.weeks_5_12)
        if live_week <= 24:
            return float(self.schedule.weeks_13_24)
        return float(self.schedule.weeks_25_plus)

    def apply_scaling(
        self,
        target_positions: dict[str, dict[str, float]],
        portfolio_value: float,
        current_date: str | date | pd.Timestamp,
    ) -> dict[str, dict[str, float]]:
        del portfolio_value
        scale_factor = self.get_scale_factor(current_date)
        scaled_positions: dict[str, dict[str, float]] = {}
        for asset, payload in target_positions.items():
            scaled_quantity = float(payload.get("quantity", 0.0)) * scale_factor
            price = float(payload.get("price", 0.0))
            scaled_positions[str(asset)] = {
                **payload,
                "quantity": float(scaled_quantity),
                "market_value": float(scaled_quantity * price),
            }
        return scaled_positions


def get_live_scale_factor(
    config_path: str | Path,
    *,
    live_start_date: str | date | pd.Timestamp,
    current_date: str | date | pd.Timestamp,
) -> float:
    config = load_config(config_path)
    scaler = CapitalScaler(config, live_start_date)
    return scaler.get_scale_factor(current_date)
