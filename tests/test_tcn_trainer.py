import numpy as np
import pandas as pd

from src.config_loader import Phase4Config
from src.tcn_trainer import TCNTrainer


def _phase4_config() -> Phase4Config:
    return Phase4Config(
        tcn_enabled=True,
        hidden_channels=8,
        n_blocks=2,
        dropout=0.0,
        n_ensemble=2,
        learning_rate=0.001,
        weight_decay=0.0001,
        max_epochs=2,
        patience=1,
        batch_size=4,
        sequence_length=5,
        validation_fraction=0.2,
        reversion_confirm_threshold=0.5,
    )


def _feature_history() -> tuple[dict[pd.Timestamp, np.ndarray], dict[pd.Timestamp, np.ndarray]]:
    dates = pd.bdate_range("2022-01-03", periods=40)
    features_by_date: dict[pd.Timestamp, np.ndarray] = {}
    residuals_by_date: dict[pd.Timestamp, np.ndarray] = {}
    for idx, date in enumerate(dates):
        base = np.full((3, 4), idx / 100.0, dtype=float)
        base[:, 0] += np.array([0.1, 0.2, 0.3])
        features_by_date[pd.Timestamp(date)] = base
        residuals_by_date[pd.Timestamp(date)] = np.array([0.01, -0.02, 0.03], dtype=float) + idx / 1000.0
    return features_by_date, residuals_by_date


def test_prepare_dataset_builds_temporally_ordered_sequences() -> None:
    trainer = TCNTrainer(_phase4_config())
    features_by_date, residuals_by_date = _feature_history()

    dataset = trainer.prepare_dataset(features_by_date, residuals_by_date)

    assert dataset.inputs.shape[1:] == (5, 3, 4)
    assert dataset.targets.shape[1] == 3
    assert dataset.signal_dates[0] < dataset.target_dates[0]
    assert len(dataset.signal_dates) == len(dataset.target_dates) == dataset.inputs.shape[0]


def test_predict_combines_ensemble_mean_and_uncertainty() -> None:
    trainer = TCNTrainer(_phase4_config())
    features_by_date, residuals_by_date = _feature_history()
    ensemble = trainer.train_ensemble(
        features_by_date,
        residuals_by_date,
        train_end_date=pd.Timestamp("2022-02-18"),
    )

    feature_sequence = np.stack(list(features_by_date.values())[:5], axis=0)
    mean, uncertainty = trainer.predict(ensemble.models, feature_sequence)

    assert mean.shape == (3,)
    assert uncertainty.shape == (3,)
    assert np.isfinite(mean).all()
    assert np.isfinite(uncertainty).all()
    assert (uncertainty > 0).all()
