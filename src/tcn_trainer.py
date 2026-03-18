from __future__ import annotations

import os
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .config_loader import Phase4Config, PipelineConfig
from .tcn_model import TCNPredictor


@dataclass(frozen=True)
class SequenceDatasetBundle:
    inputs: np.ndarray
    targets: np.ndarray
    signal_dates: list[pd.Timestamp]
    target_dates: list[pd.Timestamp]


@dataclass(frozen=True)
class TrainedModelResult:
    model: TCNPredictor
    best_validation_loss: float
    epochs_trained: int


@dataclass(frozen=True)
class ModelTrainingArtifact:
    state_dict: dict[str, torch.Tensor]
    best_validation_loss: float
    epochs_trained: int


@dataclass(frozen=True)
class EnsembleTrainingResult:
    models: list[TCNPredictor]
    validation_losses: list[float]
    signal_dates: list[pd.Timestamp]
    target_dates: list[pd.Timestamp]


class TCNTrainer:
    def __init__(self, config: PipelineConfig | Phase4Config) -> None:
        self.config = config.phase4 if isinstance(config, PipelineConfig) else config
        _configure_torch_threads(1)
        self.device = torch.device("cpu")
        self.validation_improvement_tol = 1e-4
        self.max_workers = min(self.config.n_ensemble, max(os.cpu_count() or 1, 1))

    def prepare_dataset(
        self,
        features_by_date: dict[pd.Timestamp, np.ndarray],
        residuals_by_date: dict[pd.Timestamp, np.ndarray],
    ) -> SequenceDatasetBundle:
        ordered_dates = sorted(set(features_by_date).intersection(residuals_by_date))
        if len(ordered_dates) <= self.config.sequence_length:
            raise ValueError("Not enough dated feature history to build the requested TCN sequences.")

        sequences: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        signal_dates: list[pd.Timestamp] = []
        target_dates: list[pd.Timestamp] = []

        for end_idx in range(self.config.sequence_length - 1, len(ordered_dates) - 1):
            signal_date = pd.Timestamp(ordered_dates[end_idx])
            target_date = pd.Timestamp(ordered_dates[end_idx + 1])
            sequence_dates = ordered_dates[end_idx - self.config.sequence_length + 1 : end_idx + 1]
            sequences.append(np.stack([features_by_date[pd.Timestamp(date)] for date in sequence_dates], axis=0))
            targets.append(np.asarray(residuals_by_date[target_date], dtype=float))
            signal_dates.append(signal_date)
            target_dates.append(target_date)

        return SequenceDatasetBundle(
            inputs=np.asarray(sequences, dtype=np.float32),
            targets=np.asarray(targets, dtype=np.float32),
            signal_dates=signal_dates,
            target_dates=target_dates,
        )

    def train_single_model(
        self,
        train_data: SequenceDatasetBundle,
        val_data: SequenceDatasetBundle,
        seed: int,
    ) -> TrainedModelResult:
        artifact = _train_model_core(
            config=self.config,
            train_inputs=train_data.inputs,
            train_targets=train_data.targets,
            val_inputs=val_data.inputs,
            val_targets=val_data.targets,
            seed=seed,
        )
        model = self._instantiate_model(
            n_features=train_data.inputs.shape[-1],
            n_assets=train_data.inputs.shape[-2],
        )
        model.load_state_dict(artifact.state_dict)
        model = model.to(self.device)
        model.eval()
        return TrainedModelResult(
            model=model,
            best_validation_loss=artifact.best_validation_loss,
            epochs_trained=artifact.epochs_trained,
        )

    def train_ensemble(
        self,
        features_by_date: dict[pd.Timestamp, np.ndarray],
        residuals_by_date: dict[pd.Timestamp, np.ndarray],
        train_end_date: pd.Timestamp | str,
        validation_start_date: pd.Timestamp | str | None = None,
    ) -> EnsembleTrainingResult:
        dataset = self.prepare_dataset(features_by_date, residuals_by_date)
        train_end_timestamp = pd.Timestamp(train_end_date)

        if validation_start_date is None:
            train_bundle, val_bundle = _split_by_fraction(dataset, train_end_timestamp, self.config.validation_fraction)
        else:
            train_bundle, val_bundle = _split_by_dates(dataset, pd.Timestamp(validation_start_date), train_end_timestamp)

        models: list[TCNPredictor] = []
        validation_losses: list[float] = []
        artifacts = self._train_model_artifacts(train_bundle, val_bundle)
        for artifact in artifacts:
            model = self._instantiate_model(
                n_features=train_bundle.inputs.shape[-1],
                n_assets=train_bundle.inputs.shape[-2],
            )
            model.load_state_dict(artifact.state_dict)
            model = model.to(self.device)
            model.eval()
            models.append(model)
            validation_losses.append(artifact.best_validation_loss)

        return EnsembleTrainingResult(
            models=models,
            validation_losses=validation_losses,
            signal_dates=val_bundle.signal_dates,
            target_dates=val_bundle.target_dates,
        )

    def predict(
        self,
        models: list[TCNPredictor],
        feature_sequence: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not models:
            raise ValueError("At least one trained TCN model is required for prediction.")
        feature_tensor = torch.tensor(feature_sequence[None, ...], dtype=torch.float32, device=self.device)
        means: list[np.ndarray] = []
        variances: list[np.ndarray] = []
        for model in models:
            model.eval()
            with torch.no_grad():
                prediction = model(feature_tensor).cpu().numpy()[0]
            means.append(prediction[:, 0])
            variances.append(np.exp(np.clip(prediction[:, 1], -10.0, 10.0)))

        mean_stack = np.stack(means, axis=0)
        variance_stack = np.stack(variances, axis=0)
        ensemble_mean = mean_stack.mean(axis=0)
        ensemble_variance = variance_stack.mean(axis=0) + mean_stack.var(axis=0, ddof=0)
        ensemble_uncertainty = np.sqrt(np.clip(ensemble_variance, a_min=1e-6, a_max=None))
        return ensemble_mean.astype(float), ensemble_uncertainty.astype(float)

    def _evaluate(
        self,
        model: TCNPredictor,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        model.eval()
        with torch.no_grad():
            predictions = model(inputs)
            loss = model.loss(predictions, targets)
        return float(loss.item())

    def _instantiate_model(self, n_features: int, n_assets: int) -> TCNPredictor:
        return TCNPredictor(
            n_features=n_features,
            n_assets=n_assets,
            hidden_channels=self.config.hidden_channels,
            n_blocks=self.config.n_blocks,
            dropout=self.config.dropout,
        )

    def _train_model_artifacts(
        self,
        train_bundle: SequenceDatasetBundle,
        val_bundle: SequenceDatasetBundle,
    ) -> list[ModelTrainingArtifact]:
        seeds = list(range(self.config.n_ensemble))
        if self.max_workers <= 1 or len(seeds) <= 1:
            return [
                _train_model_core(
                    config=self.config,
                    train_inputs=train_bundle.inputs,
                    train_targets=train_bundle.targets,
                    val_inputs=val_bundle.inputs,
                    val_targets=val_bundle.targets,
                    seed=seed,
                )
                for seed in seeds
            ]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(
                    _train_model_core,
                    self.config,
                    train_bundle.inputs,
                    train_bundle.targets,
                    val_bundle.inputs,
                    val_bundle.targets,
                    seed,
                )
                for seed in seeds
            ]
            return [future.result() for future in futures]


def _train_model_core(
    config: Phase4Config,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    val_inputs: np.ndarray,
    val_targets: np.ndarray,
    seed: int,
) -> ModelTrainingArtifact:
    _configure_torch_threads(1)
    _set_random_seed(seed)
    model = TCNPredictor(
        n_features=train_inputs.shape[-1],
        n_assets=train_inputs.shape[-2],
        hidden_channels=config.hidden_channels,
        n_blocks=config.n_blocks,
        dropout=config.dropout,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_inputs_tensor = torch.tensor(train_inputs, dtype=torch.float32)
    train_targets_tensor = torch.tensor(train_targets, dtype=torch.float32)
    val_inputs_tensor = torch.tensor(val_inputs, dtype=torch.float32)
    val_targets_tensor = torch.tensor(val_targets, dtype=torch.float32)

    best_state = _snapshot_state_dict(model)
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    epochs_trained = 0

    for epoch in range(config.max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        predictions = model(train_inputs_tensor)
        loss = model.loss(predictions, train_targets_tensor)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_predictions = model(val_inputs_tensor)
            validation_loss = float(model.loss(validation_predictions, val_targets_tensor).item())
        epochs_trained = epoch + 1

        if validation_loss < (best_validation_loss - 1e-4):
            best_validation_loss = validation_loss
            best_state = _snapshot_state_dict(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    return ModelTrainingArtifact(
        state_dict=best_state,
        best_validation_loss=best_validation_loss,
        epochs_trained=epochs_trained,
    )


def _split_by_fraction(
    dataset: SequenceDatasetBundle,
    train_end_date: pd.Timestamp,
    validation_fraction: float,
) -> tuple[SequenceDatasetBundle, SequenceDatasetBundle]:
    eligible_indices = [idx for idx, date in enumerate(dataset.signal_dates) if pd.Timestamp(date) <= train_end_date]
    if len(eligible_indices) < 2:
        raise ValueError("Not enough eligible samples to form train and validation splits.")
    validation_count = max(1, int(round(len(eligible_indices) * validation_fraction)))
    validation_indices = eligible_indices[-validation_count:]
    train_indices = eligible_indices[:-validation_count]
    if not train_indices:
        raise ValueError("Validation fraction left no training samples for the TCN.")
    return _select_bundle(dataset, train_indices), _select_bundle(dataset, validation_indices)


def _split_by_dates(
    dataset: SequenceDatasetBundle,
    validation_start_date: pd.Timestamp,
    validation_end_date: pd.Timestamp,
) -> tuple[SequenceDatasetBundle, SequenceDatasetBundle]:
    train_indices = [
        idx
        for idx, date in enumerate(dataset.signal_dates)
        if pd.Timestamp(date) < validation_start_date
    ]
    validation_indices = [
        idx
        for idx, date in enumerate(dataset.signal_dates)
        if validation_start_date <= pd.Timestamp(date) <= validation_end_date
    ]
    if not train_indices or not validation_indices:
        raise ValueError("Date-based TCN split did not produce both training and validation samples.")
    return _select_bundle(dataset, train_indices), _select_bundle(dataset, validation_indices)


def _select_bundle(dataset: SequenceDatasetBundle, indices: list[int]) -> SequenceDatasetBundle:
    return SequenceDatasetBundle(
        inputs=dataset.inputs[indices],
        targets=dataset.targets[indices],
        signal_dates=[dataset.signal_dates[index] for index in indices],
        target_dates=[dataset.target_dates[index] for index in indices],
    )


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _configure_torch_threads(thread_count: int) -> None:
    torch.set_num_threads(thread_count)
    try:
        torch.set_num_interop_threads(thread_count)
    except RuntimeError:
        pass


def _snapshot_state_dict(model: TCNPredictor) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _bundle_to_tensors(
    bundle: SequenceDatasetBundle,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if bundle.inputs.size == 0 or bundle.targets.size == 0:
        raise ValueError("TCN training requires non-empty input and target tensors.")
    return (
        torch.tensor(bundle.inputs, dtype=torch.float32, device=device),
        torch.tensor(bundle.targets, dtype=torch.float32, device=device),
    )
