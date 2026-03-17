import numpy as np
from gtda.homology import VietorisRipsPersistence
from persim import wasserstein
from ripser import ripser


def _random_distance_matrix(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    upper = rng.uniform(0.0, 1.0, size=(8, 8))
    symmetric = (upper + upper.T) / 2.0
    np.fill_diagonal(symmetric, 0.0)
    return symmetric


def test_ripser_and_wasserstein_smoke() -> None:
    distance_matrix_a = _random_distance_matrix(7)
    distance_matrix_b = _random_distance_matrix(11)

    ripser_result_a = ripser(distance_matrix_a, distance_matrix=True, maxdim=1)
    ripser_result_b = ripser(distance_matrix_b, distance_matrix=True, maxdim=1)

    stacked_diagram = np.vstack(
        [
            np.column_stack([diagram, np.full(len(diagram), dimension)])
            for dimension, diagram in enumerate(ripser_result_a["dgms"])
            if len(diagram) > 0
        ]
    )

    assert stacked_diagram.ndim == 2
    assert stacked_diagram.shape[1] == 3
    assert np.isfinite(stacked_diagram[:, 0]).all()
    assert (np.isfinite(stacked_diagram[:, 1]) | np.isinf(stacked_diagram[:, 1])).all()

    h0_distance = wasserstein(ripser_result_a["dgms"][0], ripser_result_b["dgms"][0])
    assert h0_distance >= 0


def test_giotto_vietoris_rips_smoke() -> None:
    distance_matrix = _random_distance_matrix(21)
    transformer = VietorisRipsPersistence(metric="precomputed", homology_dimensions=(0, 1))

    transformed = transformer.fit_transform(distance_matrix[np.newaxis, :, :])

    assert transformed.ndim == 3
    assert transformed.shape[0] == 1
    assert transformed.shape[2] == 3
