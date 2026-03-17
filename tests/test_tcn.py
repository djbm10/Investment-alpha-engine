import torch

from src.tcn_model import TCNPredictor


def test_tcn_forward_pass_returns_expected_shape() -> None:
    model = TCNPredictor(n_features=19, n_assets=8, hidden_channels=32, n_blocks=3, dropout=0.2)
    x = torch.randn(4, 20, 8, 19)

    predictions = model(x)

    assert predictions.shape == (4, 8, 2)


def test_tcn_encode_sequence_is_causal() -> None:
    torch.manual_seed(0)
    model = TCNPredictor(n_features=19, n_assets=8, hidden_channels=16, n_blocks=3, dropout=0.0)
    model.eval()
    x = torch.randn(2, 20, 8, 19)
    perturbed = x.clone()
    perturbed[:, -1, :, :] += 100.0

    with torch.no_grad():
        baseline = model.encode_sequence(x)
        changed = model.encode_sequence(perturbed)

    assert torch.allclose(baseline[:, :, :-1, :], changed[:, :, :-1, :], atol=1e-5)


def test_tcn_loss_is_finite_and_positive() -> None:
    torch.manual_seed(1)
    model = TCNPredictor(n_features=19, n_assets=8, hidden_channels=32, n_blocks=3, dropout=0.2)
    x = torch.randn(3, 20, 8, 19)
    targets = torch.randn(3, 8)

    predictions = model(x)
    loss = model.loss(predictions, targets)

    assert torch.isfinite(loss)
    assert float(loss.item()) > 0.0
