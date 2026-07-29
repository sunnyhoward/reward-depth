import pytest

from replay_kfac_ewc import fit_calibration


def test_calibration_recovers_scale_and_slope():
    predicted = [1e-4, 1e-3, 1e-2, 1e-1]
    measured = [3 * value for value in predicted]
    report = fit_calibration(predicted, measured)
    assert report.log_log_slope == pytest.approx(1.0)
    assert report.spearman_correlation == pytest.approx(1.0)
    assert report.kl_per_penalty_geometric_mean == pytest.approx(3.0)
    assert report.median_kl_per_penalty == pytest.approx(3.0)
