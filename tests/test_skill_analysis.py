import pytest

from trading_desk.engine.skill_analysis import (
    calibration_buckets,
    decompose_return,
    directional_hit,
    information_coefficient,
)


class TestInformationCoefficient:
    def test_perfect_positive_correlation_is_one(self):
        ic = information_coefficient([0.05, 0.10, 0.15], [0.01, 0.02, 0.03])
        assert ic == pytest.approx(1.0)

    def test_perfect_negative_correlation_is_minus_one(self):
        ic = information_coefficient([0.05, 0.10, 0.15], [0.03, 0.02, 0.01])
        assert ic == pytest.approx(-1.0)

    def test_no_variance_in_predictions_is_none(self):
        assert information_coefficient([0.05, 0.05, 0.05], [0.01, 0.02, 0.03]) is None

    def test_too_few_points_is_none(self):
        assert information_coefficient([0.05], [0.01]) is None

    def test_mismatched_lengths_is_none(self):
        assert information_coefficient([0.05, 0.06], [0.01]) is None


class TestDirectionalHit:
    def test_both_positive_is_hit(self):
        assert directional_hit(0.05, 0.02) is True

    def test_both_negative_is_hit(self):
        assert directional_hit(-0.05, -0.02) is True

    def test_opposite_signs_is_miss(self):
        assert directional_hit(0.05, -0.02) is False

    def test_zero_realized_is_miss(self):
        assert directional_hit(0.05, 0.0) is False


class TestCalibrationBuckets:
    def test_perfectly_calibrated_confidence_climbs_with_hit_rate(self):
        records = (
            [{"confidence": 0.3, "correct": False}] * 8 + [{"confidence": 0.3, "correct": True}] * 2
            + [{"confidence": 0.9, "correct": True}] * 9 + [{"confidence": 0.9, "correct": False}] * 1
        )
        buckets = calibration_buckets(records)
        low_bucket = next(b for b in buckets if b["range_low"] == 0.0)
        high_bucket = next(b for b in buckets if b["range_high"] == 1.0)
        assert low_bucket["hit_rate"] == pytest.approx(0.2)
        assert high_bucket["hit_rate"] == pytest.approx(0.9)
        assert high_bucket["hit_rate"] > low_bucket["hit_rate"]

    def test_empty_bucket_has_none_hit_rate(self):
        buckets = calibration_buckets([{"confidence": 0.9, "correct": True}])
        empty_bucket = next(b for b in buckets if b["range_low"] == 0.0)
        assert empty_bucket["hit_rate"] is None
        assert empty_bucket["count"] == 0

    def test_boundary_value_falls_in_last_bucket(self):
        buckets = calibration_buckets([{"confidence": 1.0, "correct": True}])
        last_bucket = buckets[-1]
        assert last_bucket["count"] == 1


class TestDecomposeReturn:
    def test_splits_beta_and_skill_contribution(self):
        result = decompose_return(total_return=0.12, benchmark_return=0.05, beta=1.2)
        assert result["beta_contribution"] == pytest.approx(0.06)
        assert result["skill_contribution"] == pytest.approx(0.06)

    def test_negative_skill_when_underperforming_beta_adjusted_benchmark(self):
        result = decompose_return(total_return=0.02, benchmark_return=0.05, beta=1.0)
        assert result["skill_contribution"] == pytest.approx(-0.03)
