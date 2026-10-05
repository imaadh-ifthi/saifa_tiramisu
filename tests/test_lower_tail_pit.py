"""
Tests for the corrected lower-tail EVT Probability Integral Transform.

These tests verify the survival-probability formulation:
    u = p_lower * (1 - G(excess))

where G is the GPD CDF and excess = lower_thr - z.
"""

import numpy as np
import pytest

from src.marginals import MarginalARQGARCH


# ------------------------------------------------------------------
# Fixture: fit a marginal model on synthetic data with known properties
# ------------------------------------------------------------------
@pytest.fixture
def fitted_model():
    """
    Create a MarginalARQGARCH model fitted on synthetic standardized
    residuals with heavier-than-normal tails, ensuring enough data for
    a stable GPD fit.
    """
    rng = np.random.default_rng(42)

    # Generate t-distributed residuals (heavy tails) for realistic data.
    z = rng.standard_t(df=5, size=2000)
    z = np.clip(z, -15.0, 15.0)

    model = MarginalARQGARCH(tail_quantile=0.95)

    # Directly set internal state to bypass GARCH fitting
    # (we only need the PIT / inverse PIT machinery).
    model.z = z
    model.scale = 1.0
    model.forecast_mean = 0.0
    model.forecast_sigma = 1.0

    model.p_lower = 0.05
    model.p_upper = 0.95

    model.lower_thr = float(np.quantile(z, 0.05))
    model.upper_thr = float(np.quantile(z, 0.95))

    upper_excess = z[z > model.upper_thr] - model.upper_thr
    lower_excess = model.lower_thr - z[z < model.lower_thr]

    model.xi_u, model.beta_u = model._fit_gpd(upper_excess)
    model.xi_l, model.beta_l = model._fit_gpd(lower_excess)

    interior = z[(z >= model.lower_thr) & (z <= model.upper_thr)]
    model.interior = np.sort(interior)
    model.fitted = True

    return model


# ==================================================================
# 1. MONOTONICITY
# ==================================================================
class TestMonotonicity:
    """
    For z1 < z2 < z3 < ... < lower_thr, the resulting PIT values
    must satisfy U1 < U2 < U3 < ... < U_threshold.
    """

    def test_lower_tail_pit_is_monotone_increasing_in_z(self, fitted_model):
        model = fitted_model

        # Generate a sequence of increasingly extreme z below the threshold.
        z_vals = np.linspace(model.lower_thr - 8.0, model.lower_thr - 0.01, 50)
        u_vals = model.transform(z_vals)

        # z is increasing, so u must also be strictly increasing.
        diffs = np.diff(u_vals)
        assert np.all(diffs > 0), (
            f"Lower-tail PIT is not monotone increasing. "
            f"Diffs with negative values: {diffs[diffs <= 0]}"
        )

    def test_lower_tail_pit_is_monotone_with_extreme_range(self, fitted_model):
        model = fitted_model

        # Even wider range including very extreme values.
        z_vals = np.linspace(model.lower_thr - 15.0, model.lower_thr, 100)
        u_vals = model.transform(z_vals)

        diffs = np.diff(u_vals)
        assert np.all(diffs >= 0), (
            "Lower-tail PIT failed monotonicity over extreme range."
        )


# ==================================================================
# 2. THRESHOLD BEHAVIOUR
# ==================================================================
class TestThresholdBehaviour:
    """
    A value at the lower threshold should map approximately to p_lower.
    """

    def test_z_at_lower_threshold_maps_near_p_lower(self, fitted_model):
        model = fitted_model

        z_at_thr = np.array([model.lower_thr])
        u_at_thr = model.transform(z_at_thr)

        assert abs(u_at_thr[0] - model.p_lower) < 0.01, (
            f"z at lower_thr mapped to u={u_at_thr[0]:.6f}, "
            f"expected ~{model.p_lower:.6f}"
        )

    def test_z_just_below_threshold_maps_just_below_p_lower(self, fitted_model):
        model = fitted_model

        z_just_below = np.array([model.lower_thr - 0.001])
        u = model.transform(z_just_below)

        assert u[0] < model.p_lower, (
            f"z just below threshold mapped to u={u[0]:.6f} "
            f"which is not below p_lower={model.p_lower:.6f}"
        )
        assert u[0] > model.p_lower * 0.5, (
            f"z barely below threshold mapped too far from p_lower: u={u[0]:.6f}"
        )


# ==================================================================
# 3. EXTREME BEHAVIOUR
# ==================================================================
class TestExtremeBehaviour:
    """
    A sufficiently extreme negative z should map close to 0.
    """

    def test_very_extreme_z_maps_near_zero(self, fitted_model):
        model = fitted_model

        z_extreme = np.array([model.lower_thr - 10.0])
        u_extreme = model.transform(z_extreme)

        assert u_extreme[0] < 0.005, (
            f"Very extreme z mapped to u={u_extreme[0]:.6f}, expected near 0"
        )

    def test_all_pit_values_in_valid_range(self, fitted_model):
        model = fitted_model

        z_range = np.linspace(model.lower_thr - 15.0, model.lower_thr, 200)
        u = model.transform(z_range)

        assert np.all(u > 0), "Some PIT values are <= 0"
        assert np.all(u < 1), "Some PIT values are >= 1"
        assert np.all(np.isfinite(u)), "Non-finite PIT values detected"


# ==================================================================
# 4. ROUND-TRIP
# ==================================================================
class TestRoundTrip:
    """
    For lower-tail uniform values u in (0, p_lower), verify:
        u -> inverse_pit(u) -> transform(z) ≈ u
    """

    def test_lower_tail_round_trip(self, fitted_model):
        model = fitted_model

        # Representative lower-tail uniforms.
        u_original = np.array([0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.049])
        u_original = u_original[u_original < model.p_lower]

        z_recovered = model.inverse_pit(u_original)
        u_reconstructed = model.transform(z_recovered)

        np.testing.assert_allclose(
            u_reconstructed, u_original,
            atol=1e-4, rtol=1e-3,
            err_msg="Lower-tail round-trip failed"
        )

    def test_upper_tail_round_trip(self, fitted_model):
        model = fitted_model

        # Representative upper-tail uniforms.
        u_original = np.array([0.951, 0.96, 0.97, 0.98, 0.99, 0.999])
        u_original = u_original[u_original > model.p_upper]

        z_recovered = model.inverse_pit(u_original)
        u_reconstructed = model.transform(z_recovered)

        np.testing.assert_allclose(
            u_reconstructed, u_original,
            atol=1e-4, rtol=1e-3,
            err_msg="Upper-tail round-trip failed"
        )

    def test_interior_round_trip(self, fitted_model):
        model = fitted_model

        # Representative interior uniforms.
        u_original = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        u_original = u_original[
            (u_original > model.p_lower) & (u_original < model.p_upper)
        ]

        z_recovered = model.inverse_pit(u_original)
        u_reconstructed = model.transform(z_recovered)

        np.testing.assert_allclose(
            u_reconstructed, u_original,
            atol=0.02, rtol=0.02,
            err_msg="Interior round-trip failed"
        )


# ==================================================================
# 5. UPPER / INTERIOR REGRESSION
# ==================================================================
class TestUpperInteriorRegression:
    """
    Verify that upper-tail and interior transforms still behave correctly.
    """

    def test_upper_tail_monotone_increasing(self, fitted_model):
        model = fitted_model

        z_vals = np.linspace(model.upper_thr + 0.01, model.upper_thr + 8.0, 50)
        u_vals = model.transform(z_vals)

        diffs = np.diff(u_vals)
        assert np.all(diffs >= 0), "Upper-tail PIT is not monotone increasing."

    def test_upper_threshold_maps_near_p_upper(self, fitted_model):
        model = fitted_model

        z_at_upper = np.array([model.upper_thr])
        u = model.transform(z_at_upper)

        assert abs(u[0] - model.p_upper) < 0.01, (
            f"z at upper_thr mapped to u={u[0]:.6f}, expected ~{model.p_upper:.6f}"
        )

    def test_interior_maps_between_p_lower_and_p_upper(self, fitted_model):
        model = fitted_model

        z_interior = np.linspace(
            model.lower_thr + 0.1, model.upper_thr - 0.1, 100
        )
        u = model.transform(z_interior)

        assert np.all(u >= model.p_lower - 0.01), (
            "Interior z mapped below p_lower"
        )
        assert np.all(u <= model.p_upper + 0.01), (
            "Interior z mapped above p_upper"
        )

    def test_full_range_monotone(self, fitted_model):
        model = fitted_model

        z_vals = np.linspace(
            model.lower_thr - 5.0, model.upper_thr + 5.0, 300
        )
        u_vals = model.transform(z_vals)

        diffs = np.diff(u_vals)
        assert np.all(diffs >= -1e-10), (
            f"Full-range PIT is not monotone. Min diff: {np.min(diffs)}"
        )


# ==================================================================
# 6. NUMERICAL SAFETY
# ==================================================================
class TestNumericalSafety:
    """
    Verify clipping and NaN/inf safety.
    """

    def test_no_nans_or_infs_in_transform(self, fitted_model):
        model = fitted_model

        z = np.linspace(-15.0, 15.0, 500)
        u = model.transform(z)

        assert np.all(np.isfinite(u)), "Transform produced NaN/inf"
        assert np.all(u >= 1e-6), "Transform produced values below clip"
        assert np.all(u <= 1.0 - 1e-6), "Transform produced values above clip"

    def test_no_nans_or_infs_in_inverse(self, fitted_model):
        model = fitted_model

        u = np.linspace(1e-6, 1.0 - 1e-6, 500)
        z = model.inverse_pit(u)

        assert np.all(np.isfinite(z)), "Inverse PIT produced NaN/inf"

    def test_inverse_pit_lower_boundary(self, fitted_model):
        model = fitted_model

        # Very small u (near 0) should produce very negative z without NaN.
        u = np.array([1e-6, 1e-5, 1e-4, 1e-3])
        z = model.inverse_pit(u)

        assert np.all(np.isfinite(z)), "Inverse PIT at small u produced NaN/inf"
        assert np.all(z < model.lower_thr), (
            "Inverse PIT at small u did not produce z below lower_thr"
        )


# ==================================================================
# 7. CONCRETE NUMERICAL EXAMPLES (for validation output)
# ==================================================================
class TestConcreteExamples:
    """
    Print concrete numerical examples for the validation report.
    """

    def test_concrete_examples(self, fitted_model):
        model = fitted_model

        print("\n" + "=" * 60)
        print("CONCRETE NUMERICAL EXAMPLES")
        print("=" * 60)
        print(f"Model parameters:")
        print(f"  lower_thr = {model.lower_thr:.6f}")
        print(f"  upper_thr = {model.upper_thr:.6f}")
        print(f"  p_lower   = {model.p_lower:.6f}")
        print(f"  p_upper   = {model.p_upper:.6f}")
        print(f"  xi_l      = {model.xi_l:.6f}")
        print(f"  beta_l    = {model.beta_l:.6f}")
        print()

        # Very extreme negative z.
        z_extreme = np.array([model.lower_thr - 10.0])
        u_extreme = model.transform(z_extreme)
        print(f"Very extreme z = {z_extreme[0]:.4f}  ->  u = {u_extreme[0]:.8f}")
        assert u_extreme[0] < 0.01, "Extreme z should map near 0"

        # Moderately extreme z.
        z_moderate = np.array([model.lower_thr - 3.0])
        u_moderate = model.transform(z_moderate)
        print(f"Moderate   z = {z_moderate[0]:.4f}  ->  u = {u_moderate[0]:.8f}")

        # At threshold.
        z_thr = np.array([model.lower_thr])
        u_thr = model.transform(z_thr)
        print(f"Threshold  z = {z_thr[0]:.4f}  ->  u = {u_thr[0]:.8f}")
        print(f"  (expected ~{model.p_lower:.6f})")
        assert abs(u_thr[0] - model.p_lower) < 0.01

        # Just inside interior.
        z_interior = np.array([0.0])
        u_interior = model.transform(z_interior)
        print(f"Interior   z = {z_interior[0]:.4f}  ->  u = {u_interior[0]:.8f}")

        # Upper threshold.
        z_upper = np.array([model.upper_thr])
        u_upper = model.transform(z_upper)
        print(f"Upper thr  z = {z_upper[0]:.4f}  ->  u = {u_upper[0]:.8f}")
        print(f"  (expected ~{model.p_upper:.6f})")

        print("=" * 60)
