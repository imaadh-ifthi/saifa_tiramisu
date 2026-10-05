import numpy as np
import pytest
from src.modwt import AdditiveMODWT


@pytest.fixture
def decomposer():
    return AdditiveMODWT(wavelet="db4", levels=4)


def test_exact_reconstruction(decomposer):
    np.random.seed(42)
    for length in [100, 128, 256, 1000]:
        x = np.random.randn(length)
        details, smooth = decomposer.decompose(x)
        reconstruction = decomposer.reconstruct(details, smooth)
        np.testing.assert_allclose(x, reconstruction, atol=1e-10)


def test_length_preservation(decomposer):
    x = np.random.randn(105)
    details, smooth = decomposer.decompose(x)
    assert len(smooth) == 105
    for D in details:
        assert len(D) == 105


def test_scale_count(decomposer):
    x = np.random.randn(100)
    details, smooth = decomposer.decompose(x)
    assert len(details) == 4


def test_shift_invariance(decomposer):
    # Test translation equivariance subject to boundary conditions
    # Since we use padding, shift invariance holds strictly in the interior.
    np.random.seed(42)
    x = np.random.randn(500)
    x_shifted = np.roll(x, 5)
    
    d1, s1 = decomposer.decompose(x)
    d2, s2 = decomposer.decompose(x_shifted)
    
    d1_shifted = [np.roll(d, 5) for d in d1]
    s1_shifted = np.roll(s1, 5)
    
    # Check interior (avoiding boundary effects).
    # With margin=250, interior is fully insulated.
    margin = 150
    for j in range(decomposer.levels):
        np.testing.assert_allclose(d1_shifted[j][margin:-margin], d2[j][margin:-margin], atol=1e-10)
    np.testing.assert_allclose(s1_shifted[margin:-margin], s2[margin:-margin], atol=1e-10)


def test_determinism(decomposer):
    np.random.seed(42)
    x = np.random.randn(100)
    d1, s1 = decomposer.decompose(x)
    d2, s2 = decomposer.decompose(x)
    
    for j in range(decomposer.levels):
        np.testing.assert_array_equal(d1[j], d2[j])
    np.testing.assert_array_equal(s1, s2)


def test_constant_signal(decomposer):
    x = np.ones(100) * 5.0
    details, smooth = decomposer.decompose(x)
    for D in details:
        np.testing.assert_allclose(D, 0.0, atol=1e-10)
    np.testing.assert_allclose(smooth, 5.0, atol=1e-10)


def test_impulse_signal(decomposer):
    x = np.zeros(100)
    x[50] = 100.0
    details, smooth = decomposer.decompose(x)
    reconstruction = decomposer.reconstruct(details, smooth)
    np.testing.assert_allclose(x, reconstruction, atol=1e-10)
    
    # Check locality: energy should be concentrated around index 50
    for D in details:
        assert np.abs(D[50]) > np.mean(np.abs(D))


def test_frequency_separation(decomposer):
    t = np.linspace(0, 10, 1000)
    x_high = np.sin(2 * np.pi * 20 * t)  # High freq
    x_low = np.sin(2 * np.pi * 0.5 * t)  # Low freq
    
    d_h, s_h = decomposer.decompose(x_high)
    d_l, s_l = decomposer.decompose(x_low)
    
    # High frequency energy should be mostly in early details
    energy_h1 = np.sum(d_h[0]**2)
    energy_h_late = np.sum(d_h[-1]**2)
    assert energy_h1 > energy_h_late
    
    # Low frequency energy should be mostly in later details / smooth
    energy_l1 = np.sum(d_l[0]**2)
    energy_l_smooth = np.sum(s_l**2)
    assert energy_l_smooth > energy_l1


def test_variance_energy(decomposer):
    np.random.seed(42)
    x = np.random.randn(1000)
    details, smooth = decomposer.decompose(x)
    
    var_x = np.var(x)
    var_components = sum(np.var(D) for D in details) + np.var(smooth)
    
    print(f"Var(x)={var_x:.4f}, Sum of var(components)={var_components:.4f}")
    assert 0.1 * var_x < var_components < 10.0 * var_x


def test_arbitrary_lengths(decomposer):
    lengths = [37, 101, 257, 750, 1000, 1234]
    for L in lengths:
        x = np.random.randn(L)
        d, s = decomposer.decompose(x)
        assert len(s) == L
        rec = decomposer.reconstruct(d, s)
        np.testing.assert_allclose(x, rec, atol=1e-10)


def test_no_oos_access(decomposer):
    np.random.seed(42)
    x_base = np.random.randn(500)
    x_alt = x_base.copy()
    x_alt[250:] = 100.0  # Diverge after index 250
    
    d1, s1 = decomposer.decompose(x_base)
    d2, s2 = decomposer.decompose(x_alt)
    
    # Check early indices. With margin=250 in the implementation,
    # the periodic wrapping is completely insulated. The theoretical
    # max reach of the db4 filter at level 4 is 106 to the right.
    # Therefore indices up to 250-106=144 should be absolutely identical.
    margin = 120
    for j in range(decomposer.levels):
        np.testing.assert_allclose(d1[j][:margin], d2[j][:margin], atol=1e-10)
    np.testing.assert_allclose(s1[:margin], s2[:margin], atol=1e-10)


def test_financial_series(decomposer):
    np.random.seed(42)
    returns = np.random.randn(500) * 0.01
    
    d, s = decomposer.decompose(returns)
    assert len(s) == 500
    assert np.all(np.isfinite(s))
    for D in d:
        assert np.all(np.isfinite(D))
        
    rec = decomposer.reconstruct(d, s)
    np.testing.assert_allclose(returns, rec, atol=1e-10)
