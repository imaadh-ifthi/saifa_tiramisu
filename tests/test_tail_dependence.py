import numpy as np
import pytest
from src.tail_dependence import empirical_tail_dependence, bootstrap_tail_dependence

def test_independence_sanity():
    np.random.seed(42)
    u1 = np.random.uniform(0, 1, 100000)
    u2 = np.random.uniform(0, 1, 100000)
    q = 0.05
    lam_L, lam_U, _, _, _ = empirical_tail_dependence(u1, u2, q)
    # Expected value is q = 0.05. Allow some noise.
    assert np.isclose(lam_L, q, atol=0.01)
    assert np.isclose(lam_U, q, atol=0.01)

def test_perfect_positive_dependence():
    np.random.seed(42)
    u1 = np.linspace(0.0001, 0.9999, 10000)
    u2 = u1.copy()
    q = 0.05
    lam_L, lam_U, _, _, _ = empirical_tail_dependence(u1, u2, q)
    assert np.isclose(lam_L, 1.0)
    assert np.isclose(lam_U, 1.0)

def test_exact_known_synthetic_dependence():
    np.random.seed(42)
    from scipy.stats import norm
    # Bivariate normal with rho=0.8
    cov = [[1, 0.8], [0.8, 1]]
    samples = np.random.multivariate_normal([0, 0], cov, 100000)
    u1 = norm.cdf(samples[:, 0])
    u2 = norm.cdf(samples[:, 1])
    
    q = 0.05
    lam_L, lam_U, _, _, _ = empirical_tail_dependence(u1, u2, q)
    # With rho=0.8, lower tail dependence > 0.05
    assert lam_L > 0.2
    assert lam_U > 0.2

def test_monotonicity():
    np.random.seed(42)
    # We create two scenarios: perfectly independent, and somewhat dependent
    u_ind1 = np.random.uniform(0, 1, 50000)
    u_ind2 = np.random.uniform(0, 1, 50000)
    
    u_base = np.random.uniform(0, 1, 50000)
    u_dep1 = u_base
    # mix with independent
    u_dep2 = 0.8 * u_base + 0.2 * np.random.uniform(0, 1, 50000)
    # convert back to rank / uniforms roughly
    from scipy.stats import rankdata
    u_dep2 = rankdata(u_dep2) / len(u_dep2)
    
    q = 0.05
    lam_ind, _, _, _, _ = empirical_tail_dependence(u_ind1, u_ind2, q)
    lam_dep, _, _, _, _ = empirical_tail_dependence(u_dep1, u_dep2, q)
    
    assert lam_dep > lam_ind

def test_bootstrap_determinism():
    np.random.seed(42)
    u1 = np.random.uniform(0, 1, 1000)
    u2 = np.random.uniform(0, 1, 1000)
    q = 0.05
    
    boot1_L, boot1_U = bootstrap_tail_dependence(u1, u2, q, n_boot=50, seed=123)
    boot2_L, boot2_U = bootstrap_tail_dependence(u1, u2, q, n_boot=50, seed=123)
    
    np.testing.assert_array_equal(boot1_L, boot2_L)
    np.testing.assert_array_equal(boot1_U, boot2_U)

def test_bootstrap_bounds():
    np.random.seed(42)
    u1 = np.random.uniform(0, 1, 1000)
    u2 = np.random.uniform(0, 1, 1000)
    q = 0.05
    
    boot_L, boot_U = bootstrap_tail_dependence(u1, u2, q, n_boot=50, seed=42)
    
    assert np.all(boot_L >= 0.0)
    assert np.all(boot_L <= 1.0 / q)
    assert np.all(boot_U >= 0.0)
    assert np.all(boot_U <= 1.0 / q)

def test_threshold_sensitivity():
    np.random.seed(42)
    u1 = np.random.uniform(0, 1, 1000)
    u2 = np.random.uniform(0, 1, 1000)
    
    for q in [0.025, 0.05, 0.10]:
        lam_L, lam_U, n, ev_L, ev_U = empirical_tail_dependence(u1, u2, q)
        assert lam_L >= 0
        assert lam_U >= 0
        assert n == 1000

def test_small_tail_sample_warning():
    np.random.seed(42)
    # Too small sample for 2.5% tail => 0 events usually
    u1 = np.random.uniform(0, 1, 10)
    u2 = np.random.uniform(0, 1, 10)
    q = 0.025
    lam_L, lam_U, n, ev_L, ev_U = empirical_tail_dependence(u1, u2, q)
    assert ev_L == 0
    assert ev_U == 0

