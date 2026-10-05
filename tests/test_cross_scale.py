import numpy as np
import pytest
from scipy.stats import norm, spearmanr
from src.cross_scale import CrossScaleCopulaCoupler

@pytest.fixture
def synthetic_stress():
    np.random.seed(42)
    T = 200
    n_scales = 6
    # Create some dependence structure among scales
    base = np.random.randn(T)
    stress = np.zeros((T, n_scales))
    for j in range(n_scales):
        stress[:, j] = base * (0.5 + 0.1*j) + np.random.randn(T)
    return stress

@pytest.fixture
def coupler(synthetic_stress):
    c = CrossScaleCopulaCoupler(scale_names=[f"Scale{j}" for j in range(6)])
    c.fit(synthetic_stress)
    return c

@pytest.fixture
def synthetic_samples():
    np.random.seed(100)
    n_sim = 500
    n_scales = 6
    n_assets = 5
    samples = {}
    for j in range(n_scales):
        # 5 assets per scale
        samples[f"Scale{j}"] = np.random.uniform(size=(n_sim, n_assets))
    return samples

def test_fit_shape(coupler):
    assert coupler.corr_matrix.shape == (6, 6)
    assert coupler.n_scales == 6

def test_valid_correlation_matrix(coupler):
    corr = coupler.corr_matrix
    # Symmetry
    np.testing.assert_allclose(corr, corr.T, atol=1e-10)
    # Diagonal 1
    np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-10)
    # Finite
    assert np.all(np.isfinite(corr))
    # Positive semidefinite
    eigvals = np.linalg.eigvalsh(corr)
    assert np.all(eigvals >= -1e-8)

def test_determinism(coupler, synthetic_samples):
    res1 = coupler.couple(synthetic_samples, seed=123)
    res2 = coupler.couple(synthetic_samples, seed=123)
    
    for scale in synthetic_samples.keys():
        np.testing.assert_array_equal(res1[scale], res2[scale])

def test_row_permutation_property(coupler, synthetic_samples):
    res = coupler.couple(synthetic_samples, seed=42)
    
    for scale in synthetic_samples.keys():
        orig = synthetic_samples[scale]
        coupled = res[scale]
        
        # Check that rows are identical, just permuted.
        # We can sort the rows based on the first column to match them.
        orig_sorted = orig[np.argsort(orig[:, 0])]
        coupled_sorted = coupled[np.argsort(coupled[:, 0])]
        
        np.testing.assert_allclose(orig_sorted, coupled_sorted, atol=1e-10)

def test_within_scale_dependence_preservation(coupler, synthetic_samples):
    res = coupler.couple(synthetic_samples, seed=42)
    
    for scale in synthetic_samples.keys():
        orig = synthetic_samples[scale]
        coupled = res[scale]
        
        # Spearman correlation matrix should be identical
        corr_orig, _ = spearmanr(orig)
        corr_coupled, _ = spearmanr(coupled)
        
        np.testing.assert_allclose(corr_orig, corr_coupled, atol=1e-10)

def test_marginal_preservation(coupler, synthetic_samples):
    res = coupler.couple(synthetic_samples, seed=42)
    
    for scale in synthetic_samples.keys():
        orig = synthetic_samples[scale]
        coupled = res[scale]
        
        for i in range(orig.shape[1]):
            orig_col = np.sort(orig[:, i])
            coupled_col = np.sort(coupled[:, i])
            np.testing.assert_allclose(orig_col, coupled_col, atol=1e-10)

def test_cross_scale_coupling(coupler, synthetic_samples):
    # Before coupling, they are independent
    # After coupling, the stress rank correlation should resemble the target correlation
    res = coupler.couple(synthetic_samples, seed=42)
    
    sim_before_corr = coupler.diag_simulated_before_corr
    sim_after_corr = coupler.diag_simulated_after_corr
    target_corr = coupler.diag_historical_corr
    
    # Check that after coupling, the off-diagonal correlation is closer to target
    # than before coupling (where it should be near 0)
    err_before = np.linalg.norm(sim_before_corr - target_corr)
    err_after = np.linalg.norm(sim_after_corr - target_corr)
    
    assert err_after < err_before
    # And after should be reasonably close to target
    assert np.max(np.abs(sim_after_corr - target_corr)) < 0.2

def test_independence_fallback():
    # If the matrix is identity, coupling should effectively leave them independent
    # or just shuffle them independently
    np.random.seed(42)
    c = CrossScaleCopulaCoupler(scale_names=[f"Scale{j}" for j in range(2)])
    stress = np.random.randn(100, 2)
    # force identical so correlation is 1, but we manually override
    c.fit(stress)
    c.corr_matrix = np.eye(2)
    
    samples = {
        "Scale0": np.random.uniform(size=(1000, 5)),
        "Scale1": np.random.uniform(size=(1000, 5))
    }
    
    res = c.couple(samples, seed=42)
    after_corr = c.diag_simulated_after_corr
    
    # Should be close to identity
    assert np.max(np.abs(after_corr - np.eye(2))) < 0.1

def test_no_look_ahead(synthetic_stress):
    # Fit only uses provided data, no future access
    stress_train = synthetic_stress[:100]
    c1 = CrossScaleCopulaCoupler(scale_names=["S0", "S1", "S2", "S3", "S4", "S5"])
    c1.fit(stress_train)
    
    stress_full = synthetic_stress.copy()
    c2 = CrossScaleCopulaCoupler(scale_names=["S0", "S1", "S2", "S3", "S4", "S5"])
    c2.fit(stress_full[:100])
    
    # Changing future data shouldn't change the fit on past data
    stress_full[100:] = 1000
    c3 = CrossScaleCopulaCoupler(scale_names=["S0", "S1", "S2", "S3", "S4", "S5"])
    c3.fit(stress_full[:100])
    
    np.testing.assert_allclose(c1.corr_matrix, c2.corr_matrix)
    np.testing.assert_allclose(c1.corr_matrix, c3.corr_matrix)

def test_scale_order(coupler, synthetic_samples):
    names = list(synthetic_samples.keys())
    res = coupler.couple(synthetic_samples, seed=42)
    assert list(res.keys()) == names

def test_dimension_safety(coupler):
    for n_sim in [100, 1000, 5000]:
        samples = {}
        for j in range(6):
            samples[f"Scale{j}"] = np.random.uniform(size=(n_sim, 5))
        res = coupler.couple(samples, seed=42)
        assert len(res["Scale0"]) == n_sim

def test_small_sample_safety():
    # Only 1 observation!
    stress = np.random.randn(1, 6)
    c = CrossScaleCopulaCoupler(scale_names=[f"Scale{j}" for j in range(6)])
    c.fit(stress)
    assert c.fitted
    assert c.corr_matrix.shape == (6, 6)
    
    # Coupling with small simulation size
    samples = {}
    for j in range(6):
        samples[f"Scale{j}"] = np.random.uniform(size=(1, 5))
    res = c.couple(samples)
    assert len(res["Scale0"]) == 1

