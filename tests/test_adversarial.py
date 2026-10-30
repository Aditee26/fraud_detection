import numpy as np, pytest
tf = pytest.importorskip("tensorflow")
from fraudlab import nn
from fraudlab.adversarial import fgsm, pgd, random_noise, mutable_mask, check_constraints, adversarial_train


@pytest.fixture(scope="module")
def trained(prepared):
    m = nn.build_ann(prepared["Xtr"].shape[1], seed=0); nn.fit_model(m, prepared["Xtr"], prepared["ytr"], prepared["Xva"], prepared["yva"], epochs=5)
    lo, hi = prepared["pp"].feature_bounds(prepared["Xtr"]); return m, lo, hi, mutable_mask(prepared["pp"].columns)


def bce(m, X, y):
    p = np.clip(m.predict(X, verbose=0).ravel(), 1e-6, 1 - 1e-6); return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def test_eps_zero_is_identity(trained, prepared):
    m, lo, hi, mask = trained; X = prepared["Xte"][:50]; y = prepared["yte"][:50]
    assert np.allclose(fgsm(m, X, y, 0.0), X) and np.allclose(pgd(m, X, y, 0.0, steps=3), X)


@pytest.mark.parametrize("attack", ["fgsm", "pgd"])
def test_budget_bounds_and_immutables_respected(trained, prepared, attack):
    m, lo, hi, mask = trained; X, y = prepared["Xte"][:200], prepared["yte"][:200]
    Xa = fgsm(m, X, y, 0.4, lo, hi, mask) if attack == "fgsm" else pgd(m, X, y, 0.4, steps=5, lo=lo, hi=hi, mask=mask)
    c = check_constraints(X, Xa, 0.4, lo, hi, mask)
    assert c["within_budget"] and c["within_bounds"] and c["immutable_unchanged"]
    assert (mask == 0).any() and np.allclose(Xa[:, mask == 0], X[:, mask == 0])


def test_per_feature_budget_is_enforced(trained, prepared):
    m, lo, hi, mask = trained; X, y = prepared["Xte"][:100], prepared["yte"][:100]
    eps = np.full(X.shape[1], 0.05, np.float32); eps[0] = 0.5
    d = np.abs(pgd(m, X, y, eps, steps=4, lo=None, hi=None) - X)
    assert d[:, 1:].max() <= 0.05 + 1e-5 and d[:, 0].max() > 0.05


def test_pgd_hurts_at_least_as_much_as_random_noise_and_fgsm_increases_loss(trained, prepared):
    m, lo, hi, mask = trained; X, y = prepared["Xte"], prepared["yte"]
    clean = bce(m, X, y); noise = bce(m, random_noise(X, .3, lo, hi, mask), y)
    f = bce(m, fgsm(m, X, y, .3, lo, hi, mask), y); p = bce(m, pgd(m, X, y, .3, steps=8, lo=lo, hi=hi, mask=mask), y)
    assert f > clean and p > clean and p >= noise


def test_adversarial_training_runs_on_train_split_only(trained, prepared):
    _, lo, hi, mask = trained
    m = nn.build_ann(prepared["Xtr"].shape[1], seed=1)
    adversarial_train(m, prepared["Xtr"], prepared["ytr"], prepared["Xva"], prepared["yva"], eps=.1, epochs=3, steps=2, lo=lo, hi=hi, mask=mask)
    assert np.isfinite(m.predict(prepared["Xte"][:10], verbose=0)).all()


def test_bounds_do_not_move_clean_points_that_are_already_out_of_range(trained, prepared):
    """Regression: clipping to training bounds once shifted out-of-range test rows beyond the eps budget."""
    m, lo, hi, mask = trained; X, y = prepared["Xte"][:100].copy(), prepared["yte"][:100]
    X[:, 0] = hi[0] + 5.0                                            # far outside the training range
    for Xa in (fgsm(m, X, y, .05, lo, hi, mask), pgd(m, X, y, .05, steps=3, lo=lo, hi=hi, mask=mask), random_noise(X, .05, lo, hi, mask)):
        assert np.abs(Xa - X).max() <= .05 + 1e-5
        assert check_constraints(X, Xa, .05, lo, hi, mask)["within_budget"]
