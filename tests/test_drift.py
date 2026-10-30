import numpy as np, pytest, joblib
from fraudlab.drift import ADWIN, DDM, DriftMonitor
from fraudlab.registry import ModelRegistry


def stream(seed, n1=1500, n2=1500, p1=.05, p2=.30):
    r = np.random.default_rng(seed); return np.concatenate([r.random(n1) < p1, r.random(n2) < p2]).astype(float)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_adwin_detects_shift_soon_after_change(seed):
    a = ADWIN(); hits = [i for i, x in enumerate(stream(seed)) if a.update(x)]
    assert hits and 1500 <= hits[0] < 1900


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_adwin_quiet_on_stationary_stream(seed):
    a = ADWIN(); r = np.random.default_rng(seed)
    assert sum(a.update(float(x)) for x in (r.random(4000) < .05)) == 0


def test_ddm_signals_drift_after_change():
    d = DDM(); r = np.random.default_rng(5)
    s = np.concatenate([r.random(2000) < .05, r.random(1500) < .35]).astype(float)
    hits = [i for i, x in enumerate(s) if d.update(x)]
    assert any(h >= 2000 for h in hits)


def test_ddm_known_limitation_documented_not_hidden():
    # DDM can false-alarm early on stationary data; we only assert the flag mechanics work.
    d = DDM(); [d.update(0.0) for _ in range(40)]; assert d.drift_detected is False
    d.update(1.0); assert isinstance(d.warning, bool)


def test_adwin_matches_river_on_clear_shift():
    river = pytest.importorskip("river.drift")
    s = stream(7); mine = ADWIN(); ref = river.ADWIN(delta=0.002)
    a = next(i for i, x in enumerate(s) if mine.update(x)); b = next(i for i, x in enumerate(s) if (ref.update(x), ref.drift_detected)[1])
    assert abs(a - b) < 400 and a >= 1500 and b >= 1500


def test_monitor_logs_events_and_uses_only_labels_passed_in(tmp_path):
    log = tmp_path / "ev.jsonl"; m = DriftMonitor(log_path=str(log)); r = np.random.default_rng(1); fired = False
    for i in range(3000):
        err = r.random() < (.05 if i < 1500 else .4)
        fired |= bool(m.observe(i, int(err), 0))               # y_true=err, y_pred=0 -> error iff err
    assert fired and log.exists() and log.read_text().strip()
    assert all(e["stream_index"] >= 0 for e in m.events)


def test_registry_never_overwrites_and_requires_approval(tmp_path):
    from sklearn.linear_model import LogisticRegression
    r = np.random.default_rng(0); X = r.normal(size=(600, 4)); y = (X[:, 0] + .3 * r.normal(size=600) > 1).astype(int)
    reg = ModelRegistry(str(tmp_path)); cur = LogisticRegression().fit(X[:100], y[:100]); v1 = reg.register_initial(cur)
    cid = reg.add_candidate(LogisticRegression().fit(X[:400], y[:400]), "retrain")
    with pytest.raises(PermissionError): reg.promote(cid)                         # no approval
    with pytest.raises(ValueError): reg.promote(cid, approved=True)               # not evaluated yet
    assert reg.current == v1
    rep = reg.evaluate_candidate(cid, X[400:], y[400:], min_pr_auc_gain=-1.0)     # force a pass to test the mechanics
    v2 = reg.promote(cid, approved=True)
    assert reg.current == v2 != v1
    assert (tmp_path / "versions" / v1 / "model.joblib").exists()                 # old model preserved
    assert reg.rollback() == v1 and reg.current == v1
