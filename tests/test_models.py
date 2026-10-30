import numpy as np, pytest
tf = pytest.importorskip("tensorflow")
from fraudlab import nn, synthetic
from fraudlab.sequences import build_sequences, check_causal
from fraudlab.graph import build_neighbors, check_edges_causal, sage_inputs, build_graphsage
from fraudlab.fusion import expanding_window_oof, LateFusion
from fraudlab.data import chrono_split, FeaturePreprocessor


def test_ann_trains_and_beats_chance(prepared):
    m = nn.build_ann(prepared["Xtr"].shape[1], seed=0)
    nn.fit_model(m, prepared["Xtr"], prepared["ytr"], prepared["Xva"], prepared["yva"], epochs=6)
    p = nn.predict(m, prepared["Xte"])
    from sklearn.metrics import average_precision_score
    assert average_precision_score(prepared["yte"], p) > prepared["yte"].mean() * 2


def test_sequences_are_causal_and_label_free(tiny_df, prepared):
    X = prepared["pp"].transform(tiny_df)
    S, WT = build_sequences(X, tiny_df.Time.values, tiny_df.account_id.values, 6)
    assert S.shape == (len(tiny_df), 6, X.shape[1] + 1) and check_causal(tiny_df.Time.values, WT)
    i = 100; acct = tiny_df.account_id.values[i]; t = tiny_df.Time.values
    prior = np.where((tiny_df.account_id.values == acct) & (t <= t[i]))[0]
    assert np.allclose(S[i, -1, :-1], X[i])                          # last step is the row itself
    assert (~np.isnan(WT[i])).sum() == min(6, len(prior))            # window holds only same-account history


def test_sequence_causality_check_catches_future_rows():
    t = np.array([1., 2., 3.]); WT = np.array([[1, np.nan], [2, 3.], [3, np.nan]])   # row 1 sees t=3 > 2
    assert not check_causal(t, WT)


def test_graph_edges_point_only_to_the_past(tiny_df):
    idx = build_neighbors(tiny_df.Time.values, [tiny_df.account_id.values, tiny_df.device_id.values], 6)
    assert check_edges_causal(tiny_df.Time.values, idx) and (idx >= 0).any()
    ok = idx >= 0; assert (idx[ok] < np.repeat(np.arange(len(idx))[:, None], idx.shape[1], 1)[ok]).all()


def test_graph_refuses_to_fabricate_structure_without_entity_ids():
    with pytest.raises(ValueError, match="cannot be built"):
        build_neighbors(np.arange(10.0), [])


def test_graphsage_forward_and_shapes(tiny_df, prepared):
    X = prepared["pp"].transform(tiny_df)
    idx = build_neighbors(tiny_df.Time.values, [tiny_df.account_id.values, tiny_df.device_id.values], 5)
    G = sage_inputs(X, idx); m = build_graphsage(X.shape[1], 5)
    out = m.predict(G, verbose=0); assert out.shape == (len(X), 1) and np.isfinite(out).all()
    assert not any("label" in k for k in G)                          # no label channel enters aggregation


def test_fusion_oof_is_out_of_sample_and_time_ordered():
    seen = []
    def fp(tr, te):
        assert tr.max() < te.min(), "train block must precede predicted block"
        seen.append((tr.max(), te.min(), te.max())); return {"a": np.full(len(te), .3), "b": np.full(len(te), .6)}
    y = np.tile([0, 0, 0, 1], 100)
    oof, names = expanding_window_oof(400, 3, fp, y)
    assert np.isnan(oof[:80]).all() and not np.isnan(oof[100:]).any()     # first block never predicted
    assert len(seen) == 3


def test_late_fusion_learns_the_better_model():
    r = np.random.default_rng(0); y = (r.random(2000) < .1).astype(int)
    good = np.clip(y * .6 + r.random(2000) * .4, .01, .99); bad = r.random(2000)
    f = LateFusion().fit(np.column_stack([good, bad]), y)
    assert f.weights[0] > abs(f.weights[1])


def test_transfer_alignment_only_shares_the_encoder():
    src_m, src_enc = nn.build_domain_net(21, adapter_dim=16, seed=0)      # 18 domain feats + 3 canonical
    tgt_m, tgt_enc = nn.build_domain_net(31, adapter_dim=16, seed=1)      # different input schema
    enc = nn.build_shared_encoder(16 + nn.CANONICAL_TAIL); enc.set_weights(src_enc.get_weights())
    m, e = nn.build_domain_net(31, encoder=enc, adapter_dim=16, seed=1)
    assert all(np.allclose(a, b) for a, b in zip(e.get_weights(), src_enc.get_weights()))   # encoder carried over
    adapter = [l for l in m.layers if l.name == "adapter"][0]; assert adapter.get_weights()[0].shape[0] == 31 - nn.CANONICAL_TAIL


def test_transfer_experiment_runs_end_to_end_tiny(tmp_path, monkeypatch):
    import transfer_learning as T
    monkeypatch.setattr(T, "OUT", str(tmp_path)); monkeypatch.setattr(T, "FRACTIONS", (0.5, 1.0))
    args = T.argparse.Namespace(seeds=1, epochs=2, n_source=2500, n_target=2500, adapter_dim=8, seed=0, source_csv=None, target_csv=None, quick=True)
    res, summ = T.run(args)
    assert set(res.arm) == {"scratch", "transfer", "xgboost"} and (res.data_origin == "SYNTHETIC").all()
    assert {"precision", "recall", "f1", "pr_auc", "roc_auc", "train_seconds"} <= set(res.columns)
    assert (tmp_path / "transfer_results_summary.csv").exists() and (tmp_path / "transfer_curves.png").exists()
    a = res[(res.arm == "scratch")].pr_auc.values; b = res[(res.arm == "transfer")].pr_auc.values
    assert not np.allclose(a, b), "transfer arm must not silently reuse the scratch predictions"
