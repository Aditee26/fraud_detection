import numpy as np, pandas as pd, pytest
from fraudlab import synthetic
from fraudlab.data import chrono_split, FeaturePreprocessor, load_kaggle
from fraudlab.metrics import evaluate_scores, select_threshold


def test_synthetic_is_reproducible_and_labelled():
    a = synthetic.generate_transactions(500, seed=3); b = synthetic.generate_transactions(500, seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert a.is_synthetic.all() and {"account_id", "device_id", "Class"} <= set(a.columns)
    assert a.Class.sum() >= 2


def test_two_domains_have_different_schemas():
    s = synthetic.generate_transactions(300, "source"); t = synthetic.generate_transactions(300, "target")
    assert synthetic.schema("source", s)["features"] != synthetic.schema("target", t)["features"]
    assert len(synthetic.schema("source", s)["features"]) != len(synthetic.schema("target", t)["features"])


def test_chrono_split_has_no_time_overlap(tiny_df):
    tr, va, te = chrono_split(tiny_df, "Time")
    assert tr.Time.max() <= va.Time.min() and va.Time.max() <= te.Time.min()
    assert len(tr) + len(va) + len(te) == len(tiny_df)


def test_chrono_split_rejects_single_class_split():
    df = pd.DataFrame({"Time": np.arange(100.0), "Class": [0] * 99 + [1]})
    with pytest.raises(ValueError):
        chrono_split(df, "Time")


def test_preprocessor_is_fit_on_train_only(prepared, tiny_df):
    pp, tr = prepared["pp"], prepared["tr"]
    assert np.allclose(prepared["Xtr"].mean(0), 0, atol=1e-4)               # train is centred...
    assert not np.allclose(prepared["Xte"].mean(0), 0, atol=1e-3)           # ...test is not (no refit on test)
    fresh = FeaturePreprocessor(pp.feature_cols, "Amount", "Time")
    with pytest.raises(RuntimeError):
        fresh.transform(tr)                                                 # cannot transform before fit
    p2 = FeaturePreprocessor.from_dict(pp.to_dict())
    assert np.allclose(p2.transform(tr), prepared["Xtr"])


def test_raw_time_is_not_a_model_input(prepared):
    assert "Time" not in prepared["pp"].columns


def test_kaggle_loader_returns_none_when_absent(tmp_path):
    assert load_kaggle(str(tmp_path / "nope.csv")) is None


def test_kaggle_loader_rejects_wrong_schema(tmp_path):
    p = tmp_path / "x.csv"; pd.DataFrame({"a": [1]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_kaggle(str(p))


def test_metrics_known_values():
    y = np.array([0, 0, 0, 1, 1]); p = np.array([.1, .2, .9, .8, .3])
    m = evaluate_scores(y, p, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 2)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and abs(m["f1"] - 0.5) < 1e-9
    assert 0 <= m["pr_auc"] <= 1 and 0 <= m["roc_auc"] <= 1


def test_threshold_selection_uses_only_given_scores():
    rng = np.random.default_rng(0); y = (rng.random(500) < .1).astype(int); p = np.clip(y * .5 + rng.random(500) * .5, 0, 1)
    t = select_threshold(y, p)
    assert 0 <= t <= 1
    t2 = select_threshold(y, p, "recall_at_precision", 0.6)
    assert 0 <= t2 <= 1
