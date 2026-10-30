import json, os, pytest

R = os.path.join("artifacts", "research")


def _need(p):
    if not os.path.exists(p): pytest.skip(f"{p} not generated yet (run the experiment)")
    return p


def test_saved_results_are_labelled_synthetic():
    for p in ["transfer/transfer_config.json", "adversarial/robustness_config.json", "multimodel/multimodel_config.json"]:
        d = json.load(open(_need(os.path.join(R, p)), encoding="utf-8"))
        assert "SYNTHETIC" in json.dumps(d).upper() or "USER_CSV" in json.dumps(d) or "KAGGLE" in json.dumps(d)


def test_adversarial_constraints_all_satisfied():
    d = json.load(open(_need(os.path.join(R, "adversarial", "robustness_config.json"))))
    assert d["all_constraints_satisfied"] is True and "physically" in d["note"]


def test_multimodel_has_no_fabricated_graph_claim():
    d = json.load(open(_need(os.path.join(R, "multimodel", "multimodel_config.json"))))
    assert "NOT available in Kaggle" in d["graph"] and d["causal_checks"]["graph_edges"] and d["causal_checks"]["lstm_windows"]


def test_research_dashboard_renders_without_dataset_or_artifacts(tmp_path, monkeypatch):
    st_testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.chdir(tmp_path)                                   # empty dir: no artifacts, no dataset
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research_dashboard.py")
    script = (f"import sys; sys.path.insert(0, r'{os.path.dirname(src)}')\n"
              "import streamlit as st, research_dashboard as rd\n"
              "t=st.tabs(['a','b','c','d'])\n"
              "with t[0]: rd.render_transfer_tab()\nwith t[1]: rd.render_pgd_tab()\n"
              "with t[2]: rd.render_drift_tab()\nwith t[3]: rd.render_results_tab()\n")
    at = st_testing.AppTest.from_string(script, default_timeout=60).run()
    assert not at.exception
    assert any("not found" in w.value.lower() or "not generated" in w.value.lower() for w in list(at.warning) + list(at.info))
