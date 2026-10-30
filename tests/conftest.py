import os, sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from fraudlab import synthetic
from fraudlab.data import chrono_split, FeaturePreprocessor


@pytest.fixture(scope="session")
def tiny_df():
    return synthetic.generate_transactions(3000, "target", fraud_rate=0.04, seed=1, n_accounts=120)


@pytest.fixture(scope="session")
def prepared(tiny_df):
    sch = synthetic.schema("target", tiny_df)
    tr, va, te = chrono_split(tiny_df, "Time")
    pp = FeaturePreprocessor(sch["features"], "Amount", "Time").fit(tr)
    return dict(tr=tr, va=va, te=te, pp=pp, Xtr=pp.transform(tr), Xva=pp.transform(va), Xte=pp.transform(te),
                ytr=tr.Class.values, yva=va.Class.values, yte=te.Class.values)
