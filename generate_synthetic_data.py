"""Write reproducible SYNTHETIC datasets to data/ (never real bank data; all rows carry is_synthetic=True).
  python generate_synthetic_data.py                 -> data/synthetic_target.csv, data/synthetic_source.csv
The 'target' file mimics the Kaggle column layout (Time, V1..V28, Amount, Class) plus account_id/device_id."""
import argparse, os
from fraudlab import synthetic

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n-target", type=int, default=20000); ap.add_argument("--n-source", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="data"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    synthetic.generate_transactions(a.n_target, "target", 0.015, seed=a.seed + 2, latent_seed=a.seed).to_csv(os.path.join(a.out, "synthetic_target.csv"), index=False)
    synthetic.generate_transactions(a.n_source, "source", 0.03, seed=a.seed + 1, latent_seed=a.seed).to_csv(os.path.join(a.out, "synthetic_source.csv"), index=False)
    print("wrote", a.out)
