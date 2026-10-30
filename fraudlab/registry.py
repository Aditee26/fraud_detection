"""File-based model registry: candidate retraining never touches the working model.

  registry.json   {"current": "v001", "history": [...], "versions": {...}}
  versions/<id>/model.joblib
  candidates/<id>/model.joblib + report.json

Flow: train_candidate() -> evaluate_candidate() -> promote(approved=True).
promote() only repoints `current`; old versions stay on disk so rollback() always works.
"""
from __future__ import annotations
import json, os, shutil
from datetime import datetime, timezone
import joblib
from .metrics import evaluate_scores


class ModelRegistry:
    def __init__(self, root: str):
        self.root = root; self.meta_path = os.path.join(root, "registry.json")
        os.makedirs(os.path.join(root, "versions"), exist_ok=True); os.makedirs(os.path.join(root, "candidates"), exist_ok=True)
        if not os.path.exists(self.meta_path): self._write({"current": None, "history": [], "versions": {}})

    def _read(self):
        with open(self.meta_path, encoding="utf-8") as f: return json.load(f)

    def _write(self, m):
        with open(self.meta_path, "w", encoding="utf-8") as f: json.dump(m, f, indent=2)

    @property
    def current(self): return self._read()["current"]

    def load_current(self):
        c = self.current
        return None if c is None else joblib.load(os.path.join(self.root, "versions", c, "model.joblib"))

    def register_initial(self, model, note="initial model"):
        m = self._read(); vid = f"v{len(m['versions']) + 1:03d}"
        os.makedirs(os.path.join(self.root, "versions", vid)); joblib.dump(model, os.path.join(self.root, "versions", vid, "model.joblib"))
        m["versions"][vid] = {"created": datetime.now(timezone.utc).isoformat(), "note": note}
        m["current"] = vid; m["history"].append({"action": "register", "version": vid}); self._write(m); return vid

    def add_candidate(self, model, note: str, report: dict | None = None):
        cid = "cand_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        d = os.path.join(self.root, "candidates", cid); os.makedirs(d)
        joblib.dump(model, os.path.join(d, "model.joblib"))
        with open(os.path.join(d, "report.json"), "w", encoding="utf-8") as f: json.dump({"note": note, **(report or {})}, f, indent=2)
        return cid

    def evaluate_candidate(self, cid, X_hold, y_hold, threshold=0.5, min_pr_auc_gain=0.02):
        """Compare current vs candidate on a labelled, chronologically LATER hold-out."""
        cur = self.load_current(); cand = joblib.load(os.path.join(self.root, "candidates", cid, "model.joblib"))
        rc = evaluate_scores(y_hold, cur.predict_proba(X_hold)[:, 1], threshold)
        rn = evaluate_scores(y_hold, cand.predict_proba(X_hold)[:, 1], threshold)
        rep = {"current": rc, "candidate": rn, "pr_auc_gain": rn["pr_auc"] - rc["pr_auc"], "min_gain_required": min_pr_auc_gain,
               "recommend_promote": bool(rn["pr_auc"] - rc["pr_auc"] >= min_pr_auc_gain)}
        p = os.path.join(self.root, "candidates", cid, "report.json")
        base = json.load(open(p, encoding="utf-8")); base["evaluation"] = rep
        json.dump(base, open(p, "w", encoding="utf-8"), indent=2); return rep

    def promote(self, cid, approved: bool = False):
        """Requires explicit approval AND a passing evaluation. Never overwrites an existing version."""
        if not approved: raise PermissionError("promotion requires approved=True (human decision)")
        rep = json.load(open(os.path.join(self.root, "candidates", cid, "report.json"), encoding="utf-8")).get("evaluation")
        if not rep or not rep["recommend_promote"]: raise ValueError("candidate has not passed evaluation")
        m = self._read(); vid = f"v{len(m['versions']) + 1:03d}"
        shutil.copytree(os.path.join(self.root, "candidates", cid), os.path.join(self.root, "versions", vid))
        m["versions"][vid] = {"created": datetime.now(timezone.utc).isoformat(), "note": f"promoted from {cid}"}
        m["history"].append({"action": "promote", "from": m["current"], "to": vid}); m["current"] = vid; self._write(m); return vid

    def rollback(self):
        m = self._read(); prev = [h for h in m["history"] if h["action"] == "promote"]
        if not prev: raise ValueError("nothing to roll back")
        m["current"] = prev[-1]["from"]; m["history"].append({"action": "rollback", "to": m["current"]}); self._write(m); return m["current"]
