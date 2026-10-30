"""Concept-drift detectors (self-contained) + prequential stream monitor.

ADWIN: adaptive window; drift is flagged when two sub-windows of the recent
history have means that differ by more than a Hoeffding-style bound
(Bifet & Gavalda, 2007). This is a straightforward O(window) implementation
that checks every cut point of a bounded window - simpler than the
exponential-histogram version in the `river` library, which is used as an
independent cross-check in the tests when installed.

DDM (Gama et al., 2004): tracks the online error rate p and its std s; warns
at p+s > p_min+2*s_min and signals drift at p+s > p_min+3*s_min.
"""
from __future__ import annotations
import json, math, os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import numpy as np


class ADWIN:
    def __init__(self, delta: float = 0.002, max_window: int = 2000, min_sub: int = 30, check_every: int = 8):
        self.delta, self.max_window, self.min_sub, self.check_every = delta, max_window, min_sub, check_every
        self.window: deque = deque()
        self._n = 0
        self.drift_detected = False

    def reset(self):
        self.window.clear(); self.drift_detected = False

    def update(self, x: float) -> bool:
        self.drift_detected = False
        self.window.append(float(x)); self._n += 1
        if len(self.window) > self.max_window:
            self.window.popleft()
        if self._n % self.check_every or len(self.window) < 2 * self.min_sub:
            return False
        w = np.fromiter(self.window, float)
        n = len(w); cs = np.concatenate([[0.0], np.cumsum(w)])
        n0 = np.arange(self.min_sub, n - self.min_sub + 1)
        if len(n0) == 0:
            return False
        n1 = n - n0
        m0 = cs[n0] / n0; m1 = (cs[n] - cs[n0]) / n1
        m = 1.0 / (1.0 / n0 + 1.0 / n1)            # harmonic mean of sub-window sizes
        dd = math.log(2.0 * math.log(max(n, 2)) / self.delta)
        var = w.var()
        eps = np.sqrt(2.0 / m * var * dd) + 2.0 / (3.0 * m) * dd
        hit = np.abs(m0 - m1) > eps
        if hit.any():
            cut = int(n0[np.argmax(hit)])
            for _ in range(cut):               # drop the stale (older) part
                self.window.popleft()
            self.drift_detected = True
        return self.drift_detected

    @property
    def mean(self):
        return float(np.mean(self.window)) if self.window else 0.0


class DDM:
    def __init__(self, min_instances: int = 30, warn: float = 2.0, drift: float = 3.0):
        self.min_instances, self.warn, self.drift = min_instances, warn, drift
        self.reset()

    def reset(self):
        self.n = 0; self.p = 1.0; self.s = 0.0
        self.p_min = float("inf"); self.s_min = float("inf"); self.ps_min = float("inf")
        self.warning = False; self.drift_detected = False

    def update(self, error: float) -> bool:
        """error: 1 if the prediction was wrong, else 0."""
        self.n += 1
        self.p += (float(error) - self.p) / self.n
        self.s = math.sqrt(self.p * (1 - self.p) / self.n)
        self.warning = self.drift_detected = False
        if self.n < self.min_instances:
            return False
        if self.p + self.s <= self.ps_min:
            self.p_min, self.s_min, self.ps_min = self.p, self.s, self.p + self.s
        if self.p + self.s > self.p_min + self.drift * self.s_min:
            self.reset()
            self.drift_detected = True      # set AFTER reset(), which clears the flag
        elif self.p + self.s > self.p_min + self.warn * self.s_min:
            self.warning = True
        return self.drift_detected


@dataclass
class DriftMonitor:
    """Prequential monitor: predict first, observe the label later.

    `observe(i, y_true, y_pred)` must only be called after the prediction for
    transaction i was already produced, so detectors never influence or see
    labels ahead of time. Events are appended to a JSONL log.
    """
    log_path: str | None = None
    adwin: ADWIN = field(default_factory=ADWIN)
    ddm: DDM = field(default_factory=DDM)
    events: list = field(default_factory=list)

    def observe(self, i: int, y_true: int, y_pred: int, label_time: float | None = None):
        err = float(int(y_true) != int(y_pred))
        fired = []
        if self.adwin.update(err): fired.append("ADWIN")
        if self.ddm.update(err): fired.append("DDM")
        for det in fired:
            ev = {"detector": det, "stream_index": int(i), "label_time": label_time,
                  "logged_at": datetime.now(timezone.utc).isoformat()}
            self.events.append(ev)
            if self.log_path:
                os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(ev) + "\n")
        return fired
