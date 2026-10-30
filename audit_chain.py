"""Local tamper-evident hash chain (no TensorFlow / blockchain credentials needed).

Extracted from blockchain_demo.py (which re-exports these names) so the audit
logic can be tested independently. Backward compatible: blocks written by the
old code verify unchanged. The optional `payload_digest` field commits to
off-chain data (SHA-256 of a salted canonical record) without storing it.

What the chain CAN detect: edited fields, deleted or inserted blocks in the
middle, reordering, broken links, bad indices/genesis.
What it CANNOT detect alone: truncation of the newest blocks, or a full
rewrite that recomputes every hash. Both need an external anchor of the head
hash (see `head_anchor()` / `verify_against_anchor()`), which on a real
deployment is what the smart contract / ledger provides.
"""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

BASE_FIELDS = ("index", "transaction_id", "actual_class", "decision", "risk_score",
               "model_version", "timestamp", "previous_hash")


@dataclass
class Block:
    index: int
    transaction_id: str
    actual_class: int
    decision: str
    risk_score: float
    model_version: str
    timestamp: str
    previous_hash: str
    hash: str
    payload_digest: Optional[str] = None


def _hash(data: dict) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def block_body(b) -> dict:
    d = {k: getattr(b, k) for k in BASE_FIELDS}
    if getattr(b, "payload_digest", None) is not None:
        d["payload_digest"] = b.payload_digest
    return d


def payload_digest(record: dict, salt: str) -> str:
    """Commitment to an off-chain record. Raw data never enters the chain."""
    return hashlib.sha256((salt + json.dumps(record, sort_keys=True, default=str)).encode()).hexdigest()


class LocalFraudChain:
    def __init__(self, blocks=None):
        self.chain: list[Block] = []
        if blocks is None:
            self._create_genesis_block()
        else:
            self.chain = [b if isinstance(b, Block) else Block(**b) for b in blocks]

    def _calculate_hash(self, block_data):
        return _hash(block_data)

    def _create_genesis_block(self):
        body = {"index": 0, "transaction_id": "GENESIS", "actual_class": -1, "decision": "GENESIS",
                "risk_score": 0.0, "model_version": "system",
                "timestamp": datetime.now(timezone.utc).isoformat(), "previous_hash": "0"}
        self.chain.append(Block(**body, hash=_hash(body)))

    def add_transaction(self, transaction_id, actual_class, decision, risk_score, model_version, payload_digest=None):
        prev = self.chain[-1]
        body = {"index": len(self.chain), "transaction_id": transaction_id, "actual_class": int(actual_class),
                "decision": decision, "risk_score": round(float(risk_score), 4), "model_version": model_version,
                "timestamp": datetime.now(timezone.utc).isoformat(), "previous_hash": prev.hash}
        if payload_digest is not None:
            body["payload_digest"] = payload_digest
        block = Block(**body, hash=_hash(body))
        self.chain.append(block)
        return block

    def verify_detailed(self) -> dict:
        return verify_blocks(self.chain)

    def verify_integrity(self) -> bool:  # legacy API
        return self.verify_detailed()["valid"]

    def head_anchor(self) -> dict:
        return {"length": len(self.chain), "head_hash": self.chain[-1].hash}

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(b) for b in self.chain], f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))


def verify_blocks(blocks) -> dict:
    """Return {valid, first_bad_index, reason}. Never raises on malformed data."""
    try:
        if not blocks:
            return {"valid": False, "first_bad_index": None, "reason": "empty chain"}
        g = blocks[0]
        if g.index != 0 or g.previous_hash != "0" or g.transaction_id != "GENESIS":
            return {"valid": False, "first_bad_index": 0, "reason": "invalid genesis block"}
        if g.hash != _hash(block_body(g)):
            return {"valid": False, "first_bad_index": 0, "reason": "genesis hash mismatch"}
        for i in range(1, len(blocks)):
            cur, prev = blocks[i], blocks[i - 1]
            if cur.index != i:
                return {"valid": False, "first_bad_index": i, "reason": "index out of sequence (deleted/inserted/reordered)"}
            if cur.previous_hash != prev.hash:
                return {"valid": False, "first_bad_index": i, "reason": "previous_hash does not match prior block"}
            if cur.hash != _hash(block_body(cur)):
                return {"valid": False, "first_bad_index": i, "reason": "stored hash does not match block contents"}
        return {"valid": True, "first_bad_index": None, "reason": "ok"}
    except Exception as e:  # malformed block objects
        return {"valid": False, "first_bad_index": None, "reason": f"malformed chain: {type(e).__name__}"}


def verify_against_anchor(blocks, anchor: dict) -> dict:
    """Detect tail truncation / full rewrite using an externally stored head hash."""
    base = verify_blocks(blocks)
    if not base["valid"]:
        return base
    if len(blocks) != anchor["length"]:
        return {"valid": False, "first_bad_index": min(len(blocks), anchor["length"]),
                "reason": f"length {len(blocks)} != anchored {anchor['length']} (truncated or extended)"}
    if blocks[-1].hash != anchor["head_hash"]:
        return {"valid": False, "first_bad_index": len(blocks) - 1, "reason": "head hash differs from anchor (chain rewritten)"}
    return {"valid": True, "first_bad_index": None, "reason": "ok (matches anchor)"}
