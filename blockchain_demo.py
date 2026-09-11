"""
blockchain_demo.py
--------------------
Blockchain-style fraud-decision audit trail.

This is a SIMULATED / LOCAL hash chain (block -> previous_hash ->
transaction_data -> timestamp -> current_hash), not a real Ethereum
deployment. It demonstrates the same tamper-evidence property a real chain
would give: each block's hash depends on the previous block's hash, so
editing any past record breaks every hash after it and integrity
verification fails. FraudAudit.sol (in this repo) is the reference smart
contract that shows what a real on-chain version of this would look like;
see README.md for how to actually deploy it to Ganache/Sepolia with Hardhat.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class Block:
    index: int
    transaction_id: str
    decision: str          # "FRAUD" or "LEGITIMATE"
    risk_score: float      # 0-100
    model_version: str
    timestamp: float
    previous_hash: str
    hash: str = field(default="")

    def compute_hash(self) -> str:
        payload = {
            "index": self.index,
            "transaction_id": self.transaction_id,
            "decision": self.decision,
            "risk_score": self.risk_score,
            "model_version": self.model_version,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
        }
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()


class LocalFraudChain:
    """A minimal append-only hash chain -- the 'blockchain' for this demo."""

    GENESIS_HASH = "0" * 64

    def __init__(self):
        self.chain: List[Block] = []
        genesis = Block(
            index=0,
            transaction_id="GENESIS",
            decision="N/A",
            risk_score=0.0,
            model_version="N/A",
            timestamp=time.time(),
            previous_hash=self.GENESIS_HASH,
        )
        genesis.hash = genesis.compute_hash()
        self.chain.append(genesis)

    def record_decision(self, transaction_id: str, decision: str,
                         risk_score: float, model_version: str) -> Block:
        prev = self.chain[-1]
        block = Block(
            index=len(self.chain),
            transaction_id=transaction_id,
            decision=decision,
            risk_score=round(float(risk_score), 4),
            model_version=model_version,
            timestamp=time.time(),
            previous_hash=prev.hash,
        )
        block.hash = block.compute_hash()
        self.chain.append(block)
        return block

    def verify_integrity(self) -> "tuple[bool, Optional[int]]":
        """Recomputes every block's hash and checks the chain links.
        Returns (is_valid, index_of_first_broken_block_or_None)."""
        for i in range(1, len(self.chain)):
            block = self.chain[i]
            prev = self.chain[i - 1]
            if block.previous_hash != prev.hash:
                return False, i
            if block.hash != block.compute_hash():
                return False, i
        return True, None

    def tamper(self, index: int, new_risk_score: float):
        """FOR DEMO ONLY: mutate a past block's data without recomputing its
        hash, to show how verify_integrity() catches the tampering."""
        if 0 < index < len(self.chain):
            self.chain[index].risk_score = new_risk_score

    def to_records(self):
        return [asdict(b) for b in self.chain]


def _demo():
    chain = LocalFraudChain()
    sample_txns = [
        ("TXN-10021", "FRAUD", 94.7, "ann-v1-adv-trained"),
        ("TXN-10022", "LEGITIMATE", 3.2, "ann-v1-adv-trained"),
        ("TXN-10023", "FRAUD", 88.1, "ann-v1-adv-trained"),
    ]
    for txn_id, decision, risk, version in sample_txns:
        block = chain.record_decision(txn_id, decision, risk, version)
        print(f"Block #{block.index}: {txn_id} -> {decision} "
              f"({risk}%) hash={block.hash[:12]}...")

    valid, bad_index = chain.verify_integrity()
    print(f"\nIntegrity check (before tampering): "
          f"{'VERIFIED' if valid else f'BROKEN at block {bad_index}'}")

    print("\n-- simulating tampering with block #1's risk score --")
    chain.tamper(1, 1.0)
    valid, bad_index = chain.verify_integrity()
    print(f"Integrity check (after tampering): "
          f"{'VERIFIED' if valid else f'BROKEN at block {bad_index}'}")


if __name__ == "__main__":
    _demo()
