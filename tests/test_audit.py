
import copy, os
import pytest
from audit_chain import LocalFraudChain, Block, verify_blocks, verify_against_anchor, payload_digest


def make(n=6):
    c = LocalFraudChain()
    for i in range(n):
        c.add_transaction(
            f"TXN-{i}", i % 2,
            "FRAUD" if i % 2 else "LEGITIMATE",
            0.1 * i, "ann-v1",
            payload_digest({"i": i}, "salt")
        )
    return c


def blocks(c):
    return [copy.deepcopy(b) for b in c.chain]


def test_fresh_chain_is_valid():
    assert make().verify_detailed()["valid"] and make().verify_integrity()


def test_modified_record_detected():
    b = blocks(make())
    b[3].decision = "LEGITIMATE" if b[3].decision == "FRAUD" else "FRAUD"
    r = verify_blocks(b)
    assert not r["valid"] and r["first_bad_index"] == 3 and "contents" in r["reason"]


def test_modified_risk_score_detected():
    b = blocks(make())
    b[2].risk_score += 0.5
    assert verify_blocks(b)["first_bad_index"] == 2


def test_recomputed_hash_still_breaks_next_link():
    from audit_chain import _hash, block_body
    b = blocks(make())
    b[2].risk_score = 0.99
    b[2].hash = _hash(block_body(b[2]))
    r = verify_blocks(b)
    assert not r["valid"] and r["first_bad_index"] == 3


def test_deleted_middle_record_detected():
    b = blocks(make())
    del b[3]
    assert not verify_blocks(b)["valid"]


def test_reordered_records_detected():
    b = blocks(make())
    b[2], b[3] = b[3], b[2]
    assert not verify_blocks(b)["valid"]


def test_invalid_hash_detected():
    b = blocks(make())
    b[4].hash = "0" * 64
    assert verify_blocks(b)["first_bad_index"] in (4, 5)


def test_broken_previous_hash_detected():
    b = blocks(make())
    b[4].previous_hash = "f" * 64
    assert not verify_blocks(b)["valid"]


def test_tampered_genesis_detected():
    b = blocks(make())
    b[0].transaction_id = "NOT-GENESIS"
    assert verify_blocks(b)["first_bad_index"] == 0


def test_inserted_record_detected():
    c = make()
    extra = make(1).chain[1]
    b = blocks(c)
    b.insert(2, extra)
    assert not verify_blocks(b)["valid"]


def test_empty_and_malformed_chain_do_not_raise():
    assert not verify_blocks([])["valid"]
    assert not verify_blocks([object()])["valid"]


def test_tail_truncation_is_only_caught_with_anchor():
    c = make()
    anchor = c.head_anchor()
    b = blocks(c)[:-1]
    assert verify_blocks(b)["valid"]
    r = verify_against_anchor(b, anchor)
    assert not r["valid"] and "truncated" in r["reason"]
    assert verify_against_anchor(blocks(c), anchor)["valid"]


def test_full_rewrite_caught_by_anchor():
    c = make()
    anchor = c.head_anchor()
    forged = make()

    forged.chain[1].transaction_id = "FORGED-TXN"

    from audit_chain import _hash, block_body

    for i in range(1, len(forged.chain)):
        forged.chain[i].previous_hash = forged.chain[i - 1].hash
        forged.chain[i].hash = _hash(block_body(forged.chain[i]))

    assert verify_blocks(forged.chain)["valid"]
    assert not verify_against_anchor(forged.chain, anchor)["valid"]


def test_save_load_roundtrip_and_legacy_artifact(tmp_path):
    c = make()
    p = tmp_path / "c.json"
    c.save(str(p))
    assert LocalFraudChain.load(str(p)).verify_integrity()

    legacy = os.path.join("artifacts", "fraud_blockchain.json")
    if os.path.exists(legacy):
        assert LocalFraudChain.load(legacy).verify_integrity()


def test_payload_digest_commits_without_exposing_data():
    d1 = payload_digest({"amount": 12.5, "card": "x"}, "s1")
    d2 = payload_digest({"amount": 12.5, "card": "x"}, "s2")
    assert d1 != d2 and len(d1) == 64 and "12.5" not in d1

    c = make(2)
    assert all("amount" not in str(vars(b)) for b in c.chain)


def test_blockchain_demo_imports_without_tensorflow_side_effects():
    import blockchain_demo
    assert blockchain_demo.LocalFraudChain is LocalFraudChain
