// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title FraudAudit
/// @notice Append-only, hash-chained log of fraud-detection decisions.
/// @dev PRIVACY: only commitments go on-chain. `recordDigest` is a salted SHA-256 of the
///      off-chain decision record (see audit_chain.payload_digest); raw transaction data,
///      account identifiers and labels never touch the chain.
///      Each entry's `chainHash` commits to the previous head, so any change to history
///      changes every later hash. `headHash` is the external anchor that makes truncation
///      and full rewrites of the OFF-chain log detectable (the Python local chain cannot
///      detect those on its own).
///      STATUS: tested on a local in-process EVM only (see contracts/test). It has NOT been
///      deployed to any public testnet or mainnet, and has had no professional audit.
contract FraudAudit {
    struct Record {
        bytes32 recordDigest;      // salted SHA-256 commitment to the off-chain record
        bytes32 modelVersionHash;  // keccak256/sha256 of the model identifier (not the string)
        bytes32 chainHash;         // keccak256(previous head, this entry)
        uint64  timestamp;
        uint16  riskScoreBps;      // 0..10000 = 0.00%..100.00%
        bool    isFraud;
    }

    uint16 public constant MAX_BPS = 10_000;

    address public owner;
    address public pendingOwner;
    bytes32 public headHash;       // GENESIS = 0x00..00
    Record[] private _records;

    event DecisionRecorded(uint256 indexed index, bytes32 indexed recordDigest, bool isFraud,
                           uint16 riskScoreBps, bytes32 modelVersionHash, bytes32 chainHash);
    event OwnershipTransferStarted(address indexed from, address indexed to);
    event OwnershipTransferred(address indexed from, address indexed to);

    error NotOwner();
    error NotPendingOwner();
    error InvalidRiskScore(uint16 given);
    error EmptyDigest();
    error IndexOutOfRange(uint256 index, uint256 length);
    error ZeroAddress();

    modifier onlyOwner() {
        if (msg.sender != owner) revert NotOwner();
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function recordDecision(bytes32 recordDigest, bool isFraud, uint16 riskScoreBps, bytes32 modelVersionHash)
        external onlyOwner returns (uint256 index)
    {
        if (recordDigest == bytes32(0)) revert EmptyDigest();
        if (riskScoreBps > MAX_BPS) revert InvalidRiskScore(riskScoreBps);
        index = _records.length;
        bytes32 newHead = keccak256(abi.encode(headHash, index, recordDigest, isFraud, riskScoreBps,
                                               modelVersionHash, uint64(block.timestamp)));
        _records.push(Record(recordDigest, modelVersionHash, newHead, uint64(block.timestamp), riskScoreBps, isFraud));
        headHash = newHead;
        emit DecisionRecorded(index, recordDigest, isFraud, riskScoreBps, modelVersionHash, newHead);
    }

    function totalRecords() external view returns (uint256) {
        return _records.length;
    }

    function getRecord(uint256 index) external view returns (Record memory) {
        if (index >= _records.length) revert IndexOutOfRange(index, _records.length);
        return _records[index];
    }

    /// @notice True iff record `index` commits to `recordDigest` (lets an auditor check an off-chain record).
    function verifyRecord(uint256 index, bytes32 recordDigest) external view returns (bool) {
        if (index >= _records.length) revert IndexOutOfRange(index, _records.length);
        return _records[index].recordDigest == recordDigest;
    }

    /// @notice Recompute the whole hash chain from stored entries; true iff it ends at `headHash`.
    function verifyChain() external view returns (bool) {
        bytes32 h = bytes32(0);
        for (uint256 i = 0; i < _records.length; i++) {
            Record storage r = _records[i];
            h = keccak256(abi.encode(h, i, r.recordDigest, r.isFraud, r.riskScoreBps, r.modelVersionHash, r.timestamp));
            if (h != r.chainHash) return false;
        }
        return h == headHash;
    }

    // Two-step ownership transfer prevents handing the audit log to a wrong address.
    function transferOwnership(address to) external onlyOwner {
        if (to == address(0)) revert ZeroAddress();
        pendingOwner = to;
        emit OwnershipTransferStarted(owner, to);
    }

    function acceptOwnership() external {
        if (msg.sender != pendingOwner) revert NotPendingOwner();
        emit OwnershipTransferred(owner, msg.sender);
        owner = msg.sender;
        pendingOwner = address(0);
    }
}
