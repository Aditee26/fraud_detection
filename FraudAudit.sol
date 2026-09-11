// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title FraudAudit
/// @notice Tamper-evident on-chain log of fraud-detection decisions.
/// @dev    This is a REFERENCE contract for the prototype. It is NOT deployed
///         to a live Ethereum network in this demo -- blockchain_demo.py
///         simulates the same append-only, hash-linked guarantee locally
///         with a lightweight Python hash chain so the concept can be shown
///         without needing a funded testnet wallet during a live viva.
///         Deploying this file for real only requires Hardhat + Ganache/
///         Sepolia (see README.md, "Going from simulated to real chain").
contract FraudAudit {

    struct FraudRecord {
        string  transactionId;   // e.g. "TXN-10021"
        bool    isFraud;         // true = FRAUD, false = LEGITIMATE
        uint16  riskScoreBps;    // risk score in basis points (0-10000 = 0-100.00%)
        uint256 timestamp;       // block timestamp at recording time
        string  modelVersion;    // e.g. "ann-v1-adv-trained"
    }

    address public owner;
    FraudRecord[] public records;

    event DecisionRecorded(
        uint256 indexed blockIndex,
        string transactionId,
        bool isFraud,
        uint16 riskScoreBps,
        string modelVersion
    );

    constructor() {
        owner = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "FraudAudit: caller is not the owner");
        _;
    }

    /// @notice Record a fraud/legitimate decision immutably on-chain.
    function recordDecision(
        string calldata transactionId,
        bool isFraud,
        uint16 riskScoreBps,
        string calldata modelVersion
    ) external onlyOwner {
        records.push(FraudRecord({
            transactionId: transactionId,
            isFraud: isFraud,
            riskScoreBps: riskScoreBps,
            timestamp: block.timestamp,
            modelVersion: modelVersion
        }));

        emit DecisionRecorded(
            records.length - 1,
            transactionId,
            isFraud,
            riskScoreBps,
            modelVersion
        );
    }

    function getRecord(uint256 index) external view returns (FraudRecord memory) {
        return records[index];
    }

    function totalRecords() external view returns (uint256) {
        return records.length;
    }
}
