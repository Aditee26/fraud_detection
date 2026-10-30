const { expect } = require("chai");
const { ethers } = require("hardhat");

const D = (s) => ethers.sha256(ethers.toUtf8Bytes(s));       // stand-in for audit_chain.payload_digest
const MODEL = ethers.keccak256(ethers.toUtf8Bytes("ann-v1"));

describe("FraudAudit", () => {
  let c, owner, other;
  beforeEach(async () => {
    [owner, other] = await ethers.getSigners();
    c = await (await ethers.getContractFactory("FraudAudit")).deploy();
  });

  it("starts empty with a zero head and the deployer as owner", async () => {
    expect(await c.totalRecords()).to.equal(0n);
    expect(await c.headHash()).to.equal(ethers.ZeroHash);
    expect(await c.owner()).to.equal(owner.address);
    expect(await c.verifyChain()).to.equal(true);
  });

  it("records a decision, emits an event and advances the head", async () => {
    await expect(c.recordDecision(D("t1"), true, 9300, MODEL)).to.emit(c, "DecisionRecorded");
    expect(await c.totalRecords()).to.equal(1n);
    const r = await c.getRecord(0);
    expect(r.isFraud).to.equal(true);
    expect(r.riskScoreBps).to.equal(9300n);
    expect(await c.headHash()).to.equal(r.chainHash);
  });

  it("chain hash matches an independent off-chain recomputation", async () => {
    const abi = ethers.AbiCoder.defaultAbiCoder();
    let head = ethers.ZeroHash;
    for (let i = 0; i < 3; i++) {
      await c.recordDecision(D("t" + i), i % 2 === 0, 100 * i, MODEL);
      const r = await c.getRecord(i);
      head = ethers.keccak256(abi.encode(
        ["bytes32", "uint256", "bytes32", "bool", "uint16", "bytes32", "uint64"],
        [head, i, r.recordDigest, r.isFraud, r.riskScoreBps, r.modelVersionHash, r.timestamp]));
      expect(r.chainHash).to.equal(head);
    }
    expect(await c.headHash()).to.equal(head);
    expect(await c.verifyChain()).to.equal(true);
  });

  it("rejects non-owners", async () => {
    await expect(c.connect(other).recordDecision(D("x"), false, 10, MODEL)).to.be.revertedWithCustomError(c, "NotOwner");
  });

  it("rejects risk scores above 10000 bps and empty digests", async () => {
    await expect(c.recordDecision(D("x"), false, 10001, MODEL)).to.be.revertedWithCustomError(c, "InvalidRiskScore");
    await expect(c.recordDecision(ethers.ZeroHash, false, 10, MODEL)).to.be.revertedWithCustomError(c, "EmptyDigest");
    await c.recordDecision(D("edge"), true, 10000, MODEL);                       // boundary accepted
  });

  it("reverts on out-of-range reads", async () => {
    await expect(c.getRecord(0)).to.be.revertedWithCustomError(c, "IndexOutOfRange");
    await expect(c.verifyRecord(5, D("x"))).to.be.revertedWithCustomError(c, "IndexOutOfRange");
  });

  it("verifyRecord detects a wrong off-chain record", async () => {
    await c.recordDecision(D("real"), true, 5000, MODEL);
    expect(await c.verifyRecord(0, D("real"))).to.equal(true);
    expect(await c.verifyRecord(0, D("tampered"))).to.equal(false);
  });

  it("has no function to edit or delete a stored record", async () => {
    const names = c.interface.fragments.filter((f) => f.type === "function").map((f) => f.name);
    for (const bad of ["updateRecord", "deleteRecord", "editRecord", "setRecord"]) expect(names).to.not.include(bad);
  });

  it("uses a two-step ownership transfer", async () => {
    await expect(c.connect(other).acceptOwnership()).to.be.revertedWithCustomError(c, "NotPendingOwner");
    await c.transferOwnership(other.address);
    expect(await c.owner()).to.equal(owner.address);          // unchanged until accepted
    await c.connect(other).acceptOwnership();
    expect(await c.owner()).to.equal(other.address);
    await expect(c.recordDecision(D("x"), false, 1, MODEL)).to.be.revertedWithCustomError(c, "NotOwner");
    await expect(c.connect(other).transferOwnership(ethers.ZeroAddress)).to.be.revertedWithCustomError(c, "ZeroAddress");
  });
});
