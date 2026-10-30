# Security notes

- **Secrets:** none are required. Local hash-chain mode needs no credentials. `.env`, `.streamlit/secrets.toml`
  and `*.zip` are git-ignored. Never commit private keys or RPC URLs; if you later deploy the contract, load them
  from environment variables.
- **Data privacy:** the Kaggle dataset stays outside Git (`data/` is ignored). `audit_chain.payload_digest` and
  `contracts/FraudAudit.sol` store only a salted SHA-256 commitment, decision, risk score and model id - never raw
  transaction fields. The legacy local chain still records `actual_class`; drop it in production. Keep the salt secret.
- **Blockchain status:** the contract was compiled and tested ONLY on Hardhat's in-process EVM (9 tests).
  It has not been deployed to any public testnet/mainnet and has not been professionally audited.
- **Tamper evidence limits:** a bare hash chain cannot detect truncation of the newest blocks or a complete
  recompute-and-rewrite. Store `head_anchor()` (or the contract's `headHash`) somewhere the log writer cannot edit.
- **Adversarial ML:** results are white-box tests against specific attacks/budgets; no robustness guarantee is claimed.
- **Repository finding:** `artifacts/gnn_reference.npz` (25 MB) holds the standardised features of 227,845 Kaggle
  training rows. Check the dataset licence before redistributing; it is excluded from the release ZIP.
- **Model files:** `.keras`/`.joblib` files execute code on load. Only load files you trust.
- **Reporting:** open a private issue or e-mail the repository owner.
