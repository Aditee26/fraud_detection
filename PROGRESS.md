# Progress log
- Cloned repo; chose `akshita` (latest) as base; branch `feature/research-extension`.
- Phases 1-9 implemented and tested (62 pytest, 9 Hardhat). Phase 10: ZIP built from a clean `git`-tracked file list and re-tested.
- **Incident (resolved):** `hardhat compile` first used the default `./artifacts` folder and deleted tracked JSON files in `artifacts/`. Detected via `git status`, restored with `git checkout`, and Hardhat output moved to `build/`. A test run of the experiments overlapping that moment also lost some new research JSONs; they were regenerated.
- **Bug found by tests and fixed:** attack bound-clipping moved already out-of-range clean points beyond the ε budget; regression test added.
- **Bug found by inspection and fixed:** first transfer smoke run showed scratch == transfer because the transfer branch reused stale predictions; guard assertion + test added.
- Not done: git commit/push/PR (no credentials in this sandbox). Commit locally with the commands in README.
