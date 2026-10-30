// Local development only: compiles with the bundled solc-js so no compiler download is needed,
// and tests run on Hardhat's in-process EVM. No network credentials are read or required.
const { subtask } = require("hardhat/config");
const { TASK_COMPILE_SOLIDITY_GET_SOLC_BUILD } = require("hardhat/builtin-tasks/task-names");
require("@nomicfoundation/hardhat-ethers");
require("@nomicfoundation/hardhat-chai-matchers");

subtask(TASK_COMPILE_SOLIDITY_GET_SOLC_BUILD, async (args, hre, runSuper) => {
  if (args.solcVersion === "0.8.26") {
    return { compilerPath: require.resolve("solc/soljson.js"), isSolcJs: true, version: "0.8.26", longVersion: "0.8.26" };
  }
  return runSuper();
});

module.exports = {
  solidity: { version: "0.8.26", settings: { optimizer: { enabled: true, runs: 200 } } },
  // Hardhat output goes under build/ - its default ./artifacts folder would collide with (and delete files from) the ML artifacts/ directory.
  paths: { sources: "./contracts", tests: "./contracts/test", artifacts: "./build/hardhat-artifacts", cache: "./build/hardhat-cache" },
};
