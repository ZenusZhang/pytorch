- When you are working on a bug, first create a standalone file that
  reproduces the bug and verify it fails in the expected way.  Use this to
  test if your changes work.  Once the change is passing, find an appropriate
  test file to add the test to and make sure to follow local conventions on
  the test file.
- If you are running the real test suite, DO NOT run the entire test suite.
  Instead run only a single test case, e.g., 'python test/test_torch.py TestTorch.test_dir'
- Do NOT run setup.py, you do not have a working build environment
- Do NOT run pre-commit, it is not setup
- To run lint, run 'lintrunner -a' (which will autoapply changes)
- Do NOT attempt to install dependencies, you do not have Internet access
- When you are ready to make a PR, do exactly these steps:
  - git stash -u
  - git reset --hard $(cat /tmp/orig_work.txt) # NB: reset to the LOCAL branch, do NOT fetch
  - git stash pop
  - Resolve conflicts if necessary

- I'm working on writing aten kernels for RISCV-Vector platform, using libtorch as runtime env on qemu and cmodel.
  - In this period, to get familiar the workflow, we should now execute the process on x86 platform.