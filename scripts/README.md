# scripts

Planned (docs/DESIGN.md, section 4):

- `run_study.py`: resume-on-failure wrapper around `lpbench run` (up to 3 attempts), with a
  path-scrubbed `wrapper.log` in the run directory (M4).
- `build_miplib_manifest.py`: builds `src/lpbench/data/miplib_benchmark.csv` from the MIPLIB
  2017 benchmark table (M2).
