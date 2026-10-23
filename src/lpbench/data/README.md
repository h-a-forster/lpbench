# Committed instance metadata

No instance data is stored here (the netlib and MIPLIB 2017 pages state no licence). These
files hold names, sizes, checksums and reference values only.

| File | Content | Source |
|---|---|---|
| `netlib_reference.csv` | Netlib problem summary (rows, columns and nonzeros including the cost row, compressed bytes, bounds/ranges flag, optimal value) and the exact optimal values of Koch (2004) | `https://netlib.org/lp/data/readme`; T. Koch, "The final NETLIB-LP results", ZIB-Report 03-05 (2003), Tables 2 and 3 (pp. 5-6), published in Operations Research Letters 32 (2004) 138-142 |
| `miplib_benchmark.csv` | MIPLIB 2017 benchmark set (240 instances): constraints, variables, nonzeros, variable types, status | `https://miplib.zib.de/tag_benchmark.html` and `downloads/benchmark-v2.test`, parsed by `scripts/build_miplib_manifest.py` |
| `checksums.csv` | SHA-256 and size of every downloaded raw file, recorded on first download by `lpbench fetch` | computed locally |
| `miplib_reference.csv` | consensus LP-relaxation objective values (DESIGN 6.3) | computed by `lpbench reference` |

`netlib_reference.csv` columns: `koch_value` is Koch's value transcribed verbatim as
`mantissa E exponent` (32 significant digits, last digit rounded); `koch_nnz` is Koch's
nonzero count of the constraint matrix (cost row excluded), which lpbench's expander
reproduces exactly for all 92 problems; `readme_flag` marks problems for which the readme
itself lists differing values (its CPLEX/MINOS table, and DFL001); `readme_rel_diff` is
`|koch - readme| / (1 + |koch|)`. The published values exclude the objective-row constant
(only E226 has one; see `instances.canonical_reference`).

Two readme entries do not describe the files netlib serves today: MAROS-R7 is listed at
4,812,587 bytes but is served as 746,489 bytes (it passes every checksum line and matches
Koch's dimensions), and BOEING1 is listed with 351 rows while its file declares 352 (it
matches Koch's nonzero count and optimum).
