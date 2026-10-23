# Security policy

Security fixes are made for the latest release.

## Reporting

Report vulnerabilities privately through the repository's **Security** tab ("Report a
vulnerability"). Do not open a public issue. Include the lpbench version, the operating
system, the impact, and steps that reproduce it.

## Scope

lpbench downloads public test problems from netlib.org and miplib.zib.de (`lpbench fetch`
only), verifies their SHA-256 checksums, and runs the installed solver libraries on them.
Other commands make no network calls.

In scope:

- lpbench reading, writing or deleting files outside the cache or output directory you chose
  (for example through instance names, unit ids or archive member names);
- `lpbench fetch` contacting hosts other than the two above, or accepting a file whose
  checksum does not match the recorded one;
- reports that execute script from data in a run directory.

Out of scope:

- vulnerabilities in the solver libraries or their dependencies (report those to their
  maintainers);
- crashes or wrong answers of a solver on a test problem (these are benchmark findings; see
  CONTRIBUTING.md for how they are handled).
