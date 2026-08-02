# Changelog

## Unreleased

- fix: a quoted continuation value that closes on its own line is
  unwrapped like pymatgen unwraps it — the quoted mirror applied only to
  an unterminated quote, so `POTIM =` followed by `"2.0"` kept its quote
  characters (and any text after the closing quote) and the raw guards
  falsely rejected an INCAR pymatgen parses and runs correctly (a
  regression window of the quoted-mirror fix, never released).
- fix: the blank-value continuation read stops at `;` like pymatgen's own
  value pattern — taking the whole next line made the raw-only guards
  judge text the parser never consumes, falsely rejecting the legal
  multi-tag spelling `TEBEG =` / `300; NSW = 5` that parses and runs
  correctly (a regression window of the continuation fix, never released).
- fix: the raw-text INCAR guards also see a QUOTED value spanning newlines
  — pymatgen's quoted branch is `re.DOTALL`, so `MAGMOM = "1` followed by
  `10000000000*1.0"` on the next line reached proc_val's unbounded list
  expansion (measured linear real RSS, ~150 GB extrapolated at 1e10)
  before the pre-parse repeat cap could see the token, while the
  byte-equivalent unquoted spelling was cleanly rejected. With no closing
  quote anywhere the raw text is kept as-is (the unbalanced-quote guards
  own that case post-parse).
- fix: the raw-text INCAR guards (repeat-count cap, corrupted-token
  rejection, real-tag repair) also see a value written on the line AFTER
  the `=` — pymatgen's `\s*=\s*` crosses the newline on the value side,
  so `MAGMOM =` followed by `10000000000*1.0` on its own line detonated
  proc_val's ~80 GB list expansion inside `Incar.from_file`, before any
  guard could see the token, and the same spelling bypassed the
  corrupted-token and repair guards. The swallow guard deliberately keeps
  the line-scoped read (a blank staying visibly blank is what makes the
  parser disagreement detectable).
- fix: the MD divergence guard's message names the per-axis bound — when
  the per-axis rule fired, the volume-only wording reported a span-volume
  far below the printed maximum (a self-contradictory diagnostic), unlike
  its NEB twin.
- fix: the unwrapped-coordinate span guards (NEB image read and the MD
  divergence guard) also bound each axis individually — the volume rule
  floors each axis at 1 Å before taking the product, so a single-axis
  excursion evaluated to its own length and stayed under the 1e9 Å³ cap
  however large it grew, while the neighbour search's per-axis image
  replication kept growing linearly (a 3.9e7 Å single-axis span is a
  measured MemoryError). The per-axis cap is 1e7 Å, calibrated between the
  largest completing case (3.9e6 Å, 2.3 s) and the smallest failing one;
  a cell whose own width exceeds the cap keeps its width as the limit.
- fix: a NEB image whose coordinates lie far outside the cell (the shape a
  diverged CONTCAR reused as an image has) is rejected with a clean input
  error — the optimization branch deliberately preserves unwrapped image
  coordinates, so the full excursion reached the backend's periodic
  neighbour search (a measured 19 GB allocation for a 2-atom cell,
  MemoryError, a retryable exit 2 in server mode). The unwrapped read now
  bounds the coordinate span with per-axis floors and a cell-aware limit;
  boundary-adjacent bands remain unwrapped and accepted.
- tests: three tests no longer assume optional/new dependencies of the
  development environment — the BAM device-collapse test mirrors the
  resolver's own torch-missing fallback instead of importing torch
  unconditionally, and the two INCAR newline/blank-swallow tests probe the
  installed pymatgen's actual parse behavior (pymatgen >= 2026 crosses a
  newline-split key and lets a blank value swallow the next assignment;
  2025-era releases drop the tag instead) and assert the guard outcome
  correct for that behavior, keeping full detection power on both.
- fix: a FIFO (named pipe) at `MODEL` is rejected with a clean input error
  instead of hanging — MODEL was the one user-supplied input path with no
  non-regular-file check, so the loader's `open()` blocked forever: a
  silent one-shot hang, and in server mode a daemon child blocked mid-load
  while holding the endpoint's pidfile, refusing every later serve on that
  socket until the orphan was killed manually. The refusal is narrowed to
  FIFOs; directory-shaped checkpoints remain legal.
- fix: the MatGL constructor-compatibility fallback no longer discards the
  `stress_unit="eV/A3"` pin — when the installed calculator declares
  `stress_unit` but rejects another keyword (such as `device`), the
  TypeError retry dropped every keyword argument, so the calculator
  silently fell back to matgl's GPa default and every reported stress and
  pressure was ~160.2× too large with exit 0. The retry now drops only the
  other keywords, and fails loudly if the verifiably-declared pin itself
  is rejected instead of computing silently wrong numbers.
- fix: the daemon launcher's readiness line no longer turns a successful
  start into exit 1 — with an unbuffered stdout whose consumer had exited
  (`PYTHONUNBUFFERED=1 vpmdk serve --daemon | head -1`), the EPIPE surfaced
  inside the success `print()` itself and a raw BrokenPipeError traceback
  reported failure for a live, model-holding resident, which a supervisor
  then orphaned. The flush-time drain guard only covered buffered stdout;
  the client's `_write_line` already guarded its immediate-write face.
- fix: the pre-load endpoint reservation is also released when `_bind()`
  fails — pidfile ownership transfers only at the last statement of
  `_bind`, so a bind failure (socket path too long, parent turned
  unwritable during the model load, `EADDRINUSE`) skipped both release
  paths and left `<socket>.pid` on disk; for a long-lived in-process
  `serve_cli` caller the leaked record named a live pid and permanently
  refused every other serve on that endpoint.
- fix: `vpmdk serve` reserves the endpoint atomically (the pidfile, via
  `O_CREAT|O_EXCL` with live-owner detection) BEFORE loading the backend —
  two concurrent serves targeting the same unused socket both passed the
  socket-path check and both loaded a potentially GPU-sized model, with the
  loser only detected at bind time. The loser now fails before the load;
  the reservation is released on the startup error tail, and a startup's
  own reservation is recognized by the deleted-socket protection (whose
  message now also covers a server that has not bound its socket yet).
- fix: the client reads the process umask without mutating it (from
  `/proc/self/status` on Linux; the `os.umask` read-back fallback is
  serialized behind a lock) — the read-back dance is process-global, so two
  concurrent `run()` calls could observe a temporary mask of 0 or restore 0
  as the process mask, making every later file world-writable.
- fix: `VPMDKClient` validates `connect_timeout` in the constructor — a
  negative, NaN, infinite, or beyond-`time_t` value previously passed
  `float()` and made the first request die inside `socket.settimeout()`,
  outside the connection-error handlers, leaving the new socket's
  descriptor to garbage collection.
- fix: a failed foreground `serve` start with an explicit `--log-file` no
  longer leaks the preflight log-probe file descriptor — the success path
  closed it right after server construction, but when backend loading or
  server construction failed only the error tail ran, so a long-lived
  caller invoking `serve_cli` repeatedly leaked one descriptor per failed
  start (eventually EMFILE).
- fix: the declared-species input gate also reads a Z-keyed element table
  (bam-torch's `uniq_element`) — the published BAM-MP-core checkpoint
  declares 89 elements spanning Z=1..94 with holes at Po/At/Rn/Fr/Ra, and a
  structure containing one of those died mid-forward-pass with a raw
  KeyError (a retryable exit 2 in server mode) instead of the clean
  input-time rejection the symbol-keyed (`element_types`) backends get.
- fix: the resident server's DEVICE identity comparison mirrors bam-torch's
  own device resolution for `MLP = BAM` — upstream maps only the literal
  `cpu` to the CPU and every other spelling (`cuda`, `cuda:1`, `gpu`,
  `cpu:0`, …) to CUDA-if-available-else-CPU with the index dropped, so raw
  string comparison rejected byte-identical request/resident pairs with
  exit 5 and silently equated pairs that genuinely differ.
- fix: the MD divergence guard floors each spatial span at 1 A before taking
  the product — for collinear or constrained motion the transverse spans are
  exactly zero, so the raw product stayed zero no matter how far the atoms
  flew apart and a one-axis divergence still reached the neighbour search.
  The bound is also cell-aware (the larger of the global cap and the cell's
  own floored bounding box), so cells the input-time cap legitimately admits
  are not newly rejected at run time.
- fix: a resident GRACE server now relays the "GRACE ignores the DEVICE tag"
  warning to `vpmdk run` clients whose request BCAR carries `DEVICE` — the
  resident builder never re-runs per request, so only one-shot runs printed
  it, breaking server/one-shot output equivalence.
- feat: new `MLP = BAM` backend for BAM-torch (Bayesian Atoms Modeling)
  RACECalculator checkpoints, including the published MPtrj foundation
  checkpoint `BAM-MP-core.pkl` (Hugging Face `myung-group/BAM_MPtrj_v1`).
  `MODEL` is a required local path (BAM-torch ships no named-model
  downloader); `DEVICE` is optional and defaults to bam-torch's own cpu
  selection when blank.
- fix: GRACE now prints a warning when the `DEVICE` tag is set —
  TPCalculator takes no device argument, so the tag was silently swallowed
  and, on TensorFlow builds without working CUDA support, users believed a
  CPU run was on GPU. Device placement follows the installed TensorFlow
  build; documented in the backends reference.
- fix: `lammps.lammpstrj` frames now carry an `element` column (LAMMPS
  `dump_modify element` style, appended after the legacy columns) — the
  type→species mapping existed nowhere in the file, so `ase.io.read`
  without an out-of-band `specorder` interpreted the type index as an
  atomic number: an Si trajectory read back as hydrogen with every
  mass-weighted quantity (kinetic energy, temperature, per-species
  MSD/RDF) silently off by the mass ratio, and multi-species cells read
  back with the wrong stoichiometry entirely.
- fix: molecular dynamics now stops with a clear "MD trajectory diverged"
  calculation error before the force call when the unwrapped coordinate
  span exceeds the supported periodic-cell volume bound — a trajectory
  thrown out of the cell by an oversized `POTIM` or `LANGEVIN_GAMMA`
  previously turned the next neighbour-search call into an OOM-grade
  allocation (a measured 152 GB request) or an uninterruptible native
  spin that ignored SIGINT and wedged a resident server.
- fix: the stress-vs-`ISIF>=3` input gate keys on whether the NEB branch
  will actually run (numbered image directories present), not on the
  INCAR-only NEB heuristic — a flat workdir whose INCAR merely carried a
  stray `SPRING`/`LCLIMB` line skipped the gate and died on the first
  optimizer step (a retryable exit 2 in server mode) while the identical
  INCAR without the tag got the clean input error (regression window of
  the previous entry, never released).
- fix: the raw INCAR assignment reader now crosses a newline between the
  key and `=` exactly as pymatgen does (`NSW` on one line, `= 1e5` on the
  next) — that spelling parsed normally in pymatgen but was invisible to
  every raw-text guard (repeat caps, corrupted-token rejection, the
  embedded-assignment rule, the SPRING repair), so all of them were
  bypassable by moving the `=` to the next line. The value-side line
  scoping that makes the blank-value swallow visible is unchanged.
- fix: reject a stress-less backend combined with `ISIF>=3` in a
  relaxation — `ISIF>=3` makes stress part of the dynamics (ASE's cell
  filter calls `get_stress()` every step), so the run died on the first
  optimizer step and server mode reported it as a retryable exit 2 for a
  permanently broken configuration. Stress-as-output cases (ISIF≤2, single
  point, MD, NEB, force constants) keep running with the existing warning.
- fix: the INCAR repeat-expansion cap tokenizes exactly as pymatgen does —
  every hand-rolled split was bypassable by one junk character
  (`MAGMOM = (2000000000*1.0)`), letting a few hundred bytes drive a
  billions-of-entries allocation inside the resident worker.
- fix: the INCAR repeat caps apply only to tags some parser layer actually
  expands (pymatgen's list-typed keys plus `MAGMOM`) — a free-text title
  like `SYSTEM = 1000001*study`, which is never expanded, was rejected
  although it ran fine before (cross-review finding).
- fix: only the always-written artifacts (OUTCAR/OSZICAR/vasprun.xml/
  CONTCAR) are preflighted unconditionally — XDATCAR (MD-only) and
  CHGCAR/energy.csv (flag-gated) are now checked by the sites that will
  actually write them, so a static run no longer aborts over an ignored
  CHGCAR directory the `Note:` line itself calls unused (cross-review
  finding).
- fix: the foreground `--log-file` probe keeps its writer fd open until the
  server's FileHandler has opened the path — closing it immediately
  delivered EOF to a FIFO's waiting reader, which exited during the model
  load and left the post-load reopen blocking forever (cross-review
  finding).
- fix: the INCAR repeat-expansion guard normalizes commas to separators —
  a Fortran-legal comma-separated `N*value,N*value,...` list was a single
  token whose expansion read as one cap-sized group, so a 6 KB INCAR could
  request a 5×10⁸-entry allocation inside the resident worker before any
  cap fired.
- fix: the same-line INCAR assignment guard now fires only when the
  embedded tag is genuinely absent from the parse — the standard VASP
  trailing-comment style (`NSW = 3   (ignored when IBRION=-1)`) parses
  exactly as written and must keep running (regression window of the
  previous entry, never released).
- fix: `vpmdk serve` drains **stderr** as well as stdout before returning —
  the exit-code guard covered stdout only, so a dead stderr consumer (the
  foreground logger and the ML-stack import warnings both write there) let
  CPython's finalization flush override a successful start or a clean
  shutdown to exit 120, which a supervisor reads as a failed start while
  the resident keeps holding VRAM.
- fix: the BCAR unknown-tag warning no longer fires for the eight
  individually-read `CHARGE_DEEPCDP_*` option tags — the vocabulary now
  harvests a set kept beside those reads, so a documented, fully-consumed
  DeePCDP SOAP configuration is not falsely called unrecognized (regression
  window of the previous entry, never released).
- fix: in server mode the BCAR unknown-tag warning is emitted at one-shot's
  position (after the `Note:` lines, and only when the INCAR read
  succeeded) — the hoisted request parse suppresses it and `run_workdir`
  re-emits it for the passed tags, restoring stdout equality between the
  two paths.
- fix: reject an INCAR line that carries a second assignment without a `;`
  separator — pymatgen reads `NSW = 5 IBRION = 2` entirely as NSW's value,
  so IBRION silently never existed and the run changed mode (relaxation →
  single point) with exit 0; the `;` spelling parses both tags and stays
  legal, and free-text `SYSTEM` titles are exempt.
- warn: a BCAR tag outside the vocabulary any consumer reads (a misspelled
  `MODELL`, `DEVCIE`, `WRITE_CHGCARR`) is now warned about instead of
  silently ignored — the run otherwise used the default model/device and
  skipped requested outputs with no diagnostic, while the byte-analogous
  INCAR typo already warned. The tag stays in the parsed mapping exactly as
  documented.
- fix: reject `MDALGO=3` (Langevin) for a single-atom cell at input time —
  ASE's `fixcm=True` default divides by N−1, so the run died on the first
  step with `ZeroDivisionError`, classified as a retryable exit 2 in server
  mode for a fixed property of the input.
- fix: the POSCAR parse no longer consults `*POTCAR*` sibling files
  (`check_for_potcar=False`) — pymatgen otherwise passed a sibling's
  symbols as `default_names`, silently relabelling a declared-Si2 deck to
  Cu2 (4.6 eV wrong, exit 0) whenever a parseable `POTCAR_Cu`/`POTCAR.bak`
  was present without an exact `POTCAR`, where real VASP reads only the
  exact file. VPMDK's own `<workdir>/POTCAR` reconciliation is the sole
  species authority; this also removes the parser's sibling-open side
  effect entirely.
- fix: reject a POSCAR that combines a negative scale factor (VASP: target
  cell volume) with Cartesian coordinates — the parser scales the lattice
  by the derived `(-scale/vol)**(1/3)` factor but multiplies the Cartesian
  positions by the raw negative number, silently producing a different
  structure (fractional 0.75 read as −30.67, energy wrong by 0.36 eV,
  corrupted CONTCAR propagating to continuations) with exit 0. Direct
  coordinates with a negative scale parse correctly and stay accepted.
- fix: the species/ion-counts cross-check no longer counts a Fortran `!`
  comment on the species line as extra species — `Si   ! silicon` parses to
  the exactly correct composition and must keep running (regression window
  of the previous entry, never released); only leading valid element
  symbols are compared.
- fix: NEB per-image single points and the IBRION=5/6/7/8 force-constants
  path now echo the INCAR's `NSW` in vasprun.xml instead of the
  step-count fallback `1`, matching the parent aggregate and the flat
  path.
- fix: a failing request-BCAR parse in server mode still streams the
  `Note: KPOINTS/WAVECAR/CHGCAR detected but not used` lines the
  byte-identical one-shot run prints first.
- fix: `NHC_NCHAINS` is read through the shared numeric extractor like
  every sibling tag — a legal Fortran trailing comma (`5,`) silently
  dropped the tag and sampled the default chain length.
- fix: reject selective-dynamics flags spelled other than bare `T`/`F` —
  pymatgen's exact `value == "T"` comparison read every other Fortran
  spelling VASP accepts (`.TRUE.`, `TRUE`, `t`, `.T.`) as False, silently
  FREEZING atoms the file marks free and rewriting CONTCAR with the inverse
  flags.
- fix: reject a POSCAR whose species line length disagrees with its
  ion-counts line — pymatgen zips the two and silently drops the trailing
  species, so `Si Ge` over `2` computed a Si2 energy for a file that says
  SiGe with exit 0.
- fix: `NEQUIX_USE_COMPILE`/`NEQUIX_COMPILE` are no longer compared against
  a jax-backend Nequix resident — upstream reads the flag only in its torch
  branch, so the tag cannot change the calculator there and comparing it
  rejected (exit 5) requests one-shot builds identically.
- docs: the scheduler example now creates its socket inside `mktemp -d`
  instead of a predictable name under shared `/tmp`, matching the
  document's own guidance and the other examples.
- fix: the unbalanced-quote INCAR guard no longer counts a following
  BLANK-valued tag as swallow evidence — pymatgen drops empty-valued tags
  for an independent reason, so a legal file combining a quoted title with
  a trailing `NPAR =` was rejected with a factually false diagnosis
  (regression window of the previous refinement, never released).
- fix: a dangling or self-referential symlink at INCAR, BCAR, or POTCAR is
  rejected as an input error instead of being read as an ABSENT file —
  `os.path.exists` follows links, so a broken `INCAR -> ../shared/INCAR`
  silently ran a requested 200-step relaxation as a single point and a
  broken BCAR silently fell back to the CHGNET default, exit 0, no
  diagnostic.
- fix: the `*POTCAR*`-sibling guard rejects only a FIFO (the node that can
  actually block pymatgen's probe open) — a directory or symlink named
  `POTCARs`/`POTCAR_backup` is ignored by pymatgen's own try/except and
  those layouts keep running (regression window of the previous FIFO fix,
  never released).
- fix: accept a VASP-4 POSCAR that carries the element symbol at the end
  of each coordinate line without a POTCAR — pymatgen reads the real
  species from those tokens; judging the species line alone rejected a
  format that computed correctly.
- fix: bound the ion count of a MULTI-LINE species/counts POSCAR block —
  pymatgen (and VASP, for >20 species groups) puts the counts on line 8+,
  where the fixed two-line window never looked, so an absurd count expanded
  per-atom lists unbounded inside the resident worker.
- fix: reject a selective-dynamics mask with fewer than three T/F flags —
  pymatgen parses the short mask and `AseAtomsAdaptor` silently maps it to
  NO constraint, so the atoms the user froze relaxed freely with exit 0 and
  a CONTCAR whose `Selective dynamics` block was gone.
- fix: `serve --daemon` with a reader-less FIFO at an explicit `--log-file`
  fails in seconds with a diagnostic instead of blocking the forked child
  forever, timing out the launcher after 600 s and leaking a stuck orphan
  (`O_NONBLOCK` open + restored blocking mode, so a FIFO with a reader
  still works).
- docs/warn: disclose that `MDALGO=1` (Andersen) freezes the center of
  mass while the reported temperature divides by all 3N degrees of
  freedom, so OSZICAR/stdout read ≈(3N−3)/3N of `TEBEG` (−25% for 4
  atoms); the sampled ensemble itself is at `TEBEG`, and VASP reports over
  3N−3.
- fix: write the LAMMPS dump's `vx vy vz` columns in `metal` units (Å/ps)
  as every dump consumer assumes — they were written in ASE's internal
  velocity unit, making every velocity 98.227× too small (kinetic
  quantities 9648×) in a file whose positions and box are exact.
- fix: guard a FIFO planted at any `*POTCAR*` sibling of POSCAR before the
  POSCAR parse — pymatgen's `Poscar.from_file` globs and opens those
  siblings itself, upstream of the existing POTCAR guard, so the resident
  worker (and one-shot) hung forever in a blocking open that
  `stop --force` cannot preempt.
- fix: bound the cell VOLUME (1e9 Å³) at input time, not just each axis —
  the neighbour search allocates bins proportional to volume/cutoff³, so a
  20000 Å cube with 2 atoms (50× under the per-axis ceiling) died asking
  for 152 GB as a "retryable" exit 2, and a 4000 Å cube wedged the
  resident worker past 200 s.
- fix: the unbalanced-quote INCAR guard now keys on the actual swallow
  evidence (a following raw tag missing from the parse) instead of a
  leading quote alone — `SYSTEM = "run #3"` and a forgotten quote on the
  last tag parse harmlessly and must keep running (a regression window of
  the previous entry, never released).
- fix: the OUTCAR `NEB: projections on to tangent (spring, REAL)` line now
  writes exactly two numbers, as real VTST does — the previously appended
  third field (a non-negative max perpendicular-force magnitude) was what
  VTST's `nebbarrier.pl` read as the interior-image force, so
  `nebspline.pl` was fed a slope ≤ 0 at every interior image and fabricated
  saddle points and minima that do not exist in the computed energies.
- fix: classify permanent path pathologies as input errors (exit 1) in
  server mode — a self-referential symlink loop at an artifact path
  (`ELOOP`), an over-long symlink target (`ENAMETOOLONG`), and a dangling
  symlink into a missing directory (`FileNotFoundError`) all fell through
  the exception-type tuple to the retryable exit 2, unlike the one-shot
  path.
- fix: the unbalanced-quote INCAR guard now also fires when the swallowing
  tag is bool/list-typed — `LWAVE = ".FALSE.` parsed to the scalar `False`
  (not a multi-line string), so the string-typed check missed it while
  every following tag was still silently deleted; the raw text is now
  judged directly.
- fix: transmit the client's umask with each `run` request and apply it
  around the calculation — output artifacts were created with the server's
  launch umask, so `umask 077; vpmdk run` silently produced world-readable
  outputs from a umask-022 resident, and the reverse broke group
  post-processing pipelines with 0600 files after exit 0.
- fix: torch's C++ `TORCH_WARN_ONCE` diagnostics now reach the client on
  every job, not only the first of a resident's lifetime — each job enables
  torch's `warnAlways`, forwarding occurrences to the Python warnings layer
  where the per-job filter scope restores exactly once-per-job.
- fix: relayed server-side stderr is dropped when the client's own stderr
  is closed (`2>&-`), matching one-shot behavior instead of injecting
  warning text into the stdout stream scripts parse.
- fix: reject a corrupted `FORCE_CONSTANTS_DISPLACEMENT`/`PHONON_DISPLACEMENT`
  token — a Fortran D exponent (`1D-2`) or a letter O for zero was read as
  its leading digits, silently turning the intended 0.01 Å step into
  1.0 Å.
- fix: reject an INCAR whose parsed value spans multiple lines — an
  unbalanced quote (`SYSTEM = "Cu bulk` with the closing quote forgotten)
  made pymatgen's DOTALL quoted-value branch silently swallow every tag up
  to the next quote in the file, so a requested 200-step relaxation ran as
  a single point with exit 0 while the OUTCAR echo still listed the lost
  tags.
- fix: relay the calculation's stderr to the submitting client — third-party
  warnings (ASE's `fixcm` NVT-sampling `FutureWarning`, pymatgen and numpy
  warnings) went to the server's private log instead of the client, and
  Python's per-process warning dedup dropped them entirely after the first
  job of a resident's lifetime. They now stream as `log` events tagged
  `"stream": "stderr"` and the client writes them back to its stderr; each
  job re-emits warnings like a fresh one-shot process.
- fix: run the model-declared species gate on the one-shot NEB path — both
  NEB call sites checked coverage before the per-image calculator existed,
  so a band with elements outside the model's declared table died with a
  raw `KeyError` traceback after writing partial per-image artifacts, while
  the flat path and the resident-server submission printed the clean
  input-error diagnostic.
- fix: reject `MDALGO=2`/`4` (Nose-Hoover chain) combined with constrained
  atoms (POSCAR selective dynamics) as an explicit input error — ASE's
  integrator keeps internal momenta that ignore constraints (and a 3N
  thermostat target with no constrained-DOF reduction), so frozen atoms
  accumulated phantom momenta and the real atoms sampled 25–85 K where
  `TEBEG` said 300 K, with exit 0 and no warning. Langevin/Andersen/CSVR
  handle constraints correctly and remain available.
- fix: classify a read-only *filesystem* (`EROFS` from a read-only mount)
  as an input error (exit 1) in server mode — Python has no `OSError`
  subclass for it, so it fell past the read-only-tree branch into the
  retryable exit 2, unlike permission-bit read-only-ness and the one-shot
  path.
- fix: `vpmdk status --json` no longer degrades a conforming server frame
  into invalid UTF-8 — a surrogate-escaped byte in a workdir or model path
  now falls back to ASCII-escaped JSON (mirroring the server's own wire
  fallback) so the output stays parseable; the human-readable rendering
  keeps byte round-tripping.
- perf: server-mode NEB gives each band image its own result cache in
  front of the single resident calculator — sharing the calculator object
  defeated ASE's per-image caching, recomputing already-computed
  geometries (a 5-image / 21-step band cost 420 forward passes where
  one-shot cost 35). Results are unchanged.
- fix: a foreground `serve` with an explicit `--log-file` now probes the
  log path before the model load, like the daemon path — an unwritable
  path cost a full checkpoint load per retry, and a reader-less FIFO at
  the path blocked startup forever with no diagnostic.
- fix: a non-regular POTCAR in a NEB workdir is classified as an input
  error (exit 1) like in a flat workdir — the POMASS disclosure ran
  outside the input-phase wrapper, so the same POTCAR was a retryable
  exit 2 in server mode and a raw traceback one-shot.
- fix: bound the PRODUCT of all leading repeat factors in an INCAR token,
  not just the first — pymatgen's nested `count1*count2*value` spelling
  (e.g. `MAGMOM = 100000*100000*1.0`, 1e10 entries) passed both the
  per-token and total caps and exhausted memory at parse time inside the
  resident server. VPMDK's own recursive MAGMOM expander had the same
  per-level hole and now bounds the cross-level product and the total.
- fix: classify a permanently unconfigured charge backend (missing
  `CHARGE_MODEL`, unresolvable DeepCDP checkpoint, or a `CHARGE_PYTHON`
  interpreter that does not exist) as an input error (exit 1) in server
  mode instead of the retryable calculation error (exit 2) — a retry
  driver re-ran the full ionic loop on the same broken directory forever,
  while one-shot exited 1 for the byte-identical input.
- fix: reject an existing-but-unwritable custom socket parent directory
  before the model load — `vpmdk serve --socket /run/vpmdk.sock` paid the
  full load and then died in `bind()` with a `Permission denied` that
  named no path at all (the default parent was already checked pre-load).
- fix: floor finite-difference displacements (`POTIM` for IBRION=5/6,
  `FORCE_CONSTANTS_DISPLACEMENT` for IBRION=7/8) at 1e-6 Å — a
  tiny-but-positive value underflowed the double-precision positions to a
  no-op and wrote an ALL-ZERO Hessian to vasprun.xml as a successful run.
- fix: rewrite the coincident-atom input guard as an O(N) grid-bucket
  algorithm. The all-pairs minimum-image matrix cost ~17 GB and ~17 s at
  input time for an ordinary 4096-atom supercell, and an allocation failure
  was rewritten into an input error for a fine structure.
- fix: bound the TOTAL `N*value` expansion per INCAR tag, not just each
  token — many tokens of 1e6 each still expanded without bound.
- fix: an explicit `--log-file` that happens to spell out the default
  `<socket>.log` name may be a symlink in the foreground too (previously
  refused after the full model load, while the identical `--daemon` line
  accepted it). Library callers passing the derived path without naming it
  keep the hardened refusal.
- fix: the pre-load pidfile gate now also mirrors the foreign-owner refusal
  — a pidfile planted by another user in a shared sticky directory passed
  the gate and aborted only after the full model load, on every retry.
- fix: disclose that `MAGMOM` is inert for results — it is attached as ASE
  initial magnetic moments, but no supported backend reads initial moments,
  so FM and AFM orderings produced byte-identical energies with exit 0 and
  no warning (while `ISPIN` was explicitly warned about).
- fix: `vpmdk serve` with fd 1 closed (`1>&-`) no longer turns a successful
  start into exit 1 — `sys.stdout` is None in that state and the stdout
  guard's flush raised AttributeError out of the command's finally.
- fix: two device-index edges — the bare string `:0` is no longer stripped
  to a blank (which silently rerouted it into autodetect acceptance), and
  `cpu:N` collapses to `cpu` for every index (torch's cpu type has one
  underlying device; `cpu:1` vs `cpu` was a spurious exit 5).
- fix: bound `SPRING` at 1e9 like every sibling scalar — `SPRING=-1e300`
  froze the NEB band silently with exit 0.
- fix: warn when `NSW>1` is given without `IBRION` — VPMDK runs a single
  point (default `IBRION=-1`) where real VASP would default to MD.
- fix: make the `SPRING` repair actually stick with real pymatgen. The
  repaired value was assigned through `Incar.__setitem__`, which re-runs the
  typed parse and re-floored it to the truncated int — a measured no-op; the
  repair now writes into the mapping's backing store.
- fix: disclose that POTCAR `POMASS` is not read. An isotope-edited POTCAR
  (the canonical VASP deuterium workflow) ran at ASE's standard-isotope mass
  with exit 0 and no disclosure; a warning now lists every differing entry.
- fix: refuse a well-formed pidfile recording a DIFFERENT socket before the
  model load — the pre-load gate mirrored only half of the pidfile writer's
  refusal, so a moved runtime directory paid the full load on every retry.
- fix: reject a tiny-positive `SMASS`/`NHC_PERIOD` (e.g. `1e-300`) whose
  damping time makes the Nose-Hoover thermostat mass underflow to zero — the
  first MD step was nan, classified as a retryable calculation failure.
- fix: disclose that an absent `LCLIMB` runs PLAIN NEB while VTST's
  documented default is `.TRUE.` (climbing image). A canonical VTST input
  silently underestimated the barrier (measured 24% low) with exit 0; the
  default is unchanged, the divergence is now warned about and documented.
- fix: extend the element-coverage gate to MatRIS (hard-wired 94-row atom
  embedding) and to models that declare their own element table (matgl's
  `element_types` is an 89-element set with holes — alpha-Po failed every
  forward pass as a "retryable" KeyError). Tag-inheriting NEB requests are
  now checked against the resident's backend instead of the CHGNET default.
- fix: refuse an unusable pre-existing pidfile (foreign content, FIFO,
  symlink) before the model load for pidfile-writing launches, instead of
  aborting after the load inside the pidfile writer. Servers running without
  a pidfile leave foreign files alone, as before.
- fix: reject a structure containing elements beyond the resident model's
  fixed coverage (CHGNet's composition head is hard-wired to atomic numbers
  up to 94) at input time. A POSCAR with Cm failed every forward pass with a
  torch shape error, classified as a retryable calculation failure — a
  spec-following retry driver re-submitted the permanently uncomputable
  structure forever.
- fix: `vpmdk serve` no longer lets a dead or full stdout override its exit
  status. The daemon parent's buffered success print made CPython's
  interpreter-exit flush fail and turn a successful start (or a clean
  foreground shutdown) into exit 120 — a launcher then read a live
  VRAM-holding resident as a failed start.
- fix: extend the fractional-integer rejection to the CHGCAR grid
  (`NGX*`/`NG*F`) and pseudo-SCF (`NELM`/`NELMIN`/`NELMDL`) families —
  `NGXF = 100.5` silently resolved a 100-point grid and `NELM = 2.7` wrote
  three mutually contradictory echoes, where VASP refuses the file.
- fix: repair `SPRING` from the raw INCAR text. pymatgen int-types this REAL
  tag, so `SPRING = -5.5` (legal in VASP/VTST) was silently floored to -5 —
  and the recent fractional-literal guard then rejected the legal file
  outright. The true float value is now restored into the parsed mapping.
- fix: key the fractional-literal rule on VASP's integer SEMANTICS instead
  of pymatgen's typing: `NHC_NCHAINS = 2.7`, `IOPT = 7.5`, and
  `ICHAIN = 0.7` (float-typed by pymatgen, then silently floored by VPMDK's
  own coercers) are now rejected like `NSW = 2.7`.
- fix: disclose VPMDK's Andersen default. `MDALGO=1` without `ANDERSEN_PROB`
  uses a collision probability of 0.1 per atom per step, where real VASP's
  documented default is 0 (collision-free NVE); a run-time warning now states
  the substitution instead of silently sampling a different ensemble.
- fix: resolve a relative `--socket`/`VPMDK_SOCKET` path from a deleted
  working directory as a clean one-line error instead of a raw
  FileNotFoundError traceback from inside the client constructor.
- fix: refuse a `serve` restart up front when the socket file was deleted
  externally under a LIVING server (naming the pid to kill), instead of
  paying for the full model load and only then failing in the pidfile guard.
- fix: echo the INCAR-requested `NSW` into vasprun.xml instead of the number
  of steps actually performed. pymatgen's `Vasprun.converged_ionic` rule is
  "converged iff `len(ionic_steps) < NSW`", so the old echo made every
  successfully converged relaxation read `converged=False` — a constant,
  information-free flag that made custodian/atomate2-style gates re-run
  healthy runs.
- fix: a broken stdout consumer (`vpmdk run | head -1`) no longer turns a
  completed calculation into a raw traceback with exit 120. Output delivery
  is best-effort; the broken stream is pointed at /dev/null and the
  documented exit code is preserved.
- fix: guard the LAMMPS trajectory writer against a FIFO at its
  (configurable) output path — the one artifact the fixed-name FIFO sweep
  could not cover.
- fix: reject a fractional literal for an integer INCAR tag (`NSW = 2.7`
  silently floored to 2; real VASP refuses the file). A trailing dot
  (`100.`) stays legal.
- fix: a FIFO planted as `KPOINTS` (read at OUTCAR-header time) or at any
  output artifact path (`OUTCAR`, `OSZICAR`, `vasprun.xml`, `CONTCAR`,
  `XDATCAR`, `CHGCAR`, `energy.csv`) no longer wedges the worker forever in
  a blocking `open()`. KPOINTS degrades to "no header lines" like the other
  unreadable cases; output paths are checked at one pre-computation choke
  point and rejected as input errors.
- fix: forward `PSTRESS` into the NEB per-image single-point and MD
  branches. Image artifacts used the raw stress convention while the parent
  reported the corrected one for the same image, so barriers computed from
  image files disagreed with the parent by the full `PV`.
- fix: stop rejecting free-text `SYSTEM` titles whose first token looks
  numeric (`SYSTEM = 1D5 sample`, `SYSTEM = Infinity study`) — a regression
  of the corrupted-token guard against files that ran correctly before. The
  numeric checks now apply only to tags some reader treats as a number.
- fix: reject `IBRION=44` (improved-dimer transition-state search) as
  unsupported instead of silently running a plain minimization away from the
  requested saddle point, matching the existing rejection of the VTST
  spelling (`ICHAIN=2`).
- fix: a stale foreign graph-converter spelling (e.g. `MATRIS_GRAPH_CONVERTER`
  on a CHGNET resident) no longer suppresses the startup detection of the
  loaded model's algorithm, which left the resident rejecting every explicit
  converter request in both directions.
- fix: strip the internal `ERROR:` readiness-pipe marker from failed
  `serve --daemon` diagnostics (previously printed doubled:
  `Error: daemon failed to start: ERROR:...`).
- fix: warn when an MD INCAR requests cell dynamics. A standard VASP NPT
  input (`IBRION=0`, `ISIF=3`, `MDALGO=3`, `PSTRESS`) silently ran fixed-cell
  NVT with exit 0 while every artifact claimed the pressure ensemble — the
  cell never moved. VPMDK MD integrates ions only; the warning states that no
  barostat is applied and `ISIF`/`PSTRESS` only affect output conventions.
- fix: reject non-regular input files (a FIFO planted as
  `POSCAR`/`INCAR`/`BCAR`/`POTCAR`) before opening them. A FIFO read-open
  blocks forever, which wedged the resident worker permanently: status
  reported busy forever, queued jobs timed out, and even `stop --force`
  could not preempt the blocked open. Symlinks to regular files stay legal.
- fix: advertise the graph-converter algorithm the loaded model actually
  carries. A tagless CHGNET resident advertised
  `GRAPH_CONVERTER_ALGORITHM=None`, so a request spelling out the algorithm
  the bundled default model already uses (`fast`) was rejected with exit 5
  while the one-shot CLI computed byte-identical numbers.
- fix: extend the MD-scalar magnitude bound to `SMASS` and the thermostat
  parameters. `SMASS=1e300` raised a raw OverflowError from `tdamp**2`
  inside ASE's thermostat (one-shot exit 1 traceback, server exit 2
  retryable), and `SMASS=-1e300`'s Langevin promotion bypassed the
  `LANGEVIN_GAMMA` bound entirely.
- fix: undo the PSTRESS transformations when the NEB parent aggregate reads
  image vasprun.xml files back. The parent writers re-apply them, so a
  PSTRESS NEB band reported `E + 2·PV` in the parent vasprun, `E + PV` in the
  parent OSZICAR/OUTCAR, and a doubly-shifted parent pressure, while the
  per-image files were correct.
- fix: treat a device index of 0 as the unindexed spelling for every device
  type. Only the literal `cuda:0` was normalized, so a `cpu:0`/`cpu`
  resident/request pair — byte-identical in torch — was rejected with exit 5
  in both directions while the one-shot CLI computed it. `:1` and higher stay
  distinct.
- fix: bound the magnitude of `TEBEG`/`TEEND`/`POTIM`/`LANGEVIN_GAMMA` at
  1e9. Overflow-scale values (an exponent typo like `1e300`) passed the
  finiteness checks and produced nan in the first force call — classified as
  a retryable calculation failure (exit 2) for a permanently broken INCAR.
  The largest value measured to complete (`TEBEG=1e6`) stays legal.
- fix: write the enthalpy `E + PSTRESS*V` into vasprun.xml's
  calculation-level energy fields and echo `PSTRESS` in the `<incar>` and
  `<parameters>` blocks, matching real VASP (scstep and OUTCAR/OSZICAR keep
  the plain energy). pymatgen consumers used to read energies that silently
  differed from a real VASP run by `PSTRESS*V` per structure, and ASE's
  reader — which subtracts `parameters['pstress']*V` — only agreed by
  accident because neither the PV term nor the declaration was present.
- fix: bound INCAR `N*value` repeat counts (1e6) before the parse. pymatgen
  expands `MAGMOM = 10000000000*1.0` into a ~160 GB list at
  `Incar.from_file` time — before any downstream guard — inside the resident
  server process; the exponent spelling escaped pymatgen and detonated
  VPMDK's own expander instead. Both layers are now behind a raw-text bound.
- fix: bound the POSCAR ion-count line (1e7 total) before `Poscar.from_file`
  expands per-atom lists; a corrupted counts line (`2000000000`) used to
  allocate tens of GB inside the server before any validation ran.
- fix: apply the PSTRESS output correction in every run mode that prints
  stress, not only relaxations. A static or MD run with `PSTRESS` set wrote
  the raw stress with `Pullay stress = 0.00`, so the same INCAR flipped the
  reported pressure convention by the full `PSTRESS` when `IBRION`/`NSW`
  changed (measured: -2.59 kB static vs -502.59 kB relax for identical
  geometry at `PSTRESS=500`). NEB and force-constants outputs are corrected
  too.
- fix: accept trailing-comma scalar INCAR values again (`NSW = 3,` is legal
  Fortran list-directed input that VASP and pymatgen both read as 3). The
  corrupted-token guard introduced in this changelog rejected them — a
  regression against files that ran correctly before.
- fix: extend the corrupted-numeric-token rejection to the tags pymatgen
  leaves untyped. `CSVR_PERIOD = 5OO` came back as the string `'5oo'`,
  escaped the typed check, and VPMDK's own reader extracted 5 — a thermostat
  100x stiffer than requested with exit 0; `LANGEVIN_GAMMA = 1O` arrived as
  the list `[1]`, equally invisible. The rejection is now keyed on the raw
  token for every scalar numeric tag, regardless of which parser types it.
- fix: bound `|PSTRESS|` at 1e6 kBar. A huge-but-finite value (an exponent
  typo like `1e300`) overflowed the optimizer's step-length norm so every
  step scaled to zero: the run completed with exit 0 and a CONTCAR identical
  to the POSCAR — the requested pressure silently had no effect — while the
  OUTCAR pressure fields carried 300-digit values.
- fix: a SIGKILLed server left as an unreaped zombie no longer blocks
  restarts. The (pid, starttime) liveness identity matched the zombie's
  surviving `/proc` entry, so `vpmdk serve` refused with "its process holds
  the model" until the negligent supervisor reaped it; process state `Z` now
  reads as not-live.
- fix: case-fold the `DEVICE` tag at the BCAR parse. torch device strings are
  lowercase-only, while server mode compares `DEVICE` case-insensitively — so
  a VASP-style `DEVICE = CPU` was accepted and computed by the server (exit 0)
  but crashed the one-shot CLI on the byte-identical directory (exit 1).
- fix: reject overflow-range lattices. The minimum-cell-width guard computed
  `inf/inf = nan` for lattices in the 1e150+ Å range (finite entries, det
  overflows) and every nan comparison is False, so the guard silently disabled
  itself — a 5.43e200 scale factor completed with exit 0, nan stress and an
  inf volume. Widths above 1e6 Å (0.1 mm) are also rejected as input errors.
- fix: reject two ions occupying the same site (a duplicated POSCAR
  coordinate line, or fractional 0.0 and 1.0 coinciding under periodic
  boundaries) at input time. Every tested backend deterministically returns
  nan for a coincident pair, which was classified as a retryable calculation
  failure (exit 2) after a full model load.
- fix: reject a corrupted numeric INCAR token instead of letting the parser
  invent a number from its leading digits. `TEBEG = 5OO` (letter O typo for
  500) ran the MD at 5 K and `NSW = 0x10` ran a single point, both exit 0
  with the OUTCAR echoing the user's original text, where VASP refuses the
  file.
- fix: report PSTRESS the way VASP defines it. The OUTCAR `external pressure`
  line printed the RAW internal pressure with `Pullay stress = 0.00`
  hard-coded, so a run converged at `PSTRESS=500` read
  `external pressure = 500.00 kB` — the transpose of VASP's output, where all
  stress output is corrected by subtracting PSTRESS, the corrected pressure
  is ~0 at convergence, and the Pullay field echoes PSTRESS. The `in kB`,
  `Total`, and vasprun.xml stress outputs carry the same correction.
- fix: report a clean one-line diagnostic (exit 1) when `vpmdk run` is
  invoked from a working directory that no longer exists, instead of an
  uncaught FileNotFoundError traceback from `os.getcwd()`.
- fix: record the server process's kernel start time in `<socket>.pid` and
  identify a live draining server by (pid, starttime). The previous
  identification matched the socket path inside the process's cmdline, which
  a default-socket `vpmdk serve` never mentions — so the force-drain
  protection was silently inapplicable to the plain invocation and a second
  serve could still double-load the model.
- fix: fail loudly when a FAIRChem prediction carries no recognizable energy
  or forces instead of fabricating 0.0 eV / zero forces (a converged-looking
  OUTCAR with exit 0 whenever a fairchem release renamed its output keys), and
  omit stress from the calculator results when the model has no stress head
  (S2EF) instead of reporting exact-zero stress — an `ISIF=3` cell relaxation
  used to converge instantly against `external pressure = -0.00 kB` with no
  warning; it now fails with ASE's `PropertyNotImplementedError`.
- fix: bound `NHC_NCHAINS` (maximum 100). ASE integrates the Nose-Hoover
  chain in a Python loop whose cost and state arrays are O(chain length), so a
  finite-huge value either wedged a resident worker for weeks (~25 minutes per
  ionic step at 1e8) or died mid-run with a multi-TiB MemoryError classified
  as retryable. Chains beyond ~10 links have no physical effect.
- fix: reject a periodic cell narrower than 0.5 Å (a POSCAR scale-factor typo,
  or a collapsed CONTCAR reused as POSCAR) as an input error. Such cells
  passed the finiteness/singularity checks and then hung the backend's
  periodic-image enumeration indefinitely — a resident worker wedged with no
  output and no exit.
- fix: bound the CHGCAR grid's TOTAL point count (1e9), not just each axis.
  `NGXF=NGYF=NGZF=99999` was per-axis legal but an 8 PB grid: the whole
  calculation ran and then died at CHGCAR-write time as a "retryable"
  MemoryError.
- fix: classify a directory sitting where `CONTCAR`/`POSCAR` must be written
  as an input error (exit 1). ase.io.write's format inspection turned it into
  a TypeError, which the server reported as a retryable calculation failure
  after paying for the whole run — disagreeing with the same obstruction on
  `OUTCAR`.
- fix: foreground servers now write the `<socket>.pid` file too, and a second
  `vpmdk serve` refuses to replace an unresponsive socket whose pidfile names
  a live server. A foreground server draining an uninterruptible job after
  `stop --force` used to be classified as a stale socket: the second serve
  unlinked it and loaded a second resident model beside the draining one.
- fix: report the NEB tangent across a periodic boundary the way the optimizer
  saw it. The `OUTCAR` `TANGENT`/`CHAIN-FORCE`/`tangential force` blocks were
  built from raw coordinate differences between neighbouring images, so any
  band whose migrating atom crosses a cell face (the normal way fractional
  coordinates wrap) reported a tangent up to 116 degrees off with an inverted
  sign, while ASE's NEB engine relaxed the correct minimum-image band in the
  same run. The tangent now uses the minimum-image displacement.
- fix: answer exit 1 (input error), not exit 2 (documented retryable), when the
  submitted workdir itself cannot be written -- an `OUTCAR` directory in the
  way, a read-only tree. Retrying reproduces those byte-for-byte, so batch
  drivers looped forever; the one-shot CLI already failed with exit 1 on the
  same tree. Other `OSError`s (disk full, network filesystems) stay exit 2.
- fix: reject FFT grids no machine can hold instead of hanging or dying late.
  A finite-but-absurd `ENCUT` (`1e30`) sent the grid search into a multi-year
  spin that wedged a server worker permanently on one request, and an explicit
  `NGXF=1e12` only failed at CHGCAR-write time as a "retryable" MemoryError.
  Both now fail fast as input errors (exit 1) above 100000 points per axis.
- fix: keep the requested ensemble during a `TEBEG` -> `TEEND` temperature ramp.
  The ramp rescaled the velocities after every ionic step, which pinned the
  instantaneous temperature to the ramp line and silently replaced the requested
  Nose-Hoover / Langevin / Andersen / CSVR run with an isokinetic one (measured
  temperature spread 85.3 K -> 12.3 K, conserved-energy drift 0.0013 eV ->
  2.37 eV) while `OUTCAR`/`vasprun.xml` still reported the requested `MDALGO`.
  The ramp now retargets the thermostat; `MDALGO=0` (NVE) still rescales,
  because there is no thermostat to retarget.
- fix: also reject an INCAR tag swallowed by a blank tag whose type is not a
  string. `LWAVE =` above `TEBEG = 900` parses to `LWAVE=True` with `TEBEG` gone,
  so the MD ran at the 300 K default while `OUTCAR` still echoed `TEBEG = 900`.
- fix: reject an INCAR whose tag was never read. A tag written with an empty
  value (`SYSTEM =`) swallows the following line, because the parser's value
  pattern continues past the end of the line -- so `IBRION = 2` below it simply
  disappeared and the run did a single point instead of the requested
  relaxation, with exit 0 and `Calculation completed.`.
- fix: keep per-axis selective dynamics in `CONTCAR`. A `T T F` row was written
  back as `T T T` (ASE's POSCAR writer does not understand the constraint
  pymatgen builds for it), so the standard `cp CONTCAR POSCAR` continuation
  silently relaxed an axis the user had frozen.
- fix: report a Nose-Hoover damping time close to `POTIM` instead of letting it
  pin the temperature or diverge silently. `SMASS` is read as a damping time in
  femtoseconds -- not as VASP's Nose mass -- which is now documented and warned
  about when the coupling is stronger than `10 * POTIM`.
- docs: stop suggesting pre-seeded velocities as a way to decorrelate server
  replicas. A `POSCAR`/`CONTCAR` velocity block is never read; velocities are
  always re-drawn at `TEBEG`.
- fix: decide a POSCAR's VASP 4/5 format on the line the parser sees. pymatgen
  truncates every POSCAR line at `#`, so an ordinary comment on the ion-count
  line (`2   # number of Si atoms`) made VPMDK classify a VASP-4 file as VASP 5,
  skip every species guard, and compute the cell as hydrogen with exit 0 and a
  CONTCAR rewritten to match.
- fix: resolve backend capabilities from the RESIDENT's configuration in server
  mode. A request that inherits the backend tags instead of restating them (the
  documented batch pattern) was checked against the CHGNET default, so an
  energy-only resident answered exit 2 -- documented as retryable -- after a full
  calculation instead of exit 1 before it, and the missing-stress warning was
  emitted only for requests that spelled the tags out.
- fix: warn and fall back to `MDALGO=0` for an MD algorithm VPMDK does not
  implement. An out-of-range value silently ran velocity-Verlet NVE while OUTCAR
  and vasprun.xml still reported the requested thermostat; the reported value is
  now the algorithm that actually ran.
- fix: stop writing an all-zero force field when the backend supplies no forces.
  An energy-only selection (`MATRIS_TASK=e`) produced an OUTCAR whose TOTAL-FORCE
  table, total drift and `FORCES: max atom, RMS` were exactly 0.00000000 -- the
  value every convergence check reads as perfectly converged -- and exited 0. The
  configuration is now refused up front (exit 1), and a backend that fails to
  deliver forces mid-run reports it instead of fabricating them. A backend
  without stress (`MATRIS_TASK=ef`) still runs and now says that the stress block
  is omitted.
- fix: write the `vasprun.xml` `scstep` energy block in the default configuration
  too. Without `WRITE_PSEUDO_SCF=1`, `ase.io.read("vasprun.xml")` raised
  IndexError and pymatgen's `Vasprun.final_energy` returned `inf eV` after a
  single warning, so the tool's primary machine-readable output could not be
  consumed by either library.
- fix: apply the INCAR mangled-value guard to VASP's compact styles: several tags
  on one line separated by `;`, and values continued with a trailing `\`. Every
  tag written that way escaped the check, so `NSW = 1e5` and `EDIFFG = -1.0D-03`
  were silently read as `1` and `-1.0` again.
- fix: reject a VASP-4 style POSCAR whose ion groups do not match the POTCAR's
  species count instead of computing the cell as hydrogen. In a NEB image
  directory, where pymatgen cannot see the band's POTCAR one level up, a whole Cu
  band was computed as hydrogen and reported as success.
- fix: key the client's default-socket hardening on the socket's parent
  DIRECTORY, matching the server. Any sibling socket name in the predictable
  default directory (`--socket $XDG_RUNTIME_DIR/vpmdk-<uid>/gpu0.sock`) skipped
  the symlink, foreign-owner and world-writable checks on the client even though
  the server refuses to bind there.
- fix: reject a VASP-4 style POSCAR (no species line) when no POTCAR is available
  instead of computing it as hydrogen. Current pymatgen fabricates `['H', ...]`
  names and only warns on stderr, so the elements were silently wrong and CONTCAR
  was rewritten to match.
- fix: stop forwarding `PSTRESS` into the constant-volume relaxations (`ISIF=4`
  and `ISIF=5`), where an external hydrostatic pressure can do no work. ASE's
  filter left a traceless remainder that collapsed the cell by up to 42%; the tag
  is now reported as ignored for those modes.
- fix: reject Fortran `D` exponents in INCAR tags that pymatgen leaves untyped
  (`NHC_PERIOD = 1D2` ran the thermostat at 1 instead of 100).

- fix: report physical forces for cell-only relaxations (`ISIF=5/6/7`). The ion
  freeze VPMDK installs internally was applied to the REPORTED forces, so
  OUTCAR's TOTAL-FORCE table, `FORCES: max atom, RMS`, the total drift and
  vasprun.xml's forces varray were exactly 0.0 and any force check read the run
  as perfectly converged. A user's own selective dynamics still applies.
- fix: apply the POTCAR species order to a POSCAR again. `Poscar.site_symbols` is
  read-only in current pymatgen, so the reconciliation raised AttributeError and
  the run was rejected as invalid input instead.
- fix: reject INCAR values written with a Fortran `D` exponent (`NSW = 1D3`,
  `EDIFFG = -1.0D-03`), which pymatgen truncates to `1` and `-1.0`.
- fix: classify an unusable `SYMPREC` as an input error rather than a retryable
  calculation failure when spglib cannot analyse the structure at that tolerance.
- fix: ignore a UTF-8 BOM in BCAR, which made the first tag unreadable and
  silently substituted the default backend.

- fix: report the thermostat energy for Nose-Hoover chain and CSVR runs, so
  OSZICAR `SP=`/`SK=`, OUTCAR `nose potential`/`nose kinetic`, vasprun.xml
  `nosepot`/`nosekinetic` are no longer hard zeros and the reported total energy
  is the conserved quantity (measured drift 1.10 eV -> 0.0013 eV over 40 steps).
- fix: refuse to delete a live server's pidfile while it drains a force stop, so
  a second `serve` can no longer start a second resident model beside it.
- fix: reject a FIFO left at `<socket>.pid` with the intended message instead of
  an opaque "File or stream is not seekable" after the model has been loaded.
- fix: reject `POTIM=0` for thermostatted MD (and a negative `POTIM` for
  Langevin) as an input error rather than failing mid-run as a retryable
  calculation error; `MDALGO=0` is unchanged.
- fix: reject an integer INCAR tag written in scientific notation (`NSW = 1e5`),
  which pymatgen silently truncates to its leading digits, instead of running one
  ionic step and reporting success.
- docs: state that `CSVR_PERIOD` is read in fs and defaults to `100 * POTIM`, so
  writing the value explicitly is not the same as omitting it.

- fix: honor `EDIFFG > 0` energy convergence instead of stopping at ASE's default
  0.05 eV/A force limit. `Optimizer.irun` reassigns `fmax` from its own default,
  so the negative (deliberately disabled) force limit was discarded and
  relaxations terminated at a criterion the INCAR never asked for, writing an
  under-relaxed CONTCAR and exiting 0.
- fix: apply `PSTRESS` with `ISIF=6`, which `ase.filters.StrainFilter` cannot
  accept, by relaxing the cell through a frozen-ion `UnitCellFilter` when a
  non-zero pressure is requested; `PSTRESS=0` keeps the previous mapping.
- fix: write the LAMMPS dump `ITEM: BOX BOUNDS` line with LAMMPS' tilt
  convention (bounds are widened by the tilt, not narrowed), so readers no
  longer reconstruct a corrupted lattice and volume for non-orthogonal cells.
- fix: reject negative `TEBEG`/`TEEND` for Andersen/Langevin, non-positive values
  for CSVR, and a negative `LANGEVIN_GAMMA` as input errors, instead of letting
  ASE produce nan trajectories that fail later as retryable calculation errors.
  `TEBEG=0` stays valid wherever the thermostat supports it.
- fix: report a diverged calculation as such when an energy is non-finite,
  instead of failing with an unrelated "not enough values to unpack" error from
  the OSZICAR formatter.

- **fix (breaking output change): write stresses and pressures in VASP's
  convention.** `OUTCAR`'s `in kB` row, its `external pressure` value, its
  `Total` row (now `-sigma * V` in eV) and `vasprun.xml`'s `<varray
  name="stress">` (now kBar) previously carried ASE's tension-positive eV/A^3
  stress. Standard readers apply VASP's documented convention -- ASE's own
  parsers use `-stress * 0.1 * ase.units.GPa` -- so every consumer silently saw
  the stress sign-inverted, and from `vasprun.xml` also scaled by 1/1602.18. A
  compressed cell now reports a POSITIVE external pressure, as VASP does.
  Anything that adapted to the previous VPMDK output needs its sign (and, for
  `vasprun.xml`, its unit) assumption removed. `examples/*/reference/` outputs
  predate this fix and still show the old sign; they are illustrative only.
- fix: pin matgl's ASE calculator to `stress_unit="eV/A3"` so MatGL/M3GNet
  stresses and pressures are not scaled by ~160.2x under matgl >= 4, which
  defaults to GPa.
- fix: reject a non-finite (`nan`/`inf`) POSCAR lattice or atom position as an
  input error in flat workdirs as well as NEB image directories, instead of
  completing with a meaningless energy, zero forces and a `nan`-propagating
  CONTCAR.
- fix: reject `CSVR_PERIOD <= 0` before the run starts (like `NHC_PERIOD`)
  rather than failing part-way through as a retryable calculation error.
- fix: keep a resident server out of the idle-timeout path until the terminal
  event of a finished job has actually been delivered, so a client that stops
  reading cannot leave the process alive with its listener closed.
- fix: exit cleanly after the third shutdown signal abandons an in-flight
  calculation, instead of aborting during interpreter finalization and reporting
  the completed shutdown as a crash.

- feat: add a POSIX resident-calculator server with `serve`, `run`, `status`,
  and `stop` CLI commands; FIFO request isolation; BCAR configuration safety;
  daemon, timeout, and lifecycle support; and a synchronous Python client.
- perf: dispatch `run`, `status`, and `stop` through a standard-library-only
  client entrypoint so client processes do not import the ML runtime.
- fix: require an explicit `DEEPMD_TYPE_MAP` for resident DeepMD servers instead
  of retaining a type ordering inferred from the startup POSCAR.
- fix: make a second shutdown signal stop the resident worker immediately so
  queued calculations cannot start during forced teardown.
- fix: compare `MATGL`/`M3GNET` and `FAIRCHEM`/`FAIRCHEM_V2`/`ESEN` as
  equivalent resident backend names when their construction settings match.
- fix: split oversized server log events and truncate oversized remote failure
  details before transmission without changing their calculation exit code.
- fix: preserve NEB compatibility with older ASE releases by omitting the new
  keyword in one-shot mode and using per-image delegates to one serial resident
  calculator in server mode.
- docs: add resident-server adoption guidance, complete CLI and client
  contracts, GPU/scheduler patterns, troubleshooting, security notes, a
  runnable batch example, and real CPU/CUDA validation records.
- docs: document that server mode replays the startup RNG state for every
  request, so identical stochastic (MD/thermostat) requests reproduce one
  trajectory instead of sampling independently as separate one-shot processes
  do, and describe how to collect an ensemble.
- fix: report the device a blank `DEVICE=` actually selects for MatGL/M3GNet and
  MatRIS residents, so `status` cannot advertise a GPU the model was never moved
  onto and requests naming the real device are no longer rejected.
- fix: recognize MatGL 2.x `PESCalculator`, load the default MatGL potential
  when that calculator requires one, and preserve the original model-load
  exception when a local model cannot be deserialized.
- fix: compare EquiformerV3 registration-module aliases after applying the
  builder's order-preserving deduplication.
- fix: reject missing path-like `MODEL` values before resident calculator
  construction so status cannot advertise a checkpoint that silently fell
  back to a backend default.
- fix: append `.pid` to the complete socket path and protect daemon pidfiles
  with socket ownership metadata, preserving unrelated files during startup
  and cleanup.
- fix: validate extensionless `MODEL` values for local-only resident backends,
  while preserving slash-containing FAIRChem model identifiers across startup
  and request directories.
- fix: report no MatGL named default when the resident calculator uses legacy
  M3GNet's own bundled default, including in public backend metadata.
- fix: preserve MatGL's named default model identity across server startup and
  request directories instead of normalizing it as a local checkpoint path.
- fix: derive the effective GRACE foundation model from the installed registry
  so server status, request validation, and public metadata match the calculator.
- test: allow the real GRACE integration smoke to use a named foundation model
  and configurable CPU/CUDA device instead of requiring PyTorch CUDA.
- fix: order concurrent run submissions by socket acceptance sequence and
  reassemble oversized log-event continuations before printing CLI output.
- fix: transmit the run client's cwd for relative charge-environment paths,
  preserve remote input failures as exit code 1, and count request-size limits
  against the JSON body rather than its NDJSON newline.
- fix: retain GRACE's documented default metadata without an installed registry,
  restore direct MatGL checkpoint construction after loader or calculator
  incompatibility, retry default potentials without unsupported device keywords,
  and support resident NEB on ASE releases without `allow_shared_calculator`.
- fix: load an explicitly selected MatGL registry model instead of substituting
  the default potential, and retain a successfully loaded potential when
  retrying older calculator signatures.
- fix: reject missing or unreadable explicit legacy M3GNet checkpoints and
  missing explicit MACE paths instead of silently constructing their defaults;
  reject empty results returned by MatGL model loaders.
- fix: centralize MODEL classification across every backend, server identity,
  and request check; distinguish omitted defaults, existing local checkpoints,
  and named models; reject unsupported explicit selectors and empty upstream
  loader results instead of silently substituting a default potential.
- fix: preserve loader-facing symlink paths while canonicalizing only resident
  comparison identity, and retain upstream-resolved selector compatibility for
  Nequix builds without registry metadata, opaque Matlantis versions, and
  FAIRChem v1/OCP checkpoint identifiers.
- fix: replace per-backend MODEL resolver branches with an exhaustive 26-backend
  capability matrix; forward explicit MatterSim selectors to `from_checkpoint`,
  delegate path-shaped FAIRChem v2/ESEN identifiers, and restore GRACE's warning
  plus effective-default behavior without misreporting the loaded model.
- fix: distinguish MatterSim preset names from missing checkpoint paths, detect
  legacy `load_path` support before forwarding presets, share GRACE default and
  name resolution through one policy resolver, forward dynamic simple-backend
  presets without silent fallback, and resolve EquiformerV3 MODEL exactly once.
- fix: parse unit-bearing numeric strings returned inside pymatgen singleton
  lists using the same rules as scalar INCAR strings.
- perf: share backend default-model resolution, parse each request BCAR once,
  canonicalize server startup configuration once, construct backend identity once
  after device detection, and share wrapper/resolved calculator discovery.
- fix: classify relative server `run.workdir` values as protocol errors rather
  than remote calculation failures.
- fix: recheck force-stop immediately after queue removal and serialize signal
  publication with job claiming so dequeued-but-unstarted work cannot execute.
