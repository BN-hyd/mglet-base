# Scalar Robin validation record

This file distinguishes executable acceptance checks, results observed during
the current development session, and limitations of MGLET's existing scalar
discretization. A successful kernel or planar check does not establish curved
surface accuracy or conservation of the physical cut-cell mass.

## Starting point and reproducibility

The continuation started from development branch
`feature/scalar-robin-surface-reaction`, commit
`4216cba278ef54bd8587f7eee71a762e8e469372`. The reported previous CI run is
[38147950453](https://github.com/BN-hyd/mglet-base/actions/runs/38147950453).
According to the supplied handoff, its build, kernel, planar, limits,
isolation, legacy, transient, MPI/restart, multilevel, invalid-input, and
related legacy regression checks passed. Its sphere suite stopped at
`N=16` with `Robin: invalid/unresolved patch 9 7 4`. These are inherited
results, not new validation of a subsequent change.

Build and run the checks in the repository root:

```sh
cmake -S . -B build -GNinja --preset=gnu-debug -DMGLET_REAL64=ON
cmake --build build -j2
python3 tests/ScalarRobin/run.py --solver "$PWD/build/src/mglet" \
  --suite sphere --work /tmp/robin-check
python3 tests/ScalarRobin/run.py --solver "$PWD/build/src/mglet" \
  --suite sphere-time --work /tmp/robin-check
python3 tests/ScalarRobin/run.py --solver "$PWD/build/src/mglet" \
  --suite sphere-steady --work /tmp/robin-check
python3 tests/ScalarRobin/run.py --solver "$PWD/build/src/mglet" \
  --suite sphere-mpi --work /tmp/robin-check
python3 tests/ScalarRobin/run.py --solver "$PWD/build/src/mglet" \
  --suite all --work /tmp/robin-all
ctest --test-dir build/tests -R ScalarRobinKernel --output-on-failure
ctest --test-dir build/tests \
  -R 'HotAndColdSpheres|Parker_micro|ScalarDecay' \
  --output-on-failure --timeout 180
```

The driver requires Python 3, NumPy, and h5py. MPI cases require `mpirun`.
Each run retains its generated JSON, STL, HDF5 grid/fields, and `solver.log`
under the requested work directory. Successful runs also write
`validation.json` with rates, budget summaries, maximum step residuals,
and independent HDF5 point-mass/volume values. The curved-surface time and
relaxation suites write additional root-level summary JSON files.
Running OpenMPI as root additionally
requires `OMPI_ALLOW_RUN_AS_ROOT=1` and
`OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1`. Tests fail on missing dependencies,
solver failure, nonfinite results, or violated assertions.

## Executable acceptance checks

The following criteria are implemented in `tests/ScalarRobin/run.py` and
`tests/robin_wall_test.F90`; they are not assertions that this session has
executed or passed them. The continuation adds material-property rejection,
unsafe-timestep rejection, VOLP diagnostics, and the separate curved-surface
time-refinement, relaxation, and MPI suites. `--suite all` includes all
three new sphere suites.

| Check | Reference and acceptance criterion |
| --- | --- |
| Kernel | Production Fortran wall law satisfies diffusion and reaction identities, zero rate, equilibrium, reverse flux, strong-rate limit, and a nonuniform multipoint linear profile within `100*epsilon` absolute error. |
| Planar | `N=12,24,48`; physical-center maximum concentration error below `0.003` and total surface-rate error below `0.003*0.16`. |
| Planar relaxation | Constant initial concentration, `N=12`, final time `5`; profile and rate errors below `0.004` and `0.004*0.16`. |
| Limits | `k_s=0`, `k_s=1e12`, `C_0=C_eq`, and reverse flux; the same planar profile/rate criteria. |
| Scalar isolation | Two scalars with different `D`, `k_s`, and `C_eq`; compare with isolated runs at identical timesteps within `2e-12` maximum concentration difference. |
| Legacy boundaries | Existing immersed `value` and `flux` conditions; physical-center maximum profile error below `0.025`. |
| Time refinement | Nonsteady planar case at fixed mesh and final time `0.05`; three timestep scales `1,1/2,1/4`, successive field differences satisfy `e_2<0.3*e_1`. |
| Restart | Two portions of the transient case compared with uninterrupted integration within `2e-11`. |
| MPI | Two connected grid blocks in serial and two MPI ranks; maximum concentration difference below `2e-11`. |
| Sphere | Finite shell with two Robin surfaces at `N=16,24,32`; each absorption and release error below `25%`; finest absorption error below `10%` and below the coarsest error. `N=16` must exercise corner-patch transfers and `N=32` must exercise tiny-patch normal rescue. |
| Sphere time refinement | Fixed `N=32`, identical STL, final time `0.15`, and timestep scales `1,1/2,1/4`; end-field physical-center RMS differences satisfy `e_2<0.3*e_1` or both lie below `1e-11` roundoff-scale error, and successive last-stage rate differences remain below `0.1% Q`. The original uptake/release accuracy limits also apply. |
| Sphere relaxation | Fixed `N=32` and timestep, continued to `t=0.6` and `t=1.2`; the final uptake/release change and supply/loss imbalance are each below `1% Q`. Final uptake/release retain the original finest-grid accuracy limits. |
| Sphere MPI and partition | `N=16`, two connected y-blocks; serial/two-rank fields agree within `2e-11` maximum difference. Single-block and split-block rates separately retain the original `25%` coarse-grid criteria. Their rate difference is recorded as a spatial partition effect, not required to be bitwise identical. |
| Multilevel | Existing 36-block, three-level grid in serial and two MPI ranks within `2e-11`; equilibrium concentration/flux preservation and positive nonequilibrium uptake. |
| Invalid inputs | Negative `k_s`, nonpositive `prmol`, nonpositive `gmol`/`rho` individually and together, and an unsafe strong-rate timestep must terminate with the expected diagnostic. Nonstandard JSON `NaN`/`Infinity` for `c_eq` are rejected by the JSON parser. |
| Discrete budget | Every logged RK step satisfies `abs(residual)<2e-12*max(1,abs(mass))`; sphere and multilevel cases also require zero external Cartesian-boundary supply at that tolerance. |
| VOLP point diagnostics | Every Robin step prints finite values, identical RK flux amounts, and the BT/VOLP storage identity within the budget tolerance. On single-level grids, final HDF5 volume-weighted point mass must independently match the log within `2e-12*max(1,abs(mass))`. Nonzero physical-storage residuals are recorded, not forced to zero. |

These tests compare scalar values at physical fluid centers; solid-side
ghost values are excluded from pointwise physical accuracy checks. They
are retained in the operator's discrete mass whenever `BT` activates them.
The RK rate printed by the solver is the final RHS evaluation's rate;
the budget amounts use the complete RK recurrence and are step integrals.

## Finite-shell analytic reference

The spherical test has an inner solid of radius `a=0.5` and an enclosing
solid outside radius `b=1.4`. Their Robin parameters are respectively
`k_a=1`, `C_eq,a=0` and `k_b=2`, `C_eq,b=1`, with `D=1`. The enclosing
surface supplies scalar through reverse Robin flux. All Cartesian box
faces lie outside the physical fluid shell, so they do not supply scalar.
The reference includes both surface transfer resistances:

\[
Q=\frac{4\pi(C_{eq,b}-C_{eq,a})}
{1/(k_a a^2)+(1/a-1/b)/D+1/(k_b b^2)}
=2.267963757287659.
\]

The inner outward loss is `+Q`; the outer outward loss is `-Q`. The steady
profile is `C(r)=A-B/r`, where

\[
B=\frac{Q}{4\pi D}=0.18047882136279927,\qquad
A=C_{eq,a}+\frac{Q}{4\pi k_a a^2}+\frac{B}{a}
=1.0828729281767957.
\]

The exact physical scalar content is therefore

\[
M_{physical}=4\pi\left[\frac{A(b^3-a^3)}{3}
-\frac{B(b^2-a^2)}{2}\right]=9.94048514819181.
\]

The initial condition is this continuum steady profile and the original
test advances to `t=0.15`. This checks the perturbed profile and rate after
evolution; it does not by itself prove convergence to the discrete steady
state. At the starting commit, the `N=16` STL uses three octahedral
subdivisions and `N=24,32` use four. Consequently the measured refinement
contains both Cartesian and surface triangulation errors. The continuation
uses four octahedral subdivisions for every grid, holding both STL surfaces
fixed to isolate Cartesian refinement. The time-refinement suite holds the
finest grid and surfaces fixed. The relaxation suite continues that solution
at a fixed timestep instead of treating initialization as proof of a
discrete steady state.

## Storage and conservation interpretation

`fluxbalance_grid` divides flux divergence by full Cartesian volume
`DDX*DDY*DDZ`; `blockbt_grid` activates neighboring solid-center unknowns.
Thus the existing transport operator conserves the leaf-cell quantity

\[
M_{BT}=\sum_i FINECELL_i\,BT_i\,\Delta V_i C_i,
\]

and Robin adds each stored GC patch's `-A*J_out/DeltaV` to exactly one
active owner. `ROBIN_BUDGET` tests this quantity against RK-integrated
surface loss and the pre-Robin RHS contribution.

`gc_stencils_mod:calc_volp_grid` instead calculates `VOLP` as an absolute
physical fluid volume in units of length cubed. `VOLP` is not a fraction,
and must not be multiplied by another Cartesian volume. A physical
storage diagnostic is

\[
M_{VOLP,point}=\sum_i FINECELL_i\,VOLP_i\,C_{point,i}.
\]

Here `C_point` is the transported cell-center point value, not a cut-cell
volume average. This is a point-value approximation to physical scalar
content, not an exact physical mass even with an exact geometric volume.
For ghost-cell runs, `robin_budget` also prints:

```text
ROBIN_VOLP_POINT_BUDGET scalar step end_time mass delta_mass supply uptake residual
```

Its supply and uptake amounts are the same RK-integrated amounts used by
the `BT` transport budget. Its residual is intentionally not asserted to
vanish. At each step the exact diagnostic relation is

\[
R_{VOLP,point}=\Delta(M_{VOLP,point}-M_{BT})+R_{BT}.
\]

The nonzero residual therefore exposes the change in storage introduced
by the different volume weights. Independently read `VOLP` and the scalar
from the final `fields.h5`, excluding grid halos, to verify the reported
point-value mass on a single level. All its interior cells are leaves,
including when split among several blocks. `FINECELL` is not written to
this HDF5 output, so the driver does not guess covered coarse cells for a
multilevel independent mass check. That path retains only the solver's
globally reduced leaf-cell diagnostics. Also
compare `sum(FINECELL*VOLP)` with `4*pi*(b**3-a**3)/3` for the shell.
This separates geometric volume error from concentration integration and
transport errors.

Budget closure for `M_BT` does not imply closure for `M_VOLP,point`. Changing
only the budget weight would neither change the scalar RHS denominator
nor turn point values into cut-cell averages. Conservation of physical
cut-volume storage remains an unfulfilled strict acceptance requirement
for this branch. Approximate physical-storage errors can be assessed by
refinement, but an exact cut-volume guarantee needs a separately extended
and validated transport discretization.

## Results of the current continuation

The local platform is WSL2, Linux `5.15.167.4-microsoft-standard-WSL2`.
Observed tools are GNU C/C++/Fortran `13.3.0`, CMake `3.28.3`, Ninja
`1.11.1`, and OpenMPI `4.1.6`. The Python environment under
`/tmp/robin-deps/venv` provides Python `3.12.3`, NumPy `2.5.4`, and h5py
`3.16.0` linked to HDF5 `2.0.0`. The solver uses OpenMPI-enabled HDF5
`1.10.10`, extracted with the other missing development dependencies under
`/tmp/robin-deps/root`; the Python and solver HDF5 libraries are separate.
The GNU debug preset enables `-fcheck=all` and floating-point traps for
invalid operations, zero division, and overflow; real64 is enabled.

The successful local build used these dependency locations:

```sh
export PATH=/tmp/robin-deps/root/usr/bin:/tmp/robin-deps/venv/bin:$PATH
export LD_LIBRARY_PATH=/tmp/robin-deps/root/usr/lib/x86_64-linux-gnu
export LIBRARY_PATH=/tmp/robin-deps/root/usr/lib/x86_64-linux-gnu
cmake -S . -B build -GNinja --preset=gnu-debug -DMGLET_REAL64=ON \
  -DCMAKE_PREFIX_PATH=/tmp/robin-deps/root/usr \
  -DHDF5_C_COMPILER_EXECUTABLE=/tmp/robin-deps/root/usr/bin/h5pcc \
  -DHDF5_Fortran_COMPILER_EXECUTABLE=/tmp/robin-deps/root/usr/bin/h5pfc
cmake --build build -j2
```

This is the equivalent concise command for a fresh cache. The actual
recovery configuration additionally used `-U 'HDF5_*'` followed by the
two explicit HDF5 wrapper settings, and a separate retry with
`-U 'MPI_*'` after allowing OpenMPI's required local sockets. Debug build
completed with exit status zero. Dependencies were extracted in `/tmp`;
no system installation or source-level build changes were needed.
After validation, the user's unmodified configure command
`cmake -S . -B build -GNinja --preset=gnu-debug -DMGLET_REAL64=ON` also
returned zero using that populated dependency cache, followed by
`cmake --build build -j2` returning zero with `ninja: no work to do`.
This final cache check did not alter the tested solver source or binary;
its configure log is `/tmp/robin-configure-final.log`.

The new invalid-input suite was executed with exit status zero:

```sh
PATH=/tmp/robin-deps/venv/bin:/tmp/robin-deps/root/usr/bin:$PATH \
LD_LIBRARY_PATH=/tmp/robin-deps/root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-} \
python3 tests/ScalarRobin/run.py --solver "$PWD/build/src/mglet" \
  --suite invalid --work /tmp/robin-invalid
```

All 11 expected failures produced the required diagnostics: negative
`k_s`, zero/negative `prmol`, zero/negative viscosity and density, two
negative material properties together, nonstandard JSON `c_eq` values
`NaN`/`Infinity`, and the unsafe strong-rate timestep. Nonpositive
material properties are rejected by the existing `init_flowcore` before
Robin scalar initialization; the tests therefore check its viscosity and
density messages. Nonfinite JSON tokens are rejected by the parser before
the Fortran `c_eq` finite-value check. This does not claim direct runtime
coverage of those unreachable Fortran guards.

The fixed-STL sphere suite was then executed under the same dependency
environment with
`python3 tests/ScalarRobin/run.py --solver "$PWD/build/src/mglet" --suite sphere --work /tmp/robin-check-fixed`
and returned zero. Actual last-stage rates at `t=0.15` were:

| N | Inner uptake | Outer outward rate | Uptake error | Supply error | Corner transfers | Tiny-normal rescues |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 2.112592994663 | -2.325718553756 | 6.85067% | 2.54655% | 56 | 0 |
| 24 | 2.174533976319 | -2.279091392813 | 4.11954% | 0.490644% | 344 | 0 |
| 32 | 2.212725540182 | -2.277201746171 | 2.43559% | 0.407325% | 616 | 21 |

The uptake error decreases across all three grids and meets the original
finest-grid `10%` threshold without changing any rate-error tolerance.
The `N=32` geometry also contains three exactly zero-measure GC patches,
reported separately. Every positive-area patch is retained, including the
21 tiny patches whose existing GC normals were zeroed by the GC area
threshold. These rates alone do not establish discrete steadiness; the
additional relaxation check supplies that evidence.

The independently read final masses and maximum per-step residuals were:

| N | BT mass | VOLP point mass | Maximum BT residual | Maximum VOLP point residual | Fluid volume |
| --- | --- | --- | --- | --- | --- |
| 16 | 12.82437857555 | 9.824944845448 | 7.10e-14 | 7.72139e-3 | 10.83212843000 |
| 24 | 11.83861635080 | 9.856925921864 | 1.59e-13 | 1.19778e-2 | 10.88373657606 |
| 32 | 11.32260465884 | 9.866706476280 | 1.90e-13 | 3.15671e-3 | 10.88865859744 |

The continuum shell volume is `10.97044154633556`; the reported fluid
volume errors are respectively `1.26078%`, `0.790351%`, and `0.745485%`.
Against the steady continuum scalar content above, the final VOLP point
mass errors are `1.16232%`, `0.840595%`, and `0.742204%`. Since `t=0.15`
has remaining transient rate imbalance, these point-mass errors contain
geometry, point-value integration, transport, and relaxation contributions.
The maximum VOLP residual is nonzero and nonmonotonic across these grids;
it must not be presented as a closed physical mass budget.

To check the complete interval, budget records were deduplicated by
`(scalar, step)` from each `solver.log` and summed over all steps. The
initial point mass is the first record's `mass-delta_mass`; the total
change below is the final mass minus that initial mass. External
Cartesian-boundary supply integrates to less than `1.1e-17` in magnitude.
Net outward surface amount includes both inner absorption and outer
Robin release, so its negative sign denotes net scalar supply.

| N | Steps | Initial VOLP point mass | Total point-mass change | Integrated net outward surface amount | Cumulative VOLP residual | Residual / initial point mass |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 375 | 9.798515389361 | +0.0264294560872 | -0.0373224890540 | -0.0108930329668 | -0.111170% |
| 24 | 844 | 9.852053612321 | +0.00487230954369 | -0.0178133931365 | -0.0129410835928 | -0.131354% |
| 32 | 1500 | 9.859484837019 | +0.00722163926149 | -0.0110134360249 | -0.00379179676340 | -0.0384584% |

Endpoint differences independently reproduce the accumulated VOLP
residuals. In comparison the accumulated BT residuals are
`-5.62e-14`, `-5.47e-14`, and `+1.79e-13`. The nonzero cumulative
physical-storage approximation residual, including its nonmonotonic
coarse-to-intermediate change, remains a substantive conservation
limitation; timestep refinement cannot remove the difference between
these two storage operators.

The curved-surface time-refinement suite passed on `N=32`. Its physical
end-field RMS differences are `4.39769774092e-14` and
`5.49293178012e-15`; the smaller timestep changes the final field at
near-roundoff magnitude. The maximum successive last-stage rate difference
is `1.43956573639e-6`, or `0.000063474% Q`, well below the `0.1% Q`
criterion. The small field differences support timestep independence at
this resolution; they do not provide a robust measured temporal order.

The finest-grid relaxation suite passed at a fixed `dt=0.0001`:

| End time | Inner uptake | Outer outward rate |
| --- | --- | --- |
| 0.15 | 2.212725540182 | -2.277201746171 |
| 0.6 | 2.218408590875 | -2.235812321110 |
| 1.2 | 2.219827871013 | -2.222645427220 |

The maximum final rate increment is `0.0131668938898`, or `0.580560% Q`;
the final supply/loss imbalance is `0.00281755620661`, or `0.124233% Q`.
Both meet the `1% Q` relaxation criteria. Final uptake and release errors
are `2.12243%` and `1.99819%`, preserving the original accuracy limits.
The end-time uptake should therefore be read as steady to the specified
increment/imbalance tolerance, not exact equality of surface rates. Its
final independently integrated VOLP point mass is `9.88605786701003`.

The spherical MPI/partition suite also passed. The `N=16` two-y-block
serial and two-rank runs have zero maximum field difference at stored
precision. Serial split-block rates are `2.11259299466241` and
`-2.32571855362147`. Their split-minus-single-block changes are
`-4.96048e-13` and `+1.34582e-10`; these are negligible here, while the
documented local receiver restriction still means partition invariance
cannot be assumed for every geometry.

The final complete Robin CTest run used the dependency environment above:

```sh
ctest --test-dir build/tests -R ScalarRobin --verbose
```

It returned zero, `2/2` CTest tests passed in `253.16 s`: the production
Fortran kernel (`0.00 s`) and integration driver (`253.15 s`). All 11
driver suites passed: planar, limits, isolation, legacy, transient,
sphere, sphere-time, sphere-steady, sphere-mpi, multilevel, and invalid.
This includes planar and curved timestep refinement, the declared rate
limits, current-RK-stage transients, discrete global budgets, independent
single-level HDF5 mass checks, scalar isolation, restart, two-rank MPI,
and the existing three-level mesh. Across the final successful run
reports, the largest absolute per-step BT residual is `2.65134e-13`.
The multilevel equilibrium surface rate is `1.31435e-17`; its
nonequilibrium rate is `0.102076836094`, with serial/MPI comparisons
passing their prescribed tolerance.

The final complete log is `/tmp/robin-ctest.log`; individual solver logs,
validation summaries, and the time/relaxation/MPI summary JSON files are
under `build/tests/ScalarRobin/runs`. These are actual local results, not
an inference from the inherited CI report.

Related legacy regressions passed with exit status zero, `3/3` tests:

```sh
PATH=/tmp/robin-deps/venv/bin:/tmp/robin-deps/root/usr/bin:$PATH \
LD_LIBRARY_PATH=/tmp/robin-deps/root/usr/lib/x86_64-linux-gnu \
ctest --test-dir build/tests -R 'HotAndColdSpheres|Parker_micro|ScalarDecay' \
  --output-on-failure --timeout 180
```

`HotAndColdSpheres` took `73.60 s`, `Parker_micro` `123.16 s`, and
`ScalarDecay` `73.70 s` (`270.47 s` total). Logs are retained at
`/tmp/robin-old-regressions.log` and under `build/tests`. All 20
`ScalarDecay` solver cases completed and its NumPy/h5py analysis ran.
Its legacy script has no concentration-error assertion; missing optional
Matplotlib skipped plotting. Thus this is a successful existing execution
regression, not additional quantitative Robin physics acceptance.

The updated workflow preserves solver logs and validation JSON summaries.
No result of a new remote CI run for the final continuation is claimed in
this record; the inherited remote CI result applies only to its original
commit.

## Remaining gates and limits

- Strict conservation of physical cut-volume scalar storage is **not
  fulfilled**. The existing transport conserves `BT*Cartesian` point
  storage; the independently evaluated VOLP point approximation has the
  nonzero per-step and cumulative residuals reported above. Exact
  cut-volume transport would require a separately validated extension.
- The corner fallback receiver is local to the owned grid and must have
  the same leaf membership. Narrow gaps, a needed receiver beyond a block
  edge, or several surfaces inside one cell can still fail explicitly.
  The new sphere partition check covers one two-block geometry, not all
  partitions or narrow-gap configurations. Dedicated failure/accuracy
  tests for these combined degeneracies have not been run.
- Multilevel checks cover a planar reactive surface, serial/MPI agreement,
  equilibrium preservation, and the discrete budget. Curved-surface
  fallback transfers across refinement boundaries and multilevel physical
  accuracy have not been independently validated. The HDF5 point-mass
  cross-check is single-level because `FINECELL` is not written there.
- Mixed Robin and legacy fixed-concentration ghost overwrites can expose
  additional nonconservative changes in the budget. Separate old-boundary
  regressions do not validate every mixed-boundary configuration.
- The explicit row-norm timestep screen and the completed quiescent DNS
  cases do not prove monotonicity or stability for arbitrary advection,
  LES, or unresolved geometry. The strong-rate limit, unsafe-step rejection,
  and timestep-refinement checks do not clip concentrations.
- New remote CI for the final pushed continuation has no recorded result
  here. The supplied earlier run cannot substitute for that result.
