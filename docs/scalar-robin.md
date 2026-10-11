# Passive scalar: finite-rate surface reaction

## Model and conventions

The new geometry condition is `type: "robin"`. It implements

\[
\partial_t C + \nabla\cdot(\boldsymbol{u} C)=D\nabla^2 C,
\qquad J_{out}=-D\partial_n C=k_s(C_w-C_{eq}).
\]

The specified normal `n` points **out of the fluid into the solid**.
Positive `J_out` removes scalar. Negative flux supplies scalar when
`C_w < C_eq`; neither flux nor concentration is clipped.
MGLET's `gc%nvecs(1:3,:)` points the other way, into fluid. Its plane is
`s = nvecs(1)*x + nvecs(2)*y + nvecs(3)*z - nvecs(4)`; fluid has `s > 0`.
Thus `J_out = D*dC/ds`.

This implementation uses constant isotropic **molecular** diffusivity
`D = flow.gmol / (flow.rho * scalar.prmol)`.
`gmol` is dynamic viscosity, `rho` density, and `prmol` the scalar molecular
Prandtl/Schmidt number. No independent `D` key is added.
`k_s` has units of length/time and must be finite and nonnegative;
`c_eq` has the scalar's units, is finite, and defaults to zero.
`gmol`, `rho`, `prmol`, and the resulting `D` must be finite and positive.
Material properties are checked before evaluating the diffusivity ratio;
two negative properties cannot pass by producing a positive ratio.
Use consistent dimensional or nondimensional units throughout an input.
For nondimensional inputs, scale `k_s` by the reference velocity.

The scalar remains passive. Configuring the same scalar as the existing
buoyancy source is rejected. No volume reaction, particle tracking, geometry
motion, or changes to fluid properties are introduced. Wall velocity must be
zero on reactive geometries. The existing LES bulk diffusivity, if enabled,
is unchanged; the surface law always uses molecular `D`. The verification
cases use DNS (`lesmodel.model: "none"`).

## Input and runnable example

Geometry keys are the **exact STL names** already used by the ghost-cell
input, including filename extensions. Each scalar has its own parameters:

```json
"scalar": {
  "solve": true,
  "scalars": [
    {
      "name": "C",
      "prmol": 1.0,
      "initial_value": 1.0,
      "robin_budget": true,
      "geometries": {
        "sphere.stl": {"type": "robin", "k_s": 0.1, "c_eq": 0.0}
      }
    }
  ]
}
```

`k_s` is required for Robin; `value` belongs to the old `value` and `flux`
conditions and is not a Robin parameter. Existing inputs and defaults are
unchanged (unspecified geometry condition = zero flux). Old geometry `flux`
is positive **into** the fluid, opposite to the new `J_out` convention.
Robin currently applies to immersed geometries, not to Cartesian outer
boundary `SWA` records in `grids.h5`.

Create a complete planar case (JSON, HDF5 grid, and STL) without a separate
grid generator:

```sh
python3 tests/ScalarRobin/run.py --generate /tmp/robin-example
cd /tmp/robin-example
/absolute/path/to/build/src/mglet parameters.json
```

The generator requires Python 3, NumPy, and h5py. It writes a quiescent fluid
with a wall at `x=0`, reservoir at `x=1`, and `D=1`, `k_s=1`, `c_eq=0.2`.
Inspect or edit the generated JSON to define multiple scalars/geometries.

## Reconstruction

`gc_robin_mod` reuses `wmindexlistn`, `wmcheckneighbor`, and `choosestencil`,
and the existing `nvecs`, `bodyid`, and `arealist`. There is no sphere-specific
normal calculation. Positive fluid-side distances are computed from each
patch's existing local plane. As in the old scalar reconstruction, points
closer than `0.2*abs(s_owner)` are excluded. A BP-open intersected owner is
also included if its distance is positive. Geometry-only weights are stored;
no scalar diffusivity or reaction parameter is cached in shared templates.

Use the same normal linear ansatz as `calcfluxtsca`,
`C(s) = C_w + g*s`, with the least-squares normal slope at fixed `C_w`:

\[
g=\frac{\sum_i d_i(C_i-C_w)}{\sum_i d_i^2},\quad
\bar C=\frac{\sum_i d_i C_i}{\sum_i d_i},\quad
\ell=\frac{\sum_i d_i^2}{\sum_i d_i}.
\]

Combining `D*g=k_s*(C_w-C_eq)` gives

\[
C_w=\frac{D\bar C+k_s\ell C_{eq}}{D+k_s\ell},\qquad
J_{out}=\frac{Dk_s}{D+k_s\ell}(\bar C-C_{eq}).
\]

This is the existing multi-observation normal fit with a Robin constraint,
not an assignment of the adjacent cell value to the wall. Only for one
observation is `ell=delta` and the usual one-point formula recovered.
`robin_wall` evaluates bounded ratios, avoiding overflow in `k_s*ell`.
For `k_s=0`, the flux is exactly zero. As `k_s` grows, `C_w -> C_eq` and
the conductance saturates at `D/ell`. Its implementation does not subtract
two nearly identical wall concentrations to evaluate the strong-rate flux.

The approximation neglects tangential variation within a patch and wall
curvature over the stencil, as does the existing normal reconstruction.
Do not assume second-order global accuracy on a curved surface. A resolved
STL surface and spatial refinement are required.

If the ordinary selection fails, a nearest eligible fluid-side BP-open
point from the existing search list provides a reported one-point fallback.
A corner-only GC intersection can have a valid wall plane and positive
area although its center has `BT=0`: the Cartesian activation mask sees
face neighbors, while the immersed patch can meet fluid diagonally. Such
a patch retains its existing GC area, plane, and body. Its flux is applied
to an eligible owned, active fluid observation with the same `FINECELL`
membership, using that receiving cell's Cartesian volume. Transfers and
one-point fallbacks are counted in startup diagnostics. Multiple patches
may contribute to one receiver; their RHS contributions are accumulated.

The existing GC geometry deliberately sets its normal to zero when a
patch's area divided by a representative Cartesian face area is below
`maccur**2`. For this specific tiny-area degeneracy, Robin borrows the
nearest valid unit GC normal from the same body in the existing search
neighborhood, and anchors its plane at the original patch's `gc%xpsw`.
It retains the full positive area and reports normal-rescue counts. No
sphere-center geometry is constructed. Missing valid neighbors still
cause a fatal error. Exactly zero-measure GC patches are counted separately;
they contribute zero absorption by definition. This local normal rescue
remains an approximation requiring spatial refinement.

No valid observation or receiver, invalid normal/area, or unresolved body
identity causes a fatal error rather than silent zero absorption. The
receiver search is limited to the existing local GC search list and the
owned grid interior. Extremely narrow gaps and patches requiring a
receiver across a block edge can therefore fail explicitly; MPI and grid
partition tests must cover the geometry used by an application. Existing
geometry processing assigns one body to each patch; multiple contacting
surfaces inside one cell must be resolved by refinement.

## Conserved quantity and RK integration

**This extends MGLET's existing conservative scalar flux discretization;
it is not a new cut-cell discretization.** `tstsca4_grid` treats stored scalar
values as Cartesian cell-center values. `fluxbalance_grid` divides the net
face flux by the **full** Cartesian volume `DDX*DDY*DDZ`. `BT` activates an
extra layer next to BP-open cells, including some solid-side ghost centers.
`VOLP` is available to postprocessing but is not the storage weight in this
transport operator. Its discrete conserved mass is

\[
M_h=\sum_{active\ leaf\ cells} BT_i\,\Delta V_i C_i.
\]

Each geometric patch contributes `-A*J_out/DeltaV` exactly once to its
active owner's or reported fallback receiver's RHS. The stage concentration
field is read without modification.
The legacy stencil routines act only on types 0 and 1, so Robin does not
also overwrite ghost values. Internal face fluxes still cancel in this
Cartesian budget. The zero-rate path reduces to the legacy zero-flux path.
Strong reaction enforces the reconstructed wall value, **not an identical
algebraic update to the legacy ghost-overwrite Dirichlet implementation**.

`M_h` is not exactly `sum(VOLP*C)`: the distinction is a spatial immersed
boundary/storage error. In particular solid-side ghost-center values can
be extrapolated negative even for a positive physical fluid solution. There
is no clipping. A study requiring exact physical cut-volume storage needs
a separately validated transport/volume discretization; this branch must
not be described as delivering that guarantee. The tests distinguish the
discrete budget from concentration/uptake errors against physical solutions.

`robin_budget: true` prints, per full step:

```text
ROBIN_BUDGET scalar step end_time mass delta_mass supply uptake residual
ROBIN_RATE   scalar step positive_outward_rate negative_outward_rate
```

`supply` and `uptake` are **step-integrated amounts**, using the same 2N RK
recurrence and coefficients as the scalar. `residual = delta_mass - supply
+ uptake`. Supply is the integrated pre-Robin RHS; it includes any existing
volume sources and non-Robin boundaries. In the verification cases there
are no volume sources, so it represents the outer-boundary contribution.
`ROBIN_RATE` reports the last RK RHS evaluation's instantaneous positive
and negative surface rates, not the end-of-step concentration's rate.
Positive and negative amounts are separated per patch before accumulation,
so opposing patches sharing a receiver do not cancel out of this report.
MPI reductions and `FINECELL` exclude halo cells and covered coarse cells.
The budget also exposes nonconservative changes made by mixed legacy
Dirichlet ghost overwrites; it cannot make those old updates conservative.

Ghost-cell runs also print `ROBIN_VOLP_POINT_BUDGET` with the same columns.
Its mass is `sum(FINECELL*VOLP*C_point)`, where `VOLP` is an absolute fluid
volume and the transported concentration is a point value. It is a
physical-storage approximation, not a cut-cell volume-average mass. The
same RK supply and uptake are retained; the resulting residual is reported
without forcing it to zero. It equals the step change in the difference
between this approximate storage and `M_h`, plus the discrete residual.
See the validation record for independently checked HDF5 masses and the
remaining strict physical-conservation acceptance limitation.

The explicit stage guard bounds the molecular diffusion row norm plus
the summed surface sink row norm at each receiver, multiplied by
`dt*max(abs(RK b))`, to 0.5. All patches sharing a receiver contribute to
that bound; checking each patch separately would miss combined stiffness.
It checks
every stage, including after the existing adaptive flow timestep changes.
An unsafe step aborts with a recommended upper bound; no scalar-specific
substepping or silent change to the coupled flow timestep is made. Because
normal stencils are nonlocal, this conservative stiffness screen is **not a
proof of monotonicity or stability for arbitrary geometry/advection/LES**.
Time refinement remains mandatory; no concentration clipping hides failures.

## Code map

1. `init_scacore` in `scacore_mod.F90`: parse per-scalar geometry flags,
   units, finite parameters, and optional budget switch.
2. `init_scalar -> create_robin`: construct geometry-only patch stencils
   after `BT` and the ghost-cell geometry exist. Rebuilt on restart from
   the current input and existing geometry/stencil cache.
3. `timeintegrate_scalar -> tstsca4 -> bound_scaflux -> fluxbalance`:
   existing face-flux assembly and divergence.
4. Existing source/Neumann calls, then `apply_robin`: evaluate the current
   stage's multi-point wall law and subtract `A*J_out/DeltaV`.
5. Existing `rkstep -> maskbt -> set_scastencils -> ftoc -> connect`:
   update scalar and synchronize hierarchy/halos. Robin is not handled by
   the ghost-value overwrite in `set_scastencils`.
6. `scalar_integral`: optional globally reduced leaf-cell budget.
7. `robin_wall_mod.F90`: small pure numerical wall-law kernel, directly
   tested by `tests/robin_wall_test.F90`.

## Build and verification

```sh
cmake -S . -B build -GNinja --preset=gnu-debug -DMGLET_REAL64=ON
cmake --build build -j2
ctest --test-dir build/tests -R 'ScalarRobin' --output-on-failure
ctest --test-dir build/tests --output-on-failure
```

For root-run OpenMPI containers set `OMPI_ALLOW_RUN_AS_ROOT=1` and
`OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1`. The standalone Python driver can also
run as `python3 tests/ScalarRobin/run.py --solver /path/to/mglet --work DIR`.
Assertions and subprocess failures return nonzero; missing dependencies
do not silently pass. All generated inputs and solver logs remain in DIR.

The unit test directly calls the production Fortran kernel. Integration
tests run the real solver: planar resistance/profile, zero/large rates,
equilibrium and reverse flux, three mesh sizes, RK timestep refinement,
multiple scalars against isolated runs, a two-Robin-sphere finite shell,
curved-surface timestep refinement and relaxation, mass budgets, MPI,
restart, and old value/flux conditions. The sphere grids share one fixed
resolved STL, so the grid comparison isolates Cartesian refinement.
The finite shell
reference is

\[
Q=\frac{4\pi(C_{eq,b}-C_{eq,a})}
{1/(k_a a^2)+(1/a-1/b)/D+1/(k_b b^2)}.
\]

This includes the outer sphere's finite transfer resistance; it is not an
infinite-domain sphere reference applied to a finite Cartesian box.
See `scalar-robin-validation.md` for actual run results and remaining gates.
