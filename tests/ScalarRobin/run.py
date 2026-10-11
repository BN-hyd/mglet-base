"""End-to-end Robin checks. Dependencies: numpy, h5py (missing = failure).

All cases run the real MGLET executable; no reimplementation of its RHS.
Use --generate DIR for a standalone, runnable planar input example.
"""
import argparse
import copy
import json
import math
import os
import shutil
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def grid(path, counts, bounds, reservoir):
    """One Cartesian grid, two halo layers, NOS + scalar SWA faces."""
    dtype = [(s, '<i4') for s in ('IGRID', 'II', 'JJ', 'KK', 'LEVEL',
             'IPARENT', 'IPOSITION', 'JPOSITION', 'KPOSITION', 'FLAGS',
             'IGRIDLVL')]
    dtype += [('BBOX', '<f8', (6,)), ('NBRGRID', '<i4', (26,))]
    gi = np.zeros(1, dtype=dtype)
    for s in ('IGRID', 'LEVEL', 'FLAGS', 'IGRIDLVL'):
        gi[s] = 1
    for s, n in zip(('II', 'JJ', 'KK'), counts):
        gi[s] = n + 4
    gi['BBOX'] = np.asarray(bounds).ravel()
    btype = [('NBOCD', '<i4'), ('TYPE', 'S8', (2,)),
             ('INTPRM', '<i4', (2, 2)), ('REALPRM', '<i4', (2, 2))]
    bc = np.zeros(1, dtype=btype)
    bc['NBOCD'] = 2
    bc['TYPE'] = [b'NOS', b'SWA']
    ns = len(reservoir)
    coords = []
    with h5py.File(path, 'w') as f:
        info = f.create_dataset('GRIDINFO', data=gi)
        # First scalar parameter block is zero flux; second fixed reservoir.
        info.attrs['INTPRMS'] = np.array([1]*ns + [0]*ns, dtype='<i4')
        info.attrs['REALPRMS'] = np.array([0.]*ns + reservoir, dtype='<f8')
        for face in ('FRONT', 'BACK', 'RIGHT', 'LEFT', 'BOTTOM', 'TOP'):
            offset = ns + 1 if face == 'BACK' else 1
            bc['INTPRM'][0, 1] = [offset, ns]
            bc['REALPRM'][0, 1] = [offset, ns]
            f.create_dataset(face, data=bc)
        for axis, n, (lo, hi) in zip('XYZ', counts, bounds):
            h = (hi-lo)/n
            x = lo + (np.arange(n+4)-1.5)*h
            coords.append(x)
            for name, values in ((axis, x), ('D'+axis, np.full(n+4, h)),
                                 ('DD'+axis, np.full(n+4, h))):
                g = f.create_group(name)
                g.attrs['MINLEVEL'] = np.int32(1)
                g.attrs['MAXLEVEL'] = np.int32(1)
                g.create_dataset('LEVEL1', data=values[None, :])
    return coords


def stl(path, triangles):
    with open(path, 'w') as f:
        f.write('solid wall\n')
        for tri in triangles:
            normal = np.cross(tri[1]-tri[0], tri[2]-tri[0])
            normal /= np.linalg.norm(normal)
            f.write('facet normal ' + ' '.join(map(str, normal)) + '\n')
            f.write('outer loop\n')
            for vertex in tri:
                f.write('vertex ' + ' '.join(map(str, vertex)) + '\n')
            f.write('endloop\nendfacet\n')
        f.write('endsolid wall\n')


def split_grid(path, n=12):
    """Two connected blocks; the reactive plane crosses their MPI interface."""
    temporary = [path.with_name(f'grid-part-{i}.h5') for i in range(2)]
    for i, name in enumerate(temporary):
        grid(name, (n, 4, 4), [(-.23, 1.), (.2*i, .2*(i+1)), (0., .4)], [1.])
    with h5py.File(temporary[0]) as f0, h5py.File(temporary[1]) as f1:
        with h5py.File(path, 'w') as f:
            infos = np.concatenate([f0['GRIDINFO'][:], f1['GRIDINFO'][:]])
            infos['IGRID'] = [1, 2]
            infos['IGRIDLVL'] = [1, 2]
            infos['NBRGRID'][0, 3] = 2  # LEFT = positive y
            infos['NBRGRID'][1, 2] = 1  # RIGHT = negative y
            info = f.create_dataset('GRIDINFO', data=infos)
            info.attrs.update(dict(f0['GRIDINFO'].attrs))
            for name in f0:
                if name == 'GRIDINFO':
                    continue
                if isinstance(f0[name], h5py.Group):
                    group = f.create_group(name)
                    group.attrs.update(dict(f0[name].attrs))
                    group.create_dataset('LEVEL1', data=np.concatenate(
                        [f0[name+'/LEVEL1'][:], f1[name+'/LEVEL1'][:]]))
                else:
                    bc = np.concatenate([f0[name][:], f1[name][:]])
                    for idx, face in ((0, 'LEFT'), (1, 'RIGHT')):
                        if name == face:
                            bc['NBOCD'][idx] = 1
                            bc['TYPE'][idx] = [b'CON', b'']
                            bc['INTPRM'][idx] = 0
                            bc['REALPRM'][idx] = 0
                    f.create_dataset(name, data=bc)
    for name in temporary:
        name.unlink()


def box(path, wall=0.):
    vertices = np.array([[x, y, z] for x in (-2., wall)
                        for y in (-2., 2.) for z in (-2., 2.)])
    triangles = []
    for face in ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
                 (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)):
        for ids in ((face[0], face[1], face[2]),
                    (face[0], face[2], face[3])):
            triangles.append(vertices[list(ids)])
    stl(path, triangles)


def sphere(path, radius, subdivisions, inward=False):
    triangles = []
    for a in (-1., 1.):
        for b in (-1., 1.):
            for c in (-1., 1.):
                t = np.array([[a, 0, 0], [0, b, 0], [0, 0, c]])
                if np.dot(np.cross(t[1]-t[0], t[2]-t[0]), t[0]) < 0:
                    t = t[[0, 2, 1]]
                triangles.append(t)
    for _ in range(subdivisions):
        refined = []
        for a, b, c in triangles:
            ab, bc, ca = [(u+v)/np.linalg.norm(u+v)
                          for u, v in ((a, b), (b, c), (c, a))]
            refined.extend([np.array(t) for t in
                            ((a, ab, ca), (ab, b, bc), (ca, bc, c),
                             (ab, bc, ca))])
        triangles = refined
    stl(path, [radius*t[[0, 2, 1] if inward else [0, 1, 2]]
               for t in triangles])


def parameters(scalars, dt, steps, spherical=False):
    walls = ['inner.stl', 'outer.stl'] if spherical else ['wall.stl']
    return {
        'io': {'grids': 'grids.h5', 'infile': 'fields.h5',
               'outfile': 'fields.h5'},
        'time': {'dt': dt, 'mtstep': steps, 'itinfo': 100000,
                 'tstat': 1.e10, 'read': False, 'write': True,
                 'continue': False, 'rkmethod': 'williamson'},
        'flow': {'gmol': 1., 'rho': 1., 'uinf': [0., 0., 0.],
                 'tu_level': 0., 'solve': False, 'lesmodel': {'model': 'none'}},
        'ib': {'type': 'ghostcell', 'blocking': 'newblock',
               'stencilfile': 'ib_stencils.h5',
               'fluidpoints': [[.7, .3, .3] if spherical else [.5, .2, .2]],
               'geometries': [{'file': p} for p in walls]},
        'scalar': {'solve': True, 'scalars': scalars}, 'statistics': []}


def planar(directory, n=24, specs=((1., 1., .2, 1.),),
           duration=.12, dt_scale=1., initial=None, kind='robin'):
    directory.mkdir(parents=True, exist_ok=True)
    bounds = [(-.23, 1.), (0., .4), (0., .4)]
    coords = grid(directory/'grids.h5', (n, 4, 4), bounds,
                  [s[3] for s in specs])
    box(directory/'wall.stl')
    h = min(1.23/n, .1)
    dt = .02*h*h/max(s[0] for s in specs)*dt_scale
    steps = math.ceil(duration/dt)
    dt = duration/steps
    scalars, exact, fluxes = [], [], []
    for index, (diff, ks, eq, c0) in enumerate(specs):
        flux = (c0-eq)/(1./diff + 1./ks) if ks > 0 else 0.
        intercept = c0-flux/diff
        expression = f'{intercept:.17g}+({flux/diff:.17g})*x'
        if kind == 'robin':
            # BT may include a solid-side cell beyond the cut-cell owner.
            # At discrete steady state that dead-end extension is constant;
            # a linear extension would introduce an unintended transient.
            cut_x = min(coords[0][2:-2], key=abs)
            expression = (f'{intercept:.17g}+({flux/diff:.17g})'
                          f'*max(x,{cut_x:.17g})')
        bc = {'type': kind, 'k_s': ks, 'c_eq': eq}
        if kind == 'value':
            bc = {'type': 'value', 'value': eq}
        if kind == 'flux':
            bc = {'type': 'flux', 'value': -flux}
        scalars.append({'name': f'C{index}', 'prmol': 1./diff,
                        'initial_value': expression if initial is None
                        else initial, 'robin_budget': kind == 'robin',
                        'geometries': {'wall.stl': bc}})
        exact.append(intercept+flux/diff*coords[0][2:-2])
        fluxes.append(flux*.4*.4)
    config = parameters(scalars, dt, steps)
    (directory/'parameters.json').write_text(json.dumps(config, indent=2))
    return config, coords, exact, fluxes


def shell(directory, n):
    directory.mkdir(parents=True, exist_ok=True)
    coords = grid(directory/'grids.h5', (n, n, n), [(-1.6, 1.6)]*3, [0.])
    # Independent analytic finite shell: Robin on BOTH concentric spheres.
    a, b, ki, ko, diff = .5, 1.4, 1., 2., 1.
    q = 4*math.pi/(1/(ki*a*a)+(1/a-1/b)/diff+1/(ko*b*b))
    coeff = q/(4*math.pi)
    expression = (f'{coeff/(ki*a*a)+coeff/(diff*a):.17g}'
                  f'-{coeff/diff:.17g}/max(sqrt(x*x+y*y+z*z),0.01)')
    subdivisions = 3 if n <= 16 else 4
    sphere(directory/'inner.stl', a, subdivisions)
    sphere(directory/'outer.stl', b, subdivisions, inward=True)
    scalar = {'name': 'C0', 'prmol': 1., 'initial_value': expression,
              'robin_budget': True,
              'geometries': {'inner.stl': {'type': 'robin', 'k_s': ki,
                                         'c_eq': 0.},
                             'outer.stl': {'type': 'robin', 'k_s': ko,
                                         'c_eq': 1.}}}
    duration = .15
    steps = math.ceil(duration/(.01*(3.2/n)**2))
    config = parameters([scalar], duration/steps, steps, spherical=True)
    (directory/'parameters.json').write_text(json.dumps(config, indent=2))
    return config, coords, q


def multilevel(directory, equilibrium):
    """Existing 36-block / three-level mesh, plane crossing refinement edges."""
    directory.mkdir(parents=True, exist_ok=True)
    candidates = [p/'tests/Parker_micro/grids.h5'
                  for p in Path(__file__).resolve().parents]
    fixture = next((p for p in candidates if p.is_file()), None)
    require(fixture is not None, 'Parker_micro grid fixture not found')
    shutil.copyfile(fixture, directory/'grids.h5')
    with h5py.File(directory/'grids.h5', 'r+') as f:
        f['GRIDINFO'].attrs['INTPRMS'] = np.array([1], dtype='<i4')
        f['GRIDINFO'].attrs['REALPRMS'] = np.array([0.], dtype='<f8')
        for face in ('FRONT', 'BACK', 'RIGHT', 'LEFT', 'BOTTOM', 'TOP'):
            data = f[face][:]
            for row in data:
                if row['TYPE'][0].strip() in (b'CON', b'PAR'):
                    continue
                row['NBOCD'] = 2
                row['TYPE'] = [b'NOS', b'SWA']
                row['INTPRM'] = [[0, 0], [1, 1]]
                row['REALPRM'] = [[0, 0], [1, 1]]
            f[face][...] = data
    box(directory/'wall.stl', wall=-.0193)
    scalar = {'name': 'C0', 'prmol': 1., 'initial_value': .7,
              'robin_budget': True,
              'geometries': {'wall.stl': {'type': 'robin', 'k_s': 1.,
                                         'c_eq': .7 if equilibrium else 0.}}}
    config = parameters([scalar], 1.e-8, 10)
    config['ib']['fluidpoints'] = [[.1, .05, .05]]
    (directory/'parameters.json').write_text(json.dumps(config, indent=2))
    return config


def execute(binary, directory, config=None, ranks=1, expect_failure=None):
    if config is not None:
        (directory/'parameters.json').write_text(json.dumps(config, indent=2))
    command = [binary, 'parameters.json']
    if ranks > 1:
        command = ['mpirun', '-np', str(ranks)]+command
    with open(directory/'solver.log', 'w') as log:
        env = dict(os.environ, GFORTRAN_UNBUFFERED_ALL='1')
        result = subprocess.run(command, cwd=directory, stdout=log,
                                stderr=subprocess.STDOUT, timeout=240, env=env)
    output = (directory/'solver.log').read_text(errors='replace')
    for path in (directory/'LOGS').glob('*'):
        if path.is_file():
            output += path.read_text(errors='replace')
    if expect_failure:
        require(result.returncode != 0 and expect_failure in output,
                f'{directory}: expected failure {expect_failure!r}, '
                f'code={result.returncode}\n{output[-5000:]}')
        return None, None
    require(result.returncode == 0, f'{directory}: solver failed\n{output[-4500:]}')
    require('SOLUTION EXPLODED' not in output, f'{directory}: exploded')
    budgets, rates = {}, {}
    for line in output.splitlines():
        p = line.split()
        if p and p[0] == 'ROBIN_BUDGET':
            values = np.array([float(v) for v in p[3:]])
            require(np.isfinite(values).all(), 'nonfinite budget')
            # time, mass, delta_mass, integrated supply, integrated uptake, residual
            scale = max(1., abs(values[1]))
            require(abs(values[-1]) < 2.e-12*scale,
                    f'{directory}: mass budget failed: {values}')
            if directory.name.startswith(('sphere', 'multilevel')):
                require(abs(values[3]) < 2.e-12*scale,
                        'closed fluid region has nonzero external supply')
            budgets[p[1]] = values
        if p and p[0] == 'ROBIN_RATE':
            rates[p[1]] = [float(v) for v in p[3:]]
    config = json.loads((directory/'parameters.json').read_text())
    for scalar in config['scalar']['scalars']:
        if any(bc['type'] == 'robin' for bc in scalar['geometries'].values()):
            require(scalar['name'] in budgets, 'missing Robin budget')
    fields = {}
    with h5py.File(directory/'grids.h5', 'r') as mesh:
        info = mesh['GRIDINFO'][:]
    with h5py.File(directory/'fields.h5', 'r') as f:
        for scalar in config['scalar']['scalars']:
            name = scalar['name']
            # fieldio2 stores flattened, transposed (i-fastest) grid data.
            blocks = []
            for g in info:
                row = f[f'VOLUMEFIELDS/{name}/LEVEL{g["LEVEL"]}'][
                    int(g['IGRIDLVL'])-1]
                blocks.append(row[:int(g['II']*g['JJ']*g['KK'])].reshape(
                    int(g['KK']), int(g['JJ']), int(g['II'])).transpose(2, 1, 0))
            fields[name] = np.asarray(blocks).squeeze()
            require(np.isfinite(fields[name]).all(), 'nonfinite concentration')
    print(f'{directory.name}: rates={rates}, budgets={budgets}', flush=True)
    if directory.name.startswith('sphere') and budgets['C0'][1] <= 0.:
        raise AssertionError('Sphere has no fluid cells:\n'+output[:7000])
    return fields, rates


def check_planar(fields, exact, rates, fluxes, tolerance=3.e-3):
    for i, reference in enumerate(exact):
        data = fields[f'C{i}']
        # MGLET HDF5 ordering is (grid, i, j, k), includes two halos.
        profile = data[2:-2, 2:-2, 2:-2].mean(axis=(1, 2))
        # Exclude solid-side ghost values from pointwise physical comparison.
        x = -.23 + (np.arange(len(reference))+.5)*1.23/len(reference)
        physical = x > 1.23/len(reference)
        error = np.max(np.abs(profile[physical]-reference[physical]))
        require(error < tolerance, f'planar profile error {error}')
        if rates is not None:
            measured = sum(rates[f'C{i}'])
            require(abs(measured-fluxes[i]) < tolerance*.16,
                    f'planar flux {measured}, expected {fluxes[i]}')
        print(f'planar C{i}: max error {error:.6g}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver')
    parser.add_argument('--generate', type=Path)
    parser.add_argument('--work', type=Path, default=Path('runs'))
    parser.add_argument('--suite', default='all')
    args = parser.parse_args()
    if args.generate:
        planar(args.generate)
        print(f'Run MGLET in {args.generate.resolve()}')
        return
    require(args.solver is not None, '--solver is required')
    binary = str(Path(args.solver).resolve())
    root = args.work.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.suite == 'all':
        failed = []
        for suite in ('planar', 'limits', 'isolation', 'legacy',
                      'transient', 'sphere', 'multilevel', 'invalid'):
            result = subprocess.run([sys.executable, __file__, '--solver',
                                     binary, '--work', str(root),
                                     '--suite', suite])
            if result.returncode:
                failed.append(suite)
        require(not failed, f'Failed suites: {failed}')
        print('Scalar Robin integration tests: PASS', flush=True)
        return
    run_suite(args.suite, binary, root)


def run_suite(suite, binary, root):
    if suite == 'planar':
        for n in (12, 24, 48):
            directory = root/f'plane-{n}'
            _, _, exact, flux = planar(directory, n=n)
            fields, rates = execute(binary, directory)
            check_planar(fields, exact, rates, flux)
        directory = root/'relax-to-steady'
        _, _, exact, flux = planar(directory, n=12, initial=1., duration=5.)
        fields, rates = execute(binary, directory)
        check_planar(fields, exact, rates, flux, tolerance=4.e-3)
        return
    if suite == 'limits':
        for name, spec in [('zero', (1., 0., .2, 1.)),
                           ('strong', (1., 1.e12, .2, 1.)),
                           ('equal', (1., 1., .2, .2)),
                           ('reverse', (1., 2., .7, .1))]:
            directory = root/name
            _, _, exact, flux = planar(directory, specs=(spec,))
            fields, rates = execute(binary, directory)
            check_planar(fields, exact, rates, flux)
        return
    if suite == 'isolation':
        specs = ((1., 1., .2, 1.), (.3, 2., .1, .8))
        directory = root/'multiple'
        combined_config, _, exact, flux = planar(directory, specs=specs)
        fields, rates = execute(binary, directory)
        check_planar(fields, exact, rates, flux)
        for i, spec in enumerate(specs):
            single = root/f'single-{i}'
            isolated_config, _, _, _ = planar(single, specs=(spec,))
            isolated_config['time'] = copy.deepcopy(combined_config['time'])
            isolated, _ = execute(binary, single, isolated_config)
            require(np.max(np.abs(fields[f'C{i}']-isolated['C0'])) < 2.e-12,
                    'scalar parameter cross-contamination')
        return
    if suite == 'legacy':
        for kind in ('flux', 'value'):
            directory = root/f'legacy-{kind}'
            _, _, exact, flux = planar(directory, kind=kind,
                                      specs=((1., 1.e12, .2, 1.),))
            fields, _ = execute(binary, directory)
            check_planar(fields, exact, None, flux, tolerance=.025)
        return
    if suite == 'transient':
        # Time convergence from a nonsteady state, fixed mesh and final time.
        solutions = []
        for scale in (1., .5, .25):
            directory = root/f'time-{scale}'
            planar(directory, n=12, initial=1., duration=.05, dt_scale=scale)
            fields, _ = execute(binary, directory)
            solutions.append(fields['C0'])
        e1 = np.linalg.norm(solutions[0]-solutions[1])
        e2 = np.linalg.norm(solutions[1]-solutions[2])
        require(e2 < .3*e1, f'RK time convergence failed: {e1}, {e2}')
        print(f'RK refinement differences: {e1:.6g}, {e2:.6g}', flush=True)
        # Restart and MPI use the same transient as an uninterrupted run.
        directory = root/'restart'
        config, _, _, _ = planar(directory, n=12, initial=1., duration=.05)
        whole = copy.deepcopy(config)
        first = copy.deepcopy(config)
        first['time']['mtstep'] = whole['time']['mtstep']//2
        execute(binary, directory, first)
        second = copy.deepcopy(config)
        second['time'].update({'read': True, 'continue': True,
                              'mtstep': whole['time']['mtstep']-first['time']['mtstep']})
        second['ib']['blocking'] = 'useblock'
        resumed, _ = execute(binary, directory, second)
        require(np.max(np.abs(resumed['C0']-solutions[0])) < 2.e-11,
                'restart differs from uninterrupted integration')
        directory = root/'mpi'
        config, _, _, _ = planar(directory, n=12, initial=1., duration=.05)
        split_grid(directory/'grids.h5')
        serial_blocks, _ = execute(binary, directory)
        parallel, _ = execute(binary, directory, ranks=2)
        require(np.max(np.abs(parallel['C0']-serial_blocks['C0'])) < 2.e-11,
                'MPI differs from serial')
        return
    if suite == 'sphere':
        sphere_errors = []
        for n in (16, 24, 32):
            directory = root/f'sphere-{n}'
            _, _, reference = shell(directory, n)
            fields, rates = execute(binary, directory)
            uptake, release = rates['C0']
            error = abs(uptake/reference-1.)
            sphere_errors.append(error)
            require(error < .25, f'sphere uptake error {error}')
            require(abs(release/reference+1.) < .25, 'sphere supply error')
            print(f'sphere N={n}: Q={uptake:.9g}, exact={reference:.9g}, '
                  f'relative error={error:.6g}', flush=True)
        require(sphere_errors[-1] < .1 and sphere_errors[-1] < sphere_errors[0],
                f'sphere refinement failed: {sphere_errors}')
        return
    if suite == 'invalid':
        for key, bad, message in [('k_s', -1., 'k_s must be finite'),
                                  ('prmol', 0., 'positive finite prmol')]:
            directory = root/f'invalid-{key}'
            config, _, _, _ = planar(directory)
            scalar = config['scalar']['scalars'][0]
            if key == 'prmol':
                scalar[key] = bad
            else:
                scalar['geometries']['wall.stl'][key] = bad
            execute(binary, directory, config, expect_failure=message)
        return
    if suite == 'multilevel':
        for equilibrium in (True, False):
            directory = root/f'multilevel-{equilibrium}'
            multilevel(directory, equilibrium)
            serial, rates = execute(binary, directory)
            parallel, _ = execute(binary, directory, ranks=2)
            require(np.max(np.abs(serial['C0']-parallel['C0'])) < 2.e-11,
                    'multilevel MPI differs from serial')
            uptake = sum(rates['C0'])
            if equilibrium:
                require(abs(uptake) < 1.e-12,
                        'multilevel equilibrium has surface flux')
                # Compare active cell values only; blocked cells are zero.
                # Coarse covered cells and interpolated halo cells are not
                # independent fluid unknowns. Compare physical fluid centers.
                with h5py.File(directory/'grids.h5') as f:
                    info = f['GRIDINFO'][:]
                    error = 0.
                    for block, g in zip(serial['C0'], info):
                        xs = f[f'X/LEVEL{g["LEVEL"]}'][int(g['IGRIDLVL'])-1]
                        physical = xs[2:-2] > -.01
                        error = max(error, np.max(np.abs(
                            block[2:-2, 2:-2, 2:-2][physical]-.7), initial=0.))
                require(error < 2.e-11,
                        f'multilevel equilibrium changed: {error}')
            else:
                require(uptake > 0., 'missing multilevel surface reaction')
            print(f'three-level equilibrium={equilibrium}: Q={uptake:.9g}',
                  flush=True)
        return
    raise ValueError(f'Unknown suite: {suite}')


if __name__ == '__main__':
    main()
