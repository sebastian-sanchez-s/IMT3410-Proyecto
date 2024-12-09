import os

bnd_tol = ['1e-2', '1e-3', '1e-4', '1e-5']
nwalkers = [10, 20, 50, 100]
nsteps = [100]

n_geom = 10
for g in range(n_geom):
    print('Running', g)
    for eps in bnd_tol:
        print('->', eps)
        for N in nwalkers:
            print('->', N)
            for s in nsteps:
                print('->', s)
                os.system(f'python run_montecarlo.py --geom tests/geom/geom_{g}.txt --bnd-tol {eps} --nwalkers {N} --nsteps {s} --savename tests/results/mc_{g}_{eps}_{N}_{s}')
