import numpy as np
import matplotlib.pyplot as plt

from time import perf_counter

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--geom')

args = parser.parse_args()

''' SDF utilities (vectorized)

Las funciones de signo (sdf) evaluadas sobre un x da la distancia de
ese punto a la frontera del conjunto definido por la función.
i.e. ||x|| - r para un círculo de radio r centrado en el origen.

En general, el interior es negativo y el exterior positivo.
'''


class SDF:
    def __init__(self, functions):
        self.functions = functions

    def dist2bnd(self, query_point):
        '''
            query_point     [N]x[2]
            functions       [F]
            -> [N]

            Retorna la distancia más chica hacia la frontera de la geometría
            definida por las funciones.
        '''
        ret = np.array([f(query_point) for f in self.functions])    # [F]x[N]
        return np.min(ret, axis=0)


# Adapted from: https://iquilezles.org/articles/distfunctions2d/


def sd_circle(query_point, circle_center, circle_radius):
    '''
        query_point     [N]x[2]
        circle_center   [2]
        -> [N]
    '''
    return np.linalg.norm(query_point - circle_center[None, :], axis=1) - circle_radius


def sd_box(query_point, bmin, bmax):
    '''
        query_point     [N]x[2]
        bmin            [2]
        -> [N]
    '''
    origin = (bmin + bmax) / 2
    qp = query_point - origin[None, :]
    upper_corner = bmax - origin

    d = np.abs(qp) - upper_corner[None, :]
    d = d.clip(min=0.0) + (np.max(d, axis=1)).clip(max=0.0)[:, None]
    return np.linalg.norm(d, axis=1)


def sd_unit_square(query_point):
    ''' query_point     [N]x[2] ->  [N]'''
    query_point_x = query_point[:, 0]
    dist_x = np.min((query_point_x, 1.0 - query_point_x), axis=0)

    query_point_y = query_point[:, 1]
    dist_y = np.min((query_point_y, 1.0 - query_point_y), axis=0)

    return np.min((dist_x, dist_y), axis=0)


''' WoS implementation '''


def Gball(r, R):
    ''' función de Green para la bola en 2d de radio R evaluada en un radio r '''
    return np.nan_to_num(np.log(R/r)) / (2*np.pi)


def WoS2D(x0, mesh, f, g, bnd_tol=1e-3, nwalkers=20, nsteps=10):
    retval = np.zeros(len(x0))
    for w in range(nwalkers):
        x = x0
        steps = 0
        bnd_dist = float('inf')
        while np.max(bnd_dist) > bnd_tol and steps < nsteps:
            bnd_dist = mesh.dist2bnd(x)

            mask = bnd_dist > bnd_tol

            r = bnd_dist[mask] * np.random.uniform(size=len(bnd_dist[mask]))
            t = np.random.uniform(0.0, 2*np.pi, size=len(bnd_dist[mask]))

            y = x[mask, :] + np.array([r*np.cos(t), r*np.sin(t)]).T
            retval[mask] += bnd_dist[mask]**2 * f(y) * Gball(r, bnd_dist[mask])

            s = np.random.uniform(0, 2*np.pi, size=len(bnd_dist[mask]))
            x[mask, :] += np.array([bnd_dist[mask]*np.cos(s), bnd_dist[mask]*np.sin(s)]).T

            steps += 1
        retval += g(x)
    return retval/nwalkers


''' Testing '''

# Problem setting
geom_primitives = []
with open(args.geom, 'r') as f_geom:
    for line in f_geom:
        type, *attr = line.strip().split(';')
        match type:
            case 'circle':
                center = np.array([float(x) for x in attr[0].split(' ')])
                radius = float(attr[1])
                geom_primitives.append((type, center, radius))


# Utilities for interpolation
num = 256
x = np.linspace(0.0, 1.0, num=num)
X = np.dstack(np.meshgrid(x, x)).reshape((-1, 2))

# plt.imshow(
#         np.min((
#             sd_unit_square(X),
#             sd_circle(X, np.array((0.5, 0.5)), 0.1),
#             # sd_circle(X, np.array((0.0, 0.0)), 0.2)
#         ), axis=0).reshape((num, num)))
# plt.colorbar()
# plt.show()
# quit()

# Solve with MonteCarlo
tmc_start = perf_counter()

sd_list = [lambda x: sd_unit_square(x)]
for (type, *attr) in geom_primitives:
    match type:
        case 'rectangle': sd_list.append(lambda x: sd_box(x, attr[0], attr[1]))
        case 'circle': sd_list.append(lambda x: sd_circle(x, attr[0], attr[1]))
        case _: raise ValueError('Bad geometry')

sd_field = SDF(sd_list)


def f_mc(x): return 8*np.pi**2 * np.cos(2*np.pi*x[:, 0]) * np.sin(2*np.pi*x[:, 1])


def g_mc(x):
    ret = np.zeros(len(x))
    dist = np.array([f(x) for f in sd_list])
    mask = np.argmin(dist, axis=0) == 0  # sd_list[0] is the unit square
    ret[mask] = np.cos(2*np.pi*x[mask, 0]) * np.sin(2*np.pi*x[mask, 1])
    return ret


u_mc = WoS2D(X, sd_field, f_mc, g_mc, bnd_tol=1e-5, nwalkers=200, nsteps=15)

tmc_stop = perf_counter()

u_mc = u_mc.reshape((num, num))

# Report
print('Time MC:', tmc_stop - tmc_start)

plt.imshow(u_mc)
plt.title('u_mc')
plt.show()
