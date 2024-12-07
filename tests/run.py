'''
Este código hace la comparativa entre el método de Montercarlo y FEM.
El código se puede dividir en 4 partes:
    1. Definición de utilidades (sdf, puntos de interpolación)
    2. Solución con Montecarlo
    3. Solución con FEM
    4. Comparativa

El tiempo de ejecución se toma usando perf_counter de la biblioteca time.
Se considera desde la construcción de la geometría hasta la solución numérica.

Bug. Por alguna razón, la solución de mc interfiere con la fem. Así que toca
    correrlos en archivos separados.
'''

import numpy as np
import matplotlib.pyplot as plt

from time import perf_counter

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
    origin = (bmax + bmin) / 2
    qp = query_point - origin[None, :]
    upper_corner = bmax - origin

    d = np.abs(qp) - upper_corner[None, :]
    d = d.clip(min=0.0) + (np.max(d, axis=1)).clip(max=0.0)[:, None]
    return np.linalg.norm(d, axis=1)


def sd_box_inside(query_point, bmin, bmax):
    bo = (bmax + bmin) / 2
    ab = (bmax - bmin) / 2
    oq = query_point - bo[None, :]

    oq = ab[None, :] - np.abs(oq)
    return np.min(oq, axis=1)


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
            bnd_dist = mesh.dist2bnd(x).clip(min=0.0)

            bnd_dist_change = bnd_dist[bnd_dist > 0.0]

            r = bnd_dist_change * np.random.uniform(bnd_tol, 1.0, size=len(bnd_dist_change))
            t = np.random.uniform(0, 2*np.pi, size=len(bnd_dist_change))

            y = x[bnd_dist > 0.0] + np.array([r*np.cos(t), r*np.sin(t)]).T
            retval[bnd_dist > 0.0] += bnd_dist_change**2 * f(y) * Gball(r, bnd_dist_change)

            s = np.random.uniform(0, 2*np.pi, size=len(bnd_dist_change))
            x[bnd_dist > 0.0] += np.array([bnd_dist_change*np.cos(s), bnd_dist_change*np.sin(s)]).T

            steps += 1
        retval += g(x)
    return retval/nwalkers


''' Testing '''

# Problem setting: Geometry
geom_primitives = [('circle', np.array((0.5, 0.5)), 0.25)]

# n_circles = 1
# for i in range(n_circles):
#     center = np.random.uniform(0.1, 0.9, size=2)
#     radius = np.random.uniform(0.05, 0.1)
#     geom_primitives.append(('circle', center, radius))

# Utilities for interpolation
num = 256
x = np.linspace(0.0, 1.0, num=num)
X = np.dstack(np.meshgrid(x, x)).reshape((-1, 2))

# Solve with MonteCarlo
tmc_start = perf_counter()

sd_list = [lambda x: sd_box_inside(x, np.array((0.0, 0.0)), np.array((1.0, 1.0)))]
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
    i = np.argmin(dist, axis=0)
    ret[i == 0] = np.cos(2*np.pi*x[i == 0, 0]) * np.sin(2*np.pi*x[i == 0, 1])
    return ret


u_mc = WoS2D(X, sd_field, f_mc, g_mc, bnd_tol=1e-5, nwalkers=20, nsteps=20)

tmc_stop = perf_counter()

u_mc = u_mc.reshape((num, num))


# Solve with FEM (ngsolve)
import ngsolve as ng
from netgen.geom2d import CSG2d, Circle, Rectangle

tng_start = perf_counter()

geom = Rectangle((0.0, 0.0), (1.0, 1.0), mat='mat1', left='left', top='top', right='right', bottom='bottom')

for (type, *attr) in geom_primitives:
    match type:
        case 'rectangle': geom = geom - Rectangle(tuple(attr[0]), tuple(attr[1]), mat='mat2', bc='bc_hole')
        case 'circle': geom = geom - Circle(tuple(attr[0]), attr[1], mat='mat2', bc='bc_hole')
        case _: raise ValueError('Bad geometry')

solid_geom = CSG2d()
solid_geom.Add(geom)

mesh_ng = ng.Mesh(solid_geom.GenerateMesh(maxh=0.05))

fes = ng.H1(mesh_ng, order=1, dirichlet='left|right|top|bottom|bc_hole')

u, v = fes.TnT()

f_ng = ng.GridFunction(fes)
f_ng.Set(8*ng.pi*ng.pi * ng.cos(2*ng.pi*ng.x) * ng.sin(2*ng.pi*ng.y))

b_ng = ng.BilinearForm(ng.grad(u)*ng.grad(v)*ng.dx).Assemble()
force = ng.LinearForm(f_ng*v*ng.dx).Assemble()

g_ng = ng.GridFunction(fes)
g_ng.Set(ng.cos(2*ng.pi*ng.x) * ng.sin(2*ng.pi*ng.y), ng.BND, definedon=mesh_ng.Boundaries('left|bottom|right|top'))

u_ng_sol = ng.GridFunction(fes)
u_ng_sol.Set(ng.CF(0), ng.BND, definedon=mesh_ng.Boundaries('bc_hole'))
u_ng_sol.Set(g_ng, ng.BND, definedon=mesh_ng.Boundaries('left|bottom|right|top'))

c = ng.Preconditioner(b_ng, 'local')
c.Update()

ng.solvers.BVP(bf=b_ng, lf=force, gf=u_ng_sol, pre=c, print=False)

tng_stop = perf_counter()

u_ng = np.zeros(len(X))
for i, p in enumerate(X):
    try:
        u_ng[i] = u_ng_sol(mesh_ng(*p))
    except Exception:
        continue
u_ng = u_ng.reshape((num, num))

# Report results
print('Time MC:', tmc_stop - tmc_start)
print('Time NG:', tng_stop - tng_start)

fig, ax = plt.subplots(ncols=3)
fig.tight_layout()
for i, (img, text) in enumerate(zip([u_mc, u_ng, u_ng-u_mc], ['mc', 'ng', 'diff'])):
    p = ax[i].imshow(img)
    ax[i].axis('off')
    ax[i].title.set_text(text)
    # fig.colorbar(p, ax=ax[i], location='bottom')
plt.show()
