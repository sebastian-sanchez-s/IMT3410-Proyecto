import numpy as np
import matplotlib.pyplot as plt

import ngsolve as ng
from netgen.geom2d import CSG2d, Circle, Rectangle


from time import perf_counter

# Problem setting
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

# Solve with FEM (ngsolve)
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
g_ng.Set(ng.cos(2*ng.pi*ng.x) * ng.sin(2*ng.pi*ng.y), definedon=mesh_ng.Boundaries('left|bottom|right|top'))

u_ng_sol = ng.GridFunction(fes)
u_ng_sol.Set(g_ng, definedon=mesh_ng.Boundaries('left|bottom|right|top'))
u_ng_sol.Set(ng.CF(0), definedon=mesh_ng.Boundaries('bc_hole'))

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
print('Time NG:', tng_stop - tng_start)

plt.imshow(u_ng)
plt.title('u_ng')
plt.show()
