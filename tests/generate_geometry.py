import numpy as np
from pathlib import Path

GEOM_DIR = Path('geom')
GEOM_DIR.mkdir(exist_ok=True)

n_geom = 10
for g in range(n_geom):
    file = open(GEOM_DIR / f'geom_{g}.txt', 'w')

    n_circles = g
    for i in range(n_circles):
        center = np.random.uniform(0.1, 0.9, size=2)
        radius = np.random.uniform(0.05, 0.1)
        file.write(f'circle;{center[0]} {center[1]};{radius}\n')

    file.close()
