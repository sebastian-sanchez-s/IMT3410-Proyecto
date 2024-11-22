# Ambiente virtual

Crea un ambiente virtual con
```sh
python -m venv venv
```
Luego, entra en ese ambiente e instala los paquetes
```sh
source venv/bin/activate
pip install -r requirements.txt
ipython kernel install --user --name=venv
```
Con esto, todos estamos corriendo la misma versión de los 
paquetes de Python.

> En jupyter, se debe cambiar el kernel (sale en la interfaz, esquina superior derecha)
