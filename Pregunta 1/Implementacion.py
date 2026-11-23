# Implemente una pirámide Laplaciana de cinco niveles (más el residual). En cada nivel, obtenga una versión 
# suavizada y reducida por un factor 2. Defina la capa de detalle como la diferencia entre la imagen del nivel 
# actual y la aproximación obtenida al expandir el siguiente nivel hasta el mismo tamaño. La interpolación y el 
# filtro de suavizado utilizados deben quedar explícitos y ser consistentes entre descomposición y 
# reconstrucción. 
# dⱼ = cⱼ₋₁ − U(cⱼ),      j = 1,2,3 

import numpy as np
from scipy.ndimage import convolve1d


B3 = np.array([1, 4, 6, 4, 1]) / 16
 
def suavizar(c, h):
    c = convolve1d(c, h, axis=0, mode="mirror")
    return convolve1d(c, h, axis=1, mode="mirror")


def reducir(c):
    return suavizar(c, B3)[::2, ::2]


def expandir(c, forma):
    z = np.zeros(forma)
    z[::2, ::2] = c
    return 4 * suavizar(z, B3)


def piramide_laplaciana(imagen, niveles=5):
    #([d_1..d_niveles], c_niveles) con d_j = c_{j-1} - U(c_j).
    detalles, c = [], imagen
    for _ in range(niveles):
        c_sig = reducir(c)
        detalles.append(c - expandir(c_sig, c.shape))
        c = c_sig
    return detalles, c


def reconstruir_piramide(detalles, residuo):
    #c_{j-1} = d_j + U(c_j), desde el nivel mas grueso hasta el original.
    c = residuo
    for d in reversed(detalles):
        c = d + expandir(c, d.shape)
    return c


def starlet(imagen, escalas=5):
   #Devuelve ([w_1..w_escalas], c_escalas), w_j = c_{j-1} - c_j.
    capas, c = [], imagen
    for j in range(escalas):
        h = np.zeros(4 * 2**j + 1)
        h[:: 2**j] = B3
        c_sig = suavizar(c, h)
        capas.append(c - c_sig)
        c = c_sig
    return capas, c


def reconstruir_starlet(capas, residuo):
    return residuo + sum(capas)