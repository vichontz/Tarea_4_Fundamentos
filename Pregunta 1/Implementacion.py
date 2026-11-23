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
