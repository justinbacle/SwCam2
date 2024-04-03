import matplotlib.pyplot as plt
import numpy as np

import lib
import const


l = np.linspace(
    -0.1,
    1.1,
    1000)
s = [lib.saturationVsLumaCurve(_, 0.5) for _ in l]

plt.plot(l, s)
plt.show()
