import numpy as np
import easing_functions as easing
import matplotlib.pyplot as plt

# 111100000000
a = easing.ExponentialEaseInOut(start=0, end=1.0)

x = np.arange(0, 1, 1 / 12)
y0 = list(map(a, x))

plt.plot(x, y0)

plt.savefig("x.png")

print(y0)
