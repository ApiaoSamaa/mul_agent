# %%
import numpy as np

t = np.linspace(0,1,20)
print(t)

# %%
n = 20
B = np.zeros((n,2))
for part in range(n):
    t_ = t[part]
    B[part,:] = (1-t_)**2 * P0 + 2*(1-t_)*t_*P1+t_**2*P2