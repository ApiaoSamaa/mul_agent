# %%
import matplotlib as mpl
import matplotlib.pyplot as plt

fig = plt.figure((8*8))
fig.add_subplot(1,1,1)
# fig.add_subplot(1,3,2)
plt.show()
# %%
import matplotlib.pyplot as plt
x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]
fig = plt.figure()
fig.add_subplot(211)
fig.add_subplot(122)
plt.scatter(x, y)
plt.show()
# %%
fig = plt.figure()  # an empty figure with no Axes
# fig, ax = plt.subplots()  # a figure with a single Axes
fig, axs = plt.subplots(2, 2)  # a figure with a 2x2 grid of Axes
plt.show()
# %%
