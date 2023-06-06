# %%
from utils import *
from draw_graphs import *
import matplotlib.pyplot as pl

# define network parameters
N = 21
k_over_2 = 2
betas = [0, 0.025, 0.4, 1.0]
labels = [ r'$\beta=0$', r'$\beta=0.025$', r'$\beta=0.4$', r'$\beta=1$']

focal_node = 0

fig, ax = pl.subplots(1,4,figsize=(9,4))

# scan beta values
for ib, beta in enumerate(betas):

    # generate small-world graphs and draw
    G = generate_small_world_graph(N, k_over_2, beta)
    draw_network(G,k_over_2,focal_node=focal_node,ax=ax[ib])

    ax[ib].set_title(labels[ib],fontsize=11)

# show
pl.subplots_adjust(wspace=0.3)
pl.show()

# %%
