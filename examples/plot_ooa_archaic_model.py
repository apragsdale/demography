import demography
import models
import matplotlib.pylab as plt

G = models.ragsdale_archaic_admixture()
dg = demography.DemoGraph(G)

fig = plt.figure(1, figsize=(8,6))

ax1 = plt.subplot(1,2,1)
demography.plotting.plot_graph(dg, 
    leaf_order=['AA', 'YRI', 'CEU', 'CHB', 'Neand'],
    ax=ax1)

ax2 = plt.subplot(1,2,2)
demography.plotting.plot_demography(dg, 
    leaf_order=['AA', 'YRI', 'CEU', 'CHB', 'Neand'],
    ax=ax2, flipped=['CEU'], padding=2)

ax1.axis('off')
ax2.axis('off')

fig.tight_layout()
plt.savefig('ooa_archaic_model.pdf')
plt.show()
