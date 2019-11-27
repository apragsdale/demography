"""
Create a plot of the Out-of-Africa demographic model, along with some summary
statistics, and something from some simulated data using this model.
"""
import numpy as np
from models import ooa
import matplotlib.pylab as plt
import demography

ooa_params = [2.11, 0.377, 0.251, 0.111, 0.224, 3.02, 0.0904, 5.77, 0.0711, 
              3.80, 0.256, 0.125, 1.07]

if __name__ == "__main__":
    Ne = 7300
    u = 1.44e-8
    r = 2e-8
    
    # create the DemoGraph
    dg = demography.DemoGraph(ooa(ooa_params))
    pop_ids = ['YRI','CEU','CHB']

    fs = dg.SFS(pop_ids, [30,30,30], engine='moments', theta=4*Ne*u)
    
    rhos = np.linspace(0,10,21)
    ld = dg.LD(pop_ids=pop_ids, rho=rhos, theta=4*Ne*u)
    
    # simulate using msprime
    ts = dg.simulate_msprime(Ne=Ne, pop_ids=pop_ids, sample_sizes=[30,30,30],
                             sequence_length=1e5, recombination_rate=r,
                             mutation_rate=u, replicates=10)
#    spectra = []
#    for tree_seq in ts:
#        spectra.append(tree_seq.allele_frequency_spectrum())
    
    # plot some stuff
    fig = plt.figure(1, figsize = (12, 5))
    
    ax1 = plt.subplot(2,3,1)
    demography.plotting.plot_graph(dg, leaf_order=pop_ids, ax=ax1)
    
    ax2 = plt.subplot(2,3,2)
    # plot the demography
    
    ax3 = plt.subplot(2,3,3)
    # plot some frequency spectra
    
    ax4 = plt.subplot(2,3,4)
    # plot some LD curves

    fig.tight_layout()
    plt.show()
