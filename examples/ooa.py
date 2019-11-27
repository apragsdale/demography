"""
Create a plot of the Out-of-Africa demographic model, along with some summary
statistics, and something from some simulated data using this model.
"""
import numpy as np
from models import ooa
import matplotlib.pylab as plt
import demography

try:
    import moments as spectrum
except ImportError:
    import dadi as spectrum

import matplotlib

# Set fontsize to 8
# matplotlib fonts are all fucked up on my laptop
#plt.rcParams['pdf.fonttype'] = 42
#plt.rcParams['font.family'] = 'Calibri'

matplotlib.rc('font',**{'pdf.fonttype':42,
                        'family':'sans-serif',
                        'sans-serif':['Calibri'],
                        'style':'normal',
                        'size':8 })
# Set label tick sizes to 7
matplotlib.rc('xtick', labelsize=7)
matplotlib.rc('ytick', labelsize=7)


ooa_params = [2.11, 0.377, 0.251, 0.111, 0.224, 3.02, 0.0904, 5.77, 0.0711, 
              3.80, 0.256, 0.125, 1.07]

if __name__ == "__main__":
    Ne = 7300
    u = 1.44e-8
    r = 2e-8
    L = 1e6
    reps = 10
    
    # create the DemoGraph
    dg = demography.DemoGraph(ooa(ooa_params))
    pop_ids = ['YRI','CEU','CHB']

    fs = dg.SFS(pop_ids, [30,30,30], engine='moments', theta=4*Ne*u)
    fs *= L*reps
    
    rhos = np.linspace(0,10,21)
    ld = dg.LD(pop_ids=pop_ids, rho=rhos, theta=4*Ne*u)
    
    # simulate using msprime
    ts = dg.simulate_msprime(Ne=Ne, pop_ids=pop_ids, sample_sizes=[30,30,30],
                             sequence_length=L, recombination_rate=r,
                             mutation_rate=u, replicates=reps)
    
    ts_fs = np.zeros((31,31,31))
    for tree_seq in ts:
        sample_sets = [tree_seq.samples()[:30], tree_seq.samples()[30:60],
                       tree_seq.samples()[60:90]]
        ts_fs += tree_seq.allele_frequency_spectrum(
                        polarised=True, sample_sets=sample_sets,
                        span_normalise=False)
    
    ts_fs = spectrum.Spectrum(ts_fs, pop_ids=pop_ids)
    
    # plot some stuff
    fig = plt.figure(1, figsize = (10,6))
    fig.clf()
    
    ax1 = plt.subplot(2,3,1)
    demography.plotting.plot_graph(dg, leaf_order=pop_ids, ax=ax1)
    
    ax2 = plt.subplot(2,3,4)
    # plot the demography
    demography.plotting.plot_demography(dg, leaf_order=pop_ids, ax=ax2,
                                        flipped=['CEU'], stacked=[('A','YRI')])
    
    ax2.axis('off')
    
    # allele frequency spectra
    ax3 = plt.subplot(2,3,2)
    
    # plot the YRI frequency spectrum, simulated data vs expected
    ax3.plot(ts_fs.marginalize([1,2]), '--', label='Simulation')
    ax3.plot(fs.marginalize([1,2]), '-', lw=1, label='Expected')
    
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    
    ax3.set_xlabel('Allele frequency')
    ax3.set_ylabel('Count')
    
    ax3.legend(frameon=False)
    
    ax4 = plt.subplot(2,3,3)
    
    # plot the YRI frequency spectrum, simulated data vs expected
    ax4.plot(ts_fs.marginalize([0,2]), '--', label='CEU (sim)')
    ax4.plot(fs.marginalize([0,2]), '-', lw=1, label='CEU (exp)')
    ax4.plot(ts_fs.marginalize([0,1]), '--', label='CHB (sim)')
    ax4.plot(fs.marginalize([0,1]), '-', lw=1, label='CHB (exp)')

    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.legend(frameon=False)
    
    ax4.set_xlabel('Allele frequency')
    ax4.set_ylabel('Count')

    # ld curves
    ax5 = plt.subplot(2,3,5)
    
    for ii,pop in enumerate(pop_ids):
        stat = 'DD_{0}_{0}'.format(ii+1)
        stat_ind = ld.names()[0].index(stat)
        y = [ld[i][stat_ind] for i in range(len(rhos))]
        if ii == 0: label = r'$D^2_{YRI}$'
        if ii == 1: label = r'$D^2_{CEU}$'
        if ii == 2: label = r'$D^2_{CHB}$'
        ax5.plot(rhos, y, label=label)
    
    ax5.set_ylabel(r'$E[D^2]$')
    ax5.set_xlabel(r'$\rho$')
    
    ax5.legend(frameon = False)
    
    ax6 = plt.subplot(2,3,6)

    for ii,jj in [(0,1),(0,2),(1,2)]:
        stat = 'DD_{0}_{1}'.format(ii+1, jj+1)
        stat_ind = ld.names()[0].index(stat)
        y = [ld[i][stat_ind] for i in range(len(rhos))]
        if (ii,jj) == (0,1): label = r'$D_{YRI}D_{CEU}$'
        if (ii,jj) == (0,2): label = r'$D_{YRI}D_{CHB}$'
        if (ii,jj) == (1,2): label = r'$D_{CEU}D_{CHB}$'
        ax6.plot(rhos, y, label=label)
    
    ax6.set_ylabel(r'$E[D_i D_j]$')
    ax6.set_xlabel(r'$\rho$')

    ax6.legend(frameon = False)

    fig.tight_layout()
    plt.savefig('ooa.pdf')
    plt.show()
