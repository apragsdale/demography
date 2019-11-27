import demography
import networkx as nx
import numpy as np
import matplotlib.pylab as plt
from models import example_multiple_mergers, complicated_merger_model, ooa, kamm_model

if __name__ == "__main__":
    fig1 = plt.figure(1, figsize=(10,5))
    fig1.clf()
    
    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(1,2,2)

    G = example_multiple_mergers()
    dg = demography.DemoGraph(G)

    demography.plotting.plot_graph(dg, leaf_order=['Left', 'Right'],
                                   leaf_locs=[.2, .7], ax=ax1)
                                   
    demography.plotting.plot_demography(dg, leaf_order=['Left', 'Right'], 
                                        ax=ax2, padding=4)
    fig1.tight_layout()


    fig2 = plt.figure(2, figsize=(10,5))
    fig2.clf()
    
    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(1,2,2)


    G2 = complicated_merger_model()
    dg2 = demography.DemoGraph(G2)

    demography.plotting.plot_graph(dg2, leaf_order=['pop1','pop2','pop3','pop4'], 
                                   leaf_locs=[.2, .4, .6, .8], ax=ax1)

    demography.plotting.plot_demography(dg2, leaf_order=['pop1','pop2','pop3','pop4'], 
                                        ax=ax2, padding=3)
    fig2.tight_layout()



    # gutenkunst out of africa parameters
    params = [2.11, 0.377, 0.251, 0.111, 0.224, 3.02, 0.0904, 5.77, 0.0711, 
              3.80, 0.256, 0.125, 1.07]
    G = ooa(params) 
    dg = demography.DemoGraph(G, Ne=8880, mutation_rate=1.44e-8)
    
    fig3 = plt.figure(3, figsize=(10,5))
    fig3.clf()
    
    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(1,2,2)
    
    demography.plotting.plot_graph(dg, leaf_order=['YRI','CEU','CHB'],
                                   leaf_locs=[.25,.5,.75], ax=ax1)
    
    demography.plotting.plot_demography(dg, leaf_order=['YRI','CEU','CHB'],
                                        ax=ax2, flipped=['CEU'],
                                        stacked=[('A','YRI')])
    
    fig3.tight_layout()



    
    # a busy Kamm model
    
    fig4 = plt.figure(4, figsize=(18,8))
    fig4.clf()

    G = kamm_model()
    dg = demography.DemoGraph(G)
    
    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(1,2,2)

    demography.plotting.plot_graph(dg,
                    leaf_order=['Mbuti','Basal Eur','LBK','Sardinian',
                                'Loschbour','MA1','Han','UstIshim','Neanderthal'],
                    leaf_locs=np.linspace(.1,.9,9), ax=ax1,
                    offset=0.005, buffer=0.002,
                    padding=0.1)


    demography.plotting.plot_demography(dg, 
                    leaf_order=['Mbuti','Basal Eur','LBK','Sardinian',
                                'Loschbour','MA1','Han','UstIshim','Neanderthal'],
                    flipped=['Neanderthal'], padding=2,
                    ax=ax2, stacked=[('root','Mbu_Losch'),('Mbu_Losch','Basal_Losch'),
                                     ('Basal_Losch','Ust_Losch'),('Ust_Losch','Han_Losch'),
                                     ('Han_Losch','MA1_Losch'),('MA1_Losch','LBK_Losch'),
                                     ('LBK_Losch','Loschbour'),('Neand_const','Neanderthal')])
        
    fig4.tight_layout()
    # a lot happens in the distant past, so we focous more recently
    ax1.set_ylim([0,0.15])


    # to see the difference between stacked and unstacked
    
    fig5 = plt.figure(5, figsize=(10,5))
    fig5.clf()

    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(1,2,2)

    dg = demography.DemoGraph(ooa(params))
    
    demography.plotting.plot_demography(dg, leaf_order=['YRI','CEU','CHB'],
                                        ax=ax1, flipped=['CEU'])
    
    demography.plotting.plot_demography(dg, leaf_order=['YRI','CEU','CHB'],
                                        ax=ax2, flipped=['CEU'],
                                        stacked=[('A','YRI')])
    
    fig5.tight_layout()
    
    

    plt.show()
