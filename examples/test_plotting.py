import demography
import networkx as nx
import numpy as np
import matplotlib.pylab as plt

def example_multiple_mergers():
    G = nx.DiGraph()
    G.add_node('Ancestral', nu=1, T=0)
    G.add_node('Early 1', nu=1, T=0.1)
    G.add_node('Early 2', nu=1, T=0.1)
    G.add_node('Pop A', nu=2, T=0.1, pulse={('Pop B',.5,.05)})
    G.add_node('Pop B', nu=1, T=0.2)
    G.add_node('Pre', nu=2, T=0.1)
    G.add_node('Left', nu=1, T=0.3)
    G.add_node('Right', nu=1, T=0.1)
    G.add_edges_from([('Ancestral','Early 1'),('Ancestral','Early 2'),('Early 1','Left'),('Early 2','Pop B'),('Pop A','Pre')])
    G.add_weighted_edges_from([('Early 1','Pop A',.25),('Early 2','Pop A',.75),
        ('Pre','Right',.1),('Pop B','Right',.9)])
    return G



def complicated_merger_model():
    G = nx.DiGraph()
    G.add_node('root', nu=1, T=0)
    G.add_node('A', nu=2, T=.2)
    G.add_node('B', nu=1.5, T=.1)
    G.add_node('C', nu=.5, T=.2)
    G.add_node('D', nu=2., T=.1)
    G.add_node('E', nu=.2, T=.1)
    G.add_node('F', nu=1, T=.1)
    G.add_node('G', nu=2, T=.2)
    G.add_node('H', nu=.3, T=.1)
    G.add_node('I', nu=.2, T=.05)
    G.add_node('J', nu=.1, T=.05)
    G.add_node('K', nu=.5, T=.15, pulse={('G', .5, .2)}) # halfway down K branch, 0.2 pulse to G
    G.add_node('L', nu=1, T=.15)
    G.add_node('pop1', nu=.2, T=.1)
    G.add_node('pop2', nu=.2, T=.05)
    G.add_node('pop3', nu=.2, T=.1)
    G.add_node('pop4', nu=.2, T=.15)
    
    G.add_edges_from([('root','A'), ('root','B'), ('B','C'), ('B','D'), ('A','E'),
        ('E','F'), ('E','G'), ('F','H'), ('G','J'), ('G','pop3'), ('H','pop1'),
        ('H','I'), ('C','K'), ('C','L')])
    G.add_weighted_edges_from([('D','E',.2), ('A','E',.8)])
    G.add_weighted_edges_from([('I','pop2',.4), ('J','pop2',.6)])
    G.add_weighted_edges_from([('K','pop4',.5), ('L','pop4',.5)])
    return G


"""
Out of Africa example
"""

def ooa(params):
    """
    The 13 parameter out of Africa model from Gutenkunst et al. (2009)
    """
    (nuA, TA, nuB, TB, nuEu0, nuEuF, nuAs0,
        nuAsF, TF, mAfB, mAfEu, mAfAs, mEuAs) = params
    
    G  = nx.DiGraph()
    G.add_node('root', nu=1, T=0)
    G.add_node('A', nu=nuA, T=TA)
    G.add_node('B', nu=nuB, T=TB, m={'YRI':mAfB})
    G.add_node('YRI', nu=nuA, T=TB+TF, m={'B':mAfB, 'CEU':mAfEu, 'CHB': mAfAs})
    G.add_node('CEU', nu0=nuEu0, nuF=nuAsF, T=TF, m={'YRI':mAfEu, 'CHB':mEuAs})
    G.add_node('CHB', nu0=nuAs0, nuF=nuAsF, T=TF, m={'YRI':mAfAs, 'CEU':mEuAs})
    
    G.add_edges_from([('root','A'), ('A','B'), ('A','YRI'), ('B','CEU'),
                      ('B','CHB')])
    
    return G

def kamm_model():
    G = nx.DiGraph()


if __name__ == "__main__":
    fig = plt.figure(1, figsize=(10,5))
    fig.clf()
    
    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(1,2,2)

    G = example_multiple_mergers()
    dg = demography.DemoGraph(G)

    demography.plotting.plot_graph(dg, leaf_order=['Left', 'Right'],
                                   leaf_locs=[.2, .7], ax=ax1, padding=.2, offset=.05)

    G2 = complicated_merger_model()
    dg2 = demography.DemoGraph(G2)

    demography.plotting.plot_graph(dg2, leaf_order=['pop1','pop2','pop3','pop4'], 
                                   leaf_locs=[.2, .4, .6, .8], ax=ax2, offset=0.02)

    fig.tight_layout()
    #plt.show()

    # gutenkunst out of africa parameters
    params = [2.11, 0.377, 0.251, 0.111, 0.0904, 5.77, 0.224, 3.02, 0.0711, 
              3.80, 0.256, 0.125, 1.07]
    G = ooa(params) 
    dg = demography.DemoGraph(G, Ne=8880, mutation_rate=1.44e-8)
    
    fig2 = plt.figure(2, figsize=(10,5))
    fig2.clf()
    
    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(1,2,2)
    
    demography.plotting.plot_graph(dg, leaf_order=['YRI','CEU','CHB'],
                                   leaf_locs=[.25,.5,.75], ax=ax1)
    
    demography.plotting.plot_demography(dg, leaf_order=['YRI','CEU','CHB'],
                                        ax=ax2, padding=0.1)
    
    fig2.tight_layout()
    plt.show()


