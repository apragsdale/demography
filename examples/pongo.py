"""
Orangutan models.
"""


import networkx as nx
import numpy as np
import demography


def initialize_graph():
    G = nx.DiGraph()
    G.add_node('root', nu=1, T=0)
    return G


"""

"""

def locke_IM(params=None, Ne=17934):
    if params is None:
        (nuB0, nuBF, nuS0, nuSF, T, mBS, mSB) = ( 0.592, 0.491, 0.408, 2.1, 
            0.562, 0.239, 0.395)  # NOQA
    else:
        (nuB0, nuBF, nuS0, nuSF, T, mBS, mSB) = params

    G = initialize_graph()
    G.add_node('Bornean', nu0=nuB0, nuF=nuBF, T=T, m={'Sumatran': mBS})
    G.add_node('Sumatran', nu0=nuS0, nuF=nuSF, T=T, m={'Bornean': mSB})
    edges = [('root','Bornean'), ('root','Sumatran')]
    G.add_edges_from(edges)
    dg = demography.DemoGraph(G, Ne=Ne)
    return dg

