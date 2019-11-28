"""
Generic demographic models.
For each model, parameters may be passed. If None are passed, it uses the
default parameters given in each model's documentation.

Note that all sizes are given relative to the ancestral or reference size, so
nu=0.5 would mean that the population is 1/2 the size of the ancestral
population. Time is given in units of 2N_ref generations.

Each function returns a DemoGraph object.
"""

import networkx as nx
import numpy as np
import demography


def initialize_graph():
    G = nx.DiGraph()
    G.add_node('root', nu=1, T=0)
    return G


"""
Three epoch model, each epoch of constant size.
Default parameters: nuB=0.5, TB=0.2, nuF=3.0, TF=0.05.
"""


def three_epoch(params=None):
    if params is None:
        (nuB, TB, nuF, TF) = (0.5, 0.2, 3.0, 0.05)
    else:
        (nuB, TB, nuF, TF) = params

    G = initialize_graph()
    G.add_node('pop_B', nu=nuB, T=TB)
    G.add_node('pop', nu=nuF, T=TF)
    edges = [('root', 'pop_B'), ('pop_B', 'B')]
    G.add_edges_from(edges)
    return demography.DemoGraph(G)


"""
Bottleneck followed by exponential growth.
Default parameters: nuB=0.5, TB=0.2, nuF=3.0, TF=0.5.
Similar to the three_epoch model, except that in the final epoch, the growth
is exponential.
"""


def bottleneck_growth(params=None):
    if params is None:
        (nuB, TB, nuF, TF) = (0.5, 0.2, 3.0, 0.05)
    else:
        (nuB, TB, nuF, TF) = params

    G = initialize_graph()
    G.add_node('pop_B', nu=nuB, T=TB)
    G.add_node('pop', nu0=nuB, nuF=nuF, T=TF)
    edges = [('root', 'pop_B'), ('pop_B', 'B')]
    G.add_edges_from(edges)
    return demography.DemoGraph(G)


"""
Two-population split with migration.
Default parameters: nuA=1.0, nu1=2.0, nu2=3.0, T=0.2, m12=0.5, m21=2.0
"""


def split_IM(params=None):
    if params is None:
        (nuA, nu1, nu2, T, m12, m21) = (1.0, 2.0, 3.0, 0.2, 0.5, 2.0)
    else:
        (nuA, nu1, nu2, T, m12, m21) = params

    G = initialize_graph()
    G.nodes['root']['nu'] = nuA
    G.add_node('pop1', nu=nu1, T=T, m={'pop2': m12})
    G.add_node('pop2', nu=nu2, T=T, m={'pop1': m21})
    edges = [('root', 'pop1'), ('root', 'pop2')]
    G.add_edges_from(edges)
    return demography.DemoGraph(G)
