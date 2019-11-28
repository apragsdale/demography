"""
Homo sapiens models.
For each model, parameters may be passed. If None are passed, it uses the
default parameters given in each model's documentation, which match the
published parameters from the original cited study.

Note that all sizes are given relative to the ancestral or reference size, so
nu=0.5 would mean that the population is 1/2 the size of the ancestral
population. Time is given in units of 2N_ref generations.

Each function returns a DemoGraph object. Ne is set to the effective population
size inferred or assumed in the original model.
"""

import networkx as nx
import numpy as np
import demography


def initialize_graph():
    G = nx.DiGraph()
    G.add_node('root', nu=1, T=0)
    return G


"""
The Gutenkunst et al (2009) Out of Africa model.
Default parameters (with Ne=7300):
    nuA = 1.685
    TA = 0.219
    nuB = 0.288
    TB = 0.325
    nuEu0 = 0.137
    nuEuF = 4.07
    nuAs0 = 0.0699
    nuAsF = 7.41
    TF = 0.0581
    mAfB = 3.65
    mAfEu = 0.438
    mAfAs = 0.277
    mEuAs = 1.40
"""


def ooa_gutenkunst(params=None, Ne=7300):
    if params is None:
        (nuA, TA, nuB, TB, nuEu0, nuEuF, nuAs0, nuAsF, TF, mAfB, mAfEu, mAfAs,
            mEuAs) = (1.685, 0.219, 0.288, 0.325, 0.137, 4.07, 0.0699, 7.41,
                      0.0581, 3.65, 0.438, 0.277, 1.40)
    else:
        (nuA, TA, nuB, TB, nuEu0, nuEuF, nuAs0, nuAsF, TF, mAfB, mAfEu, mAfAs,
            mEuAs) = params

    G = initialize_graph()
    G.add_node('A', nu=nuA, T=TA)
    G.add_node('B', nu=nuB, T=TB,
               m={'YRI': mAfB})
    G.add_node('YRI', nu=nuA, T=TB+TF,
               m={'B': mAfB, 'CEU': mAfEu, 'CHB': mAfAs})
    G.add_node('CEU', nu0=nuEu0, nuF=nuEuF, T=TF,
               m={'YRI': mAfEu, 'CHB': mEuAs})
    G.add_node('CHB', nu0=nuAs0, nuF=nuAsF, T=TF,
               m={'YRI': mAfAs, 'CEU': mEuAs})
    edges = [('root', 'A'), ('A', 'B'), ('A', 'YRI'), ('B', 'CEU'),
             ('B', 'CHB')]
    G.add_edges_from(edges)
    dg =  demography.DemoGraph(G)
    dg.Ne = Ne
    return dg


"""
The Tennessen et al (2013) 2 population model.
Default parameters (with Ne=7310):
    nuAf0 = 1.98
    nuAfF = 59.1
    nuB = 0.255
    nuEu0 = 0.141
    nuEu1 = 0.678
    nuEu2 = 36.7
    mAfB = 2.19
    mAfEu = 0.366
    TAf = 0.265
    TOOA = 0.0766
    TEu1 = 0.0490
    TEu2 = 0.0140
"""


def ooa_tennessen(params=None, Ne=7310):
    if params is None:
        (nuAf0, nuAfF, nuB, nuEu0, nuEu1, nuEu2, mAfB, mAfEu, TAf, TOOA, TEu1,
            TEu2) = (1.98, 59.1, 0.255, 0.141, 0.678, 36.7, 2.19, 0.366, 0.265,
                     0.0766, 0.0490, 0.0140)
    else:
        (nuA, TA, nuB, TB, nuEu0, nuEuF, nuAs0, nuAsF, TF, mAfB, mAfEu, mAfAs,
            mEuAs) = params

    G = initialize_graph()
    G.add_node('Af0', nu=nuAf0, T=TAf)
    G.add_node('B', nu=nuB, T=TOOA,
               m={'Af1': mAfB})
    G.add_node('Af1', nu=nuAf0, T=TOOA+TEu1,
               m={'B': mAfB, 'Eu1': mAfB})
    G.add_node('Eu1', nu0=nuEu0, nuF=nuEu1, T=TEu1,
               m={'YRI': mAfB})
    G.add_node('YRI', nu0=nuAf0, nuF=nuAfF, T=TEu2,
               m={'CEU': mAfB})
    G.add_node('CEU', nu0=nuEu1, nuF=nuEu2, T=TEu2,
               m={'YRI': mAfB})
    edges = [('root', 'Af0'), ('Af0', 'B'), ('Af0', 'Af1'), ('Af1', 'YRI'),
             ('B', 'Eu1'), ('Eu1', 'CEU')]
    G.add_edges_from(edges)
    dg =  demography.DemoGraph(G)
    dg.Ne = Ne
    return dg


"""
Browning American model.
"""


"""
Ragsdale Archaic admixture model.
"""


"""
Kamm Basal European model.
"""
