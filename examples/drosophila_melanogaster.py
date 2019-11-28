"""
Drosophila melanogaster models.
"""

import networkx as nx
import numpy as np
import demography


def initialize_graph():
    G = nx.DiGraph()
    G.add_node('root', nu=1, T=0)
    return G


def sheehan_song_three_epoch(params=None, Ne=652700):
    if params == None:
        (nuB, TB, nuF, TF) = (0.223, 1.53, 0.834, 0.153)
    else:
        (nuB, TB, nuF, TF) = params

    G = initialize_graph()
    G.add_node('bottle', nu=nuB, T=TB)
    G.add_node('D. mel', nu=nuF, T=TF)
    edges = [('root', 'bottle'), ('bottle', 'D. mel')]
    G.add_edges_from(edges)
    dg = demography.Demograph(G, Ne=Ne)
    return dg


def li_stephan(params=None, Ne=1720600):
    if params == None:
        (nuA0, nuE1, nuE0, TA, TE1, TE2) = (5, 0.00128, 0.625, 0.128,
                                            0.000988, 0.0449)
    else:
        (nuA0, nuE1, nuE0, TA, TE1, TE2) = params

    G = initialize_graph()
    G.add_node('A0', nu=nuA0, T=TA)
    G.add_node('E_bottle', nu=nuE1, T=TE1)
    G.add_node('AFR', nu=nuA0, T=TE1+TE2)
    G.add_node('EUR', nu=nuE0, T=TE2)
    edges = [('root', 'A0'), ('A0', 'AFR'), ('A0', 'E_bottle'),
             ('E_bottle', 'EUR')]
    G.add_edges_from(edges)
    dg = demography.DemoGraph(G, Ne=Ne)
    return dg
