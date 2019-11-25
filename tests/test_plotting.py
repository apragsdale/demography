import demography
import plotting
import networkx as nx
import numpy as np

def example_multiple_mergers():
    G = nx.DiGraph()
    G.add_node('root', nu=1, T=0)
    G.add_node('Wookie', nu=1, T=0.1)
    G.add_node('Ewok', nu=1, T=0.1)
    G.add_node('Vulcan', nu=2, T=0.1)
    G.add_node('Tardigrade', nu=1, T=0.2)
    G.add_node('Kzin', nu=2, T=0.1)
    G.add_node('T. rex', nu=1, T=0.3)
    G.add_node('Aquatic Ape', nu=1, T=0.1)
    G.add_edges_from([('root','Wookie'),('root','Ewok'),('Wookie','T. rex'),('Ewok','Tardigrade'),('Vulcan','Kzin')])
    G.add_weighted_edges_from([('Wookie','Vulcan',.25),('Ewok','Vulcan',.75),
        ('Kzin','Aquatic Ape',.1),('Tardigrade','Aquatic Ape',.9)])
    return G



G = example_multiple_mergers()
dg = demography.DemoGraph(G)

demography.plotting.plot_graph(dg, leaf_order=['T. rex','Aquatic Ape'], leaf_locs=[.2,.7], show=True)

