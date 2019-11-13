"""
Tests for creating DemoGraph object, and that we catch errors in input topologies.
"""
import unittest
import numpy as np
import networkx as nx
import demography
from demography.demo_class import InvalidGraph


import moments.LD




def test_graph():
    G = nx.DiGraph()
    G.add_node('root', nu=1, T=0)
    G.add_node('A', nu=2., T=.5)
    G.add_node('B', nu=1.5, T=.2)
    G.add_node('C', nu=1, T=.1)
    G.add_node('D', nu=1, T=.1)
    G.add_node('pop1', nu=1, T=.05)
    G.add_node('pop2', nu0=.5, nuF=3, T=.15)
    G.add_edges_from([('root','A'),('A','B'),('A','C'),('C','D'),('C','pop2')])
    G.add_weighted_edges_from([('B','pop1',0.7), ('D','pop1',0.3)])
    return G

def test_moments_ld(theta=0.001, rho=None):
    y = moments.LD.snm(rho=rho, theta=theta)
    return y

class TestMomentsLDIntegration(unittest.TestCase):
    """
    Tests parsing the DemoGraph object to pass to moments.LD
    """
    def returns_expected_ld_curves(self):
        G = test_graph()
        dg = demography.DemoGraph(G)
        Y = dg.LD(theta=0.001, rho=1.0, pop_ids=['pop1','pop2'])
        
