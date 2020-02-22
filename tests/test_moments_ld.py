"""
Tests for creating DemoGraph object, and that we catch errors in input topologies.
"""
import unittest
import numpy as np
import networkx as nx
import demography
from demography.util import InvalidGraph

import moments.LD as mold

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
    dg = demography.DemoGraph(G)
    return dg

def simple_graph():
    G = nx.DiGraph()
    G.add_node('root', nu=1, T=0)
    G.add_node('pop1', nu=1, T=.5)
    G.add_node('pop2', nu=1, T=.3)
    G.add_edges_from([('root','pop1'),('root','pop2')])
    dg = demography.DemoGraph(G)
    return dg

class TestMomentsLD(unittest.TestCase):
    """
    Tests parsing the DemoGraph object to pass to moments.LD
    """
    def test_expected_ld_curve(self):
        dg = test_graph()
        Y = dg.LD(theta=0.001, rho=1.0, pop_ids=['pop1','pop2'])
        Ymold = mold.Demography.evolve(dg.G, theta=0.001, rho=1.0, pop_ids=['pop1','pop2'])
        self.assertTrue(np.allclose(Y[0], Ymold[0]))
        self.assertTrue(np.allclose(Y[1], Ymold[1]))

    def test_frozen_pops_ld(self):
        dg = simple_graph()
        y = dg.LD(rho=1.0, theta=0.001, pop_ids=['pop1','pop2'])
        y2 = mold.Demographics1D.snm(rho=1.0, theta=0.001)
        y2 = y2.split(1)
        y2.integrate([1,1], 0.3, rho=1., theta=0.001)
        y2.integrate([1,1], 0.2, rho=1., theta=0.001, frozen=[False, True])
        self.assertTrue(np.allclose(y[0], y2[0]))
        self.assertTrue(np.allclose(y[1], y2[1]))
    
suite = unittest.TestLoader().loadTestsFromTestCase(TestMomentsLD)

if __name__ == '__main__':
    unittest.main()
