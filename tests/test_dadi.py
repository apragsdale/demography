"""
Tests for creating DemoGraph object, and that we catch errors in input topologies.
"""
import unittest
import numpy as np
import networkx as nx
import demography
from demography.util import InvalidGraph

import dadi

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

class TestMomentsIntegration(unittest.TestCase):
    """
    Tests parsing the DemoGraph object to pass to moments.LD
    """

    def test_dadi_grid(self):
        pts = 100
        xx = demography.integration.get_dadi_grid(pts)
        self.assertTrue(len(xx) == pts)
        pts = [40,50,60]
        xx = demography.integration.get_dadi_grid(pts)
        self.assertTrue(len(xx) == 3)
        self.assertTrue(len(xx[0]) == pts[0])

    def test_dadi_equilibrium(self):
        pts = [100,120,140]
        grid = demography.integration.get_dadi_grid(pts)
        phi = demography.integration.dadi_root_equilibrium(pts, grid,
                                nu=1.0, theta=1.0, gamma=0.0, h=0.5)
        fs = demography.integration.sample_dadi(phi, grid, pts, [20])
        self.assertTrue(np.allclose(fs[1:-1],1./np.linspace(1,19,19), 0.005))

    def test_check_pts_sample_size(self):
        G = test_graph()
        dg = demography.DemoGraph(G)
        self.assertRaises(AssertionError, dg.SFS, ['pop1','pop2'], [30,30], pts=20, engine='dadi')

    def test_dadi_pass(self):
        G = nx.DiGraph()
        G.add_node('root', nu=1, T=0)
        G.add_node('pop1', nu=1, T=0.0)
        G.add_edge('root', 'pop1')
        dg = demography.DemoGraph(G)
        fs = dg.SFS(['pop1'], [10], engine='dadi', pts=30)
        self.assertTrue(len(fs) == 11)
        fs = dg.SFS(['pop1'], [10], engine='dadi', pts=[50,60,70])
        self.assertTrue(np.allclose(fs[1:-1], 1./np.linspace(1,9,9), 0.005))

#        G = test_graph()
#        dg = demography.DemoGraph(G)
#        (present_pops, integration_times, nus, migration_matrices, frozen_pops, 
#            selfing_rates, events) = demography.integration.get_moments_arguments(dg)

suite = unittest.TestLoader().loadTestsFromTestCase(TestMomentsIntegration)

if __name__ == '__main__':
    unittest.main()
