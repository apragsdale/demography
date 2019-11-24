"""
Tests for creating DemoGraph object, and that we catch errors in input topologies.
"""
import unittest
import numpy as np
import networkx as nx
import demography
from demography.util import InvalidGraph

import dadi, moments

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

    def test_multi_epoch(self):
        G = nx.DiGraph()
        G.add_node('root', nu=1, T=0)
        G.add_node('A', nu=2, T=0.3)
        G.add_node('B', nu=3, T=0.1)
        G.add_edges_from([('root', 'A'), ('A', 'B')])
        dg = demography.DemoGraph(G)
        fs = dg.SFS(['B'], [10], engine='dadi', pts=30)
        grid = dadi.Numerics.default_grid(30)
        phi = dadi.PhiManip.phi_1D(grid, nu=1.)
        phi = dadi.Integration.one_pop(phi, grid, nu=2, T=0.3)
        phi = dadi.Integration.one_pop(phi, grid, nu=3, T=0.1)
        fs2 = dadi.Spectrum.from_phi(phi, [10], (grid,))
        self.assertTrue(np.allclose(fs.data, fs2.data))

    def test_population_split_two_pop(self):
        G = nx.DiGraph()
        G.add_node('root', nu=1, T=0)
        G.add_node('A', nu=1, T=0)
        G.add_node('B', nu=1, T=0)
        G.add_edges_from([('root', 'A'), ('root', 'B')])
        dg = demography.DemoGraph(G)
        fs = dg.SFS(['A','B'], [10, 10], engine='dadi', pts=20)
        fs = dg.SFS(['A','B'], [10, 10], engine='dadi', pts=[20,25,30])
        fs = dg.SFS(['A','B'], [10, 10], engine='dadi', pts=[60,80,100])
        fs_moments = moments.Demographics2D.snm([10,10])
        self.assertTrue(np.allclose(fs_moments, fs, .02))

    def test_split(self):
        pts = 30
        grid = demography.integration.get_dadi_grid(pts)
        phi = demography.integration.dadi_root_equilibrium(pts, grid)
        phi = demography.integration.dadi_split(phi, grid, pts, 'A', 'B', 'C',
                                                ['A'], ['B','C'])
        self.assertTrue(phi.shape == (30,30))
        phi2 = demography.integration.dadi_split(phi, grid, pts, 'A', 'C', 'D',
                                                ['A','B'], ['B','C','D'])
        phi3 = demography.integration.dadi_split(phi, grid, pts, 'B', 'C', 'D',
                                                ['A','B'], ['A','C','D'])
        self.assertTrue(phi2.shape == (30,30,30))
        self.assertTrue(phi3.shape == (30,30,30))
        pts = [20,30,40]
        grid = demography.integration.get_dadi_grid(pts)
        phi = demography.integration.dadi_root_equilibrium(pts, grid)
        phi = demography.integration.dadi_split(phi, grid, pts, 'A', 'B', 'C',
                                                ['A'], ['B','C'])
        self.assertTrue(np.all([this_phi.shape == (pt,pt) for pt, this_phi in
                                zip(pts, phi)]))

    def test_marginalize(self):
        pts = 30
        grid = demography.integration.get_dadi_grid(pts)
        phi = demography.integration.dadi_root_equilibrium(pts, grid)
        phi = demography.integration.dadi_split(phi, grid, pts, 'A', 'B', 'C',
                                                ['A'], ['B','C'])
        phi = demography.integration.dadi_split(phi, grid, pts, 'C', 'D', 'E',
                                                ['B','C'], ['B','D','E'])
        phi = demography.integration.dadi_marginalize(phi, grid, pts, 'D',
                                                ['B','D','E'], ['B','E'])
        self.assertTrue(phi.ndim == 2)
        self.assertTrue(phi.shape == (30,30))

    def test_integration_with_marginalize(self):
        G = nx.DiGraph()
        G.add_node('root', nu=1, T=0)
        G.add_node('pop1', nu=2, T=.2)
        G.add_node('pop2', nu=.5, T=.1)
        G.add_edges_from([('root','pop1'), ('root','pop2')])
        dg = demography.DemoGraph(G)
        fs = dg.SFS(['pop1'], [10], engine='dadi', pts=30)
        self.assertTrue(len(fs) == 11)
        self.assertTrue(fs.pop_ids == ['pop1'])

    def test_frozen(self):
        G = nx.DiGraph()
        G.add_node('root', nu=1, T=0)
        G.add_node('pop1', nu=2, T=.2)
        G.add_node('pop2', nu=.5, T=.1)
        G.add_edges_from([('root','pop1'), ('root','pop2')])
        dg = demography.DemoGraph(G)
        fs = dg.SFS(['pop1', 'pop2'], [10,2], engine='dadi', pts=30)
        self.assertTrue(fs.ndim == 2)
        self.assertTrue(fs.pop_ids == ['pop1', 'pop2'])
        self.assertTrue(np.all(fs.sample_sizes == [10,2]))

    def test_merge_two_pops(self):
        pts = 30
        grid = demography.integration.get_dadi_grid(pts)
        phi = demography.integration.dadi_root_equilibrium(pts, grid)
        phi = demography.integration.dadi_split(phi, grid, pts, 'A', 'B', 'C',
                                                ['A'], ['B','C'])
        phi = demography.integration.dadi_merge(phi, grid, pts, ['B','C'],
                                            (.25,.75), 'D', ['B','C'], ['D'])
        self.assertTrue(phi.ndim == 1)
        self.assertTrue(len(phi) == 30)

        phi = demography.integration.dadi_root_equilibrium(pts, grid)
        phi = demography.integration.dadi_split(phi, grid, pts, 'A', 'B', 'C',
                                                ['A'], ['B','C'])
        phi = demography.integration.integrate_phis(phi, grid, pts, [3, .5],
                                                    .2, m=np.zeros((2,2)),
                                                    frozen=[False,False])
        self.assertTrue(np.allclose(dadi.PhiManip.remove_pop(phi, grid, 1), 
                        demography.integration.dadi_merge(phi, grid, pts,
                            ['B','C'], (0,1), 'D', ['B','C'], ['D'])))
        self.assertTrue(np.allclose(dadi.PhiManip.remove_pop(phi, grid, 2), 
                        demography.integration.dadi_merge(phi, grid, pts,
                            ['B','C'], (1,0), 'D', ['B','C'], ['D'])))

    def test_merge_two_pops_result(self):
        G = nx.DiGraph()
        G.add_node('root', nu=1, T=0)
        G.add_node('pop1', nu=2, T=.1)
        G.add_node('pop2', nu=.5, T=.1)
        G.add_node('pop3', nu=1, T=.1)
        G.add_edges_from([('root','pop1'), ('root','pop2')])
        G.add_weighted_edges_from([('pop1','pop3',.7), ('pop2','pop3',.3)])
        dg = demography.DemoGraph(G)
        fs_dadi = dg.SFS(['pop3'], [10], engine='dadi', pts=[60,80,100])
        fs_moments = dg.SFS(['pop3'], [10], engine='moments')
        self.assertTrue(np.allclose(fs_dadi, fs_moments, .006))


suite = unittest.TestLoader().loadTestsFromTestCase(TestMomentsIntegration)

if __name__ == '__main__':
    unittest.main()
