"""
Tests for creating DemoGraph object, and that we catch errors in input topologies.
"""
import unittest
import numpy as np
import networkx as nx
import demography
from demography.demo_class import InvalidGraph

def example_graph_simple():
    G = nx.DiGraph()
    G.add_node('root', nu=1, T=0)
    G.add_node('A', nu=1, T=0.1)
    G.add_node('B', nu=1, T=0.2)
    G.add_node('C', nu=2, T=0.1)
    G.add_edges_from([('root','A'), ('A','B'), ('root','C')])
    return G

def example_graph_merger(f1, f2):
    G = nx.DiGraph()
    G.add_node('root', nu=1, T=0)
    G.add_node('A', nu=1, T=0.1)
    G.add_node('B', nu=1, T=0.2)
    G.add_node('C', nu=2, T=0.1)
    G.add_node('D', nu=1, T=0.05)
    G.add_edges_from([('root','A'), ('A','C'), ('root','B')])
    G.add_weighted_edges_from([('B','D',f1), ('C','D',f2)])
    return G

def example_multiple_mergers():
    G = nx.DiGraph()
    G.add_node('root', nu=1, T=0)
    G.add_node('A', nu=1, T=0.1)
    G.add_node('B', nu=1, T=0.1)
    G.add_node('C', nu=2, T=0.1)
    G.add_node('D', nu=1, T=0.2)
    G.add_node('E', nu=2, T=0.1)
    G.add_node('F', nu=1, T=0.3)
    G.add_node('G', nu=1, T=0.1)
    G.add_edges_from([('root','A'),('root','B'),('A','F'),('B','D'),('C','E')])
    G.add_weighted_edges_from([('A','C',.5),('B','C',.5),
        ('E','G',.5),('D','G',.5)])
    return G

def example_multiple_mergers_mismatch():
    G = nx.DiGraph()
    G.add_node('root', nu=1, T=0)
    G.add_node('A', nu=1, T=0.1)
    G.add_node('B', nu=1, T=0.1)
    G.add_node('C', nu=2, T=0.15)
    G.add_node('D', nu=1, T=0.2)
    G.add_node('E', nu=2, T=0.1)
    G.add_node('F', nu=1, T=0.3)
    G.add_node('G', nu=1, T=0.1)
    G.add_edges_from([('root','A'),('root','B'),('A','F'),('B','D'),('C','E')])
    G.add_weighted_edges_from([('A','C',.5),('B','C',.5),
        ('E','G',.5),('D','G',.5)])
    return G

class TestGraphStructure(unittest.TestCase):
    """
    Tests the Demography class functions extract the correct topology and catch errors
    in defined demography objects.
    """
    
    def test_finds_root(self):
        G = example_graph_simple()
        dg = demography.DemoGraph(G)
        self.assertEqual(dg.root, 'root')

    def test_single_root(self):
        G = nx.DiGraph()
        G.add_node('root1', nu=1, T=0)
        G.add_node('root2', nu=1, T=0)
        G.add_node('A', nu=1, T=.1)
        G.add_weighted_edges_from([('root1','A',.5), ('root2','A',.5)])
        self.assertRaises(InvalidGraph, demography.DemoGraph, G)

    def test_no_loops(self):
        G = nx.DiGraph()
        G.add_node('root', nu=1, T=0)
        G.add_node('A', nu=1, T=0.1)
        G.add_node('B', nu=1, T=0.2)
        G.add_node('C', nu=2, T=0.1)
        G.add_edges_from([('root','A'),('A','B'),('B','C'),('C','A')])
        self.assertRaises(InvalidGraph, demography.DemoGraph, G)

    def test_all_times_align(self):
        G = example_multiple_mergers()
        dg = demography.DemoGraph(G)
        self.assertTrue(demography.demo_class.all_merger_times_align(dg))
        G = example_multiple_mergers_mismatch()
        self.assertRaises(InvalidGraph, demography.DemoGraph, G)

    def test_theta_computation(self):
        G = example_graph_simple()
        Ne = 1e4
        u = 1e-8
        theta = 4*Ne*u
        dg = demography.DemoGraph(G, Ne=Ne, mutation_rate=u)
        self.assertTrue(np.isclose(dg.get_theta(), theta))
        dg2 = demography.DemoGraph(G, Ne=Ne, mutation_rate=u, sequence_length=5)
        self.assertTrue(np.isclose(dg2.get_theta(), theta*5))
        dg3 = demography.DemoGraph(G)
        self.assertTrue(np.isclose(dg3.get_theta(), 1))
    
    def test_catch_merger_sum(self):
        G = example_graph_merger(.2, .5)
        self.assertRaises(InvalidGraph, demography.DemoGraph, G)
    
    def test_merger_without_weights(self):
        G = nx.DiGraph()
        G.add_node('root', nu=1, T=0)
        G.add_node('A', nu=1, T=0.1)
        G.add_node('B', nu=1, T=0.2)
        G.add_node('C', nu=2, T=0.1)
        G.add_edges_from([('root','A'),('root','B'),('A','C'),('B','C')])
        self.assertRaises(InvalidGraph, demography.DemoGraph, G)
    
    def test_three_pop_merger(self):
        G = nx.DiGraph()
        G.add_node('root', nu=1, T=0)
        G.add_node('A', nu=1, T=0.1)
        G.add_node('B', nu=1, T=0.2)
        G.add_node('C', nu=2, T=0.1)
        G.add_node('D', nu=1, T=0.1)
        G.add_node('E', nu=2, T=0.1)
        G.add_edges_from([('root','A'),('root','B'),('B','C'),('B','D'),
            ('A','E'), ('C','E'), ('D','E') ])
        self.assertRaises(InvalidGraph, demography.DemoGraph, G)

    #def test_samples_only_from_leaves():
    #    pass

    #def 


suite = unittest.TestLoader().loadTestsFromTestCase(TestGraphStructure)

if __name__ == '__main__':
    unittest.main()
