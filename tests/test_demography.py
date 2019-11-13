"""
Tests for creating DemoGraph object, and that we catch errors in input topologies.
"""
import unittest
import numpy as np
import networkx as nx
import demography

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

def example_graph_multiple_roots():
    G = nx.DiGraph()
    
class TestGraphStructure(unittest.TestCase):
    """
    Tests the Demography class functions extract the correct topology and catch errors
    in defined demography objects.
    """
    
    def test_finds_root(self):
        G = example_graph_simple()
        dg = demography.DemoGraph(G)
        self.assertEqual(dg.root, 'root')

    #def test_single_root():
    #    pass

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
    
    #def test_catch_merger_sum():
    #    G1 = example_graph_merger(.2, .8)
    #    G2 = example_graph_merger(.3, .4)
        # how do you check if a function catches an error and raises an Exception?
    #    pass

    #def test_samples_only_from_leaves():
    #    pass

    #def 


suite = unittest.TestLoader().loadTestsFromTestCase(TestGraphStructure)

if __name__ == '__main__':
    unittest.main()
