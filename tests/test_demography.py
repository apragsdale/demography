import unittest
import numpy as np
import networkx as nx
import demography.demo_class as demo_class

class TestGraphStructure(unnittest.TestCase):
    """
    Tests the Demography class functions extract the correct topology and catch errors
    in defined demography objects.
    """
    def test_finds_root(self):
        G = nx.DiGraph()
        G.add_node('root', nu=1, T=0)
        G.add_node('A', nu=1, T=0.1)
        G.add_node('B', nu=1, T=0.2)
        G.add_node('C', nu=2, T=0.1)
        G.add_edges_from([('root','A'), ('A','B'), ('root','C')])
        dg = demo_class(G)
        self.assertEqual(dg.root, 'root')

    def test_finds_leaves(self):
        pass

    def test_single_root():
        pass

    def test_catch_merger_sum():
        pass
