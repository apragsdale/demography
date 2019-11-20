"""
Tests for creating DemoGraph object, and that we catch errors in input topologies.
"""
import unittest
import numpy as np
import networkx as nx
import demography
from demography.util import InvalidGraph

import msprime

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

class TestMsprimeFunctions(unittest.TestCase):
    """
    Tests parsing the DemoGraph object to pass to moments.LD
    """
    def test_get_growth_rates(self):
        G = test_graph()
        dg = demography.DemoGraph(G)
        Ne = 1e4
        growth_rates = demography.msprime_functions.get_population_growth_rates(dg, Ne)
        self.assertTrue(growth_rates['root'] == 0)
        self.assertTrue(growth_rates['A'] == 0)
        self.assertTrue(growth_rates['C'] == 0)
        self.assertTrue(growth_rates['pop1'] == 0)
        self.assertTrue(growth_rates['pop2'] != 0)
        self.assertTrue(np.isclose(3, 0.5*np.exp(growth_rates['pop2'] * 2*Ne*dg.G.nodes['pop2']['T'])))

    def test_pop_configs(self):
        G = test_graph()
        dg = demography.DemoGraph(G)
        Ne = 1e4
        growth_rates = demography.msprime_functions.get_population_growth_rates(dg, Ne)
        population_configurations, pop_indexes = demography.msprime_functions.get_population_configurations(dg, ['pop1','pop2'], growth_rates, Ne)
        self.assertTrue(population_configurations[pop_indexes['root']].initial_size == Ne)
        self.assertTrue(population_configurations[pop_indexes['pop1']].initial_size == Ne)
        self.assertTrue(population_configurations[pop_indexes['pop2']].initial_size == 3.0*Ne)
        self.assertTrue(population_configurations[pop_indexes['pop2']].growth_rate == growth_rates['pop2'])

    def test_migration_matrix(self):
        m12 = 3
        m21 = 7
        G = nx.DiGraph()
        G.add_node('root', nu=1, T=0)
        G.add_node('A', nu=1, T=0.3, m={'B':m12})
        G.add_node('B', nu=1, T=0.3, m={'A':m21})
        G.add_edges_from([('root','A'),('root','B')])
        dg = demography.DemoGraph(G)
        Ne = 1e4
        growth_rates = demography.msprime_functions.get_population_growth_rates(dg, Ne)
        population_configurations, pop_indexes = demography.msprime_functions.get_population_configurations(dg, ['A','B'], growth_rates, Ne)
        mig_mat = demography.msprime_functions.get_migration_matrix(dg, ['A','B'], pop_indexes, Ne)
        self.assertTrue(mig_mat[pop_indexes['A']][pop_indexes['B']] == m12 / 2 / Ne)
        self.assertTrue(mig_mat[pop_indexes['B']][pop_indexes['A']] == m21 / 2 / Ne)
        self.assertTrue(np.array(mig_mat).shape == (3,3))
        self.assertTrue(len(np.where(np.array(mig_mat) != 0)[0]) == 2)

    def test_msprime_from_graph(self):
        G = test_graph()
        dg = demography.DemoGraph(G)
        dg.Ne = 1e4
        pc, mm, de = dg.msprime_inputs()
        self.assertTrue(len(pc) == 7)
        self.assertTrue(len(mm) == 7)
    
    def test_get_samples(self):
        G = test_graph()
        dg = demography.DemoGraph(G)
        pop_ids = ['pop1','pop2']
        ns = [10,20]
        samples = dg.msprime_samples(pop_ids, ns)
        self.assertTrue(len(samples) == np.sum(ns))

suite = unittest.TestLoader().loadTestsFromTestCase(TestMsprimeFunctions)

if __name__ == '__main__':
    unittest.main()
