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

def ooa():
    """
    The 13 parameter out of Africa model from Gutenkunst et al. (2009)
    """
    params = [2.11, 0.377, 0.251, 0.111, 0.224, 3.02, 0.0904, 5.77, 0.0711, 
              3.80, 0.256, 0.125, 1.07]
    
    (nuA, TA, nuB, TB, nuEu0, nuEuF, nuAs0,
        nuAsF, TF, mAfB, mAfEu, mAfAs, mEuAs) = params
    
    G  = nx.DiGraph()
    G.add_node('root', nu=1, T=0)
    G.add_node('A', nu=nuA, T=TA)
    G.add_node('B', nu=nuB, T=TB, m={'YRI':mAfB})
    G.add_node('YRI', nu=nuA, T=TB+TF, m={'B':mAfB, 'CEU':mAfEu, 'CHB': mAfAs})
    G.add_node('CEU', nu0=nuEu0, nuF=nuEuF, T=TF, m={'YRI':mAfEu, 'CHB':mEuAs})
    G.add_node('CHB', nu0=nuAs0, nuF=nuAsF, T=TF, m={'YRI':mAfAs, 'CEU':mEuAs})
    
    G.add_edges_from([('root','A'), ('A','B'), ('A','YRI'), ('B','CEU'),
                      ('B','CHB')])
    
    return G
    
def example_three_split():
    G = nx.DiGraph()
    G.add_node('root', nu=1, T=0)
    G.add_node('pop1', nu=1, T=1)
    G.add_node('pop2', nu=1, T=1)
    G.add_node('pop3', nu=1, T=1)
    G.add_edges_from([('root','pop1'), ('root','pop2'), ('root','pop3')])
    return demography.DemoGraph(G)


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

    def test_ooa_migration(self):
        G = ooa()
        dg = demography.DemoGraph(G)
        pop_configs, mig_mat, demo_events = dg.msprime_inputs(Ne = 7300)
        # how can I test this...
    
    def test_initial_migration_matrix(self):
        G = ooa()
        dg = demography.DemoGraph(G, Ne=7310)
        pop_configs, mig_mat, demo_events = dg.msprime_inputs(Ne = 7300)
        contemp_pops = [3,4,5]
        self.assertTrue(np.all([mig_mat[i][j] == 0 for i in range(6) for j in range(6) if (i not in contemp_pops or j not in contemp_pops)]))
    
    def test_three_way_split(self):
        dg = example_three_split()
        Ne = 10000
        pop_configs, mig_mat, demo_events = dg.msprime_inputs(Ne=Ne)
        moves = [0,0,0]
        for de in demo_events:
            if de.type == 'mass_migration':
                if de.dest == 0:
                    moves[de.source-1] = 1
        self.assertTrue(np.all(moves))

suite = unittest.TestLoader().loadTestsFromTestCase(TestMsprimeFunctions)

if __name__ == '__main__':
    unittest.main()
