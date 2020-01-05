"""

"""
import unittest
import numpy as np
import networkx as nx
import demography
from demography.util import InvalidGraph

import msprime

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
    

class TestCoalFunctions(unittest.TestCase):
    """
    Tests parsing the DemoGraph object to pass to moments.LD
    """
    def test_pop_sizes(self):
        N0 = 10000
        r = 0.001
        gen = 0
        Nt = demography.coalescence_rates.drift_rate_func(N0, r, gen)
        self.assertTrue(Nt(0) == N0)
        self.assertTrue(Nt(100) == N0 * np.exp(-r * 100))
    
    def test_initial_sizes(self):
        Ne = 7310
        dg = demography.DemoGraph(ooa(), Ne=Ne)
        (pop_config, mig_mat, demo_events) = dg.msprime_inputs()
        N_t = demography.coalescence_rates.pop_sizes(pop_config, demo_events)
        self.assertTrue(N_t[0][5](0) == Ne * 5.77)
        self.assertTrue(N_t[0][0](0) == Ne)
    
    def test_migration_matrix(self):
        Ne = 7310
        dg = demography.DemoGraph(ooa(), Ne=Ne)
        (pop_config, mig_mat, demo_events) = dg.msprime_inputs()
        ms = demography.coalescence_rates.migration_matrices(pop_config,
                mig_mat, demo_events)
        self.assertTrue(np.all(ms[max(ms.keys())] == 0))
    
    def test_constant_single_pop(self):
        pop_config = [msprime.PopulationConfiguration(initial_size=10000)]
        mig_mat = [[0]]
        demo_events = []
        gens = 100
        rate = demography.coalescence_rates.get_rates(0, 0, pop_config,
                    mig_mat, demo_events, gens)
        self.assertTrue(np.all(rate == 1./2/10000))
    
    def test_exp_single_pop(self):
        pop_config = [msprime.PopulationConfiguration(
                        initial_size=10000,
                        growth_rate=0.001)]
        mig_mat = [[0]]
        demo_events = []
        gens = 100
        rate = demography.coalescence_rates.get_rates(0, 0, pop_config,
                    mig_mat, demo_events, gens)
        ts = np.linspace(0,99,100)
        self.assertTrue(np.allclose(rate, 1./2/10000 * np.exp(0.001*ts)))
    
    def test_pulse_events(self):
        pop_config = [msprime.PopulationConfiguration(initial_size=1000),
                      msprime.PopulationConfiguration(initial_size=1000)]
        demo_events = [msprime.MassMigration(source=1, dest=0, proportion=1,
                                             time=100)]
        pulses = demography.coalescence_rates.pulse_events(pop_config,
                                                           demo_events)
        self.assertTrue(len(pulses.keys()) == 1)
        self.assertTrue(list(pulses.keys())[0] == 100)
        ys = [np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])]
        for y in ys:
            self.assertTrue(np.all(pulses[100][0].dot(y) == np.array([1,0,0])))
    
    def test_all_lineages_moved(self):
        y = np.ones(21)
        Ne = 7310
        dg = demography.DemoGraph(ooa(), Ne=Ne)
        (pop_config, mig_mat, demo_events) = dg.msprime_inputs()
        pulses = demography.coalescence_rates.pulse_events(pop_config,
                                                           demo_events)
        for gen in sorted(pulses.keys()):
            for pulse in pulses[gen]:
                y = pulse.dot(y)
        self.assertTrue(y[0] == 21)
        self.assertTrue(np.all(y[1:] == 0))
    
    def test_migration_matrix_sum(self):
        Ne = 7310
        dg = demography.DemoGraph(ooa(), Ne=Ne)
        (pop_config, mig_mat, demo_events) = dg.msprime_inputs()
        M = demography.coalescence_rates.get_mig_transition(mig_mat)
        self.assertTrue(np.allclose(np.sum(M, axis=0), 1))

    def test_no_coalescence_without_migration(self):
        pop_config = [msprime.PopulationConfiguration(initial_size=1000),
                      msprime.PopulationConfiguration(initial_size=1000)]
        mig_mat = [[0,0],[0,0]]
        demo_events = [msprime.MassMigration(source=1, dest=0, proportion=1,
                                             time=100)]
        pulses = demography.coalescence_rates.pulse_events(pop_config,
                                                           demo_events)
        rates = demography.coalescence_rates.get_rates(0, 1, pop_config,
                    mig_mat, demo_events, 200)
        self.assertTrue(np.all(rates[:100] == 0))
        self.assertTrue(np.all(rates[100:] == 0.0005))
    
    def test_coalescence_with_migration(self):
        pop_config = [msprime.PopulationConfiguration(initial_size=1000),
                      msprime.PopulationConfiguration(initial_size=1000)]
        mig_mat = [[0,0.001],[0.001,0]]
        demo_events = [msprime.MassMigration(source=1, dest=0, proportion=1,
                                             time=100),
                       msprime.MigrationRateChange(rate=0, time=100)]
        pulses = demography.coalescence_rates.pulse_events(pop_config,
                                                           demo_events)
        rates = demography.coalescence_rates.get_rates(0, 1, pop_config,
                    mig_mat, demo_events, 200)
        self.assertTrue(np.all(rates >= 0))
        self.assertTrue(np.all(rates[:-1]-rates[1:] <= 1e-16))
        
        
suite = unittest.TestLoader().loadTestsFromTestCase(TestCoalFunctions)

if __name__ == '__main__':
    unittest.main()
