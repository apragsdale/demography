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
    
    dg = demography.DemoGraph(G)
    dg.Ne = 7310
    return dg


def dg_with_selfing():
    G = nx.DiGraph()
    G.add_node('root', nu=1, T=0)
    G.add_node('A', nu=1, T=1)
    G.add_node('B', nu=1, T=1, selfing=0.5)
    G.add_edges_from([('root','A'), ('root','B')])
    return demography.DemoGraph(G)


def dg_without_selfing():
    G = nx.DiGraph()
    G.add_node('root', nu=1, T=0)
    G.add_node('A', nu=1, T=1)
    G.add_node('B', nu=1, T=1)
    G.add_edges_from([('root','A'), ('root','B')])
    return demography.DemoGraph(G)


class TestTmrcaFunctions(unittest.TestCase):
    """
    Tests parsing the DemoGraph object to pass to moments.LD
    """
    def test_check_selfing(self):
        dg = dg_with_selfing()
        any_selfing = demography.tmrcas.check_selfing(dg)
        self.assertTrue(any_selfing)
        dg = dg_without_selfing()
        any_selfing = demography.tmrcas.check_selfing(dg)
        self.assertFalse(any_selfing)
    
    def test_tmrca_names(self):
        order = 2
        num_pops = 3
        tmrcas = demography.tmrcas.tmrca_vector(order, num_pops)
        self.assertTrue(len(tmrcas) == order*num_pops*(num_pops+1)//2)
        self.assertTrue(tmrcas[1] == 'T1_0_0')

    def test_steady_state(self):
        order = 2
        Ne = 1000
        P = demography.tmrcas.transition(order, [Ne], [[1]])
        T = np.linalg.inv(np.eye(2) - P).dot(np.ones(2))
        dg = dg_without_selfing()
        T2 = demography.tmrcas.steady_state_tmrca(dg, Ne, order)
        self.assertTrue(np.all(T==T2))
        self.assertTrue(np.allclose(T, [2*Ne*(4*Ne-1), 2*Ne]))

    def test_transition_sums(self):
        order = 4
        num_pops = 5
        m = np.random.rand(num_pops**2).reshape(num_pops, num_pops)
        for i in range(len(m)):
            m[i] /= np.sum(m[i])
        P = demography.tmrcas.transition(order, [np.inf]*num_pops, m)
        s = [1]
        for i in range(1, order):
            s.append(s[-1]+2**i)
        s = s[::-1] * (num_pops * (num_pops+1) // 2)
        self.assertTrue(np.allclose(np.sum(P, axis=1), s))
    
    def test_get_gens_in_interval(self):
        Ne = 1000
        T_elapsed = 0.0
        T = 1
        current_gen = 0
        t, T_elapsed, current_gen, integ_time = demography.tmrcas.gens_in_interval_as_t(
            Ne, T_elapsed, T, current_gen, 0)
        self.assertTrue(len(t) == 2*Ne)
        self.assertTrue(t[0] == 0)
        self.assertTrue(np.isclose(t[-1], 1-1/2/Ne))

    def test_get_pop_sizes(self):
        Ne = 1000
        T_elapsed = 0.0
        T = 0.05
        current_gen = 0
        
        nus = [1,2,3,4,5]
        Ns, T_elapsed, current_gen, integ_time = demography.tmrcas.get_pop_sizes(
            Ne, nus, T_elapsed, T, current_gen, 0)
        self.assertTrue(len(Ns) == 5)
        self.assertTrue(np.allclose([N[0] for N in Ns], [nu * Ne for nu in nus]))
        self.assertTrue(current_gen == int(T*2*Ne))
        self.assertTrue(np.isclose(T_elapsed, T))
        self.assertTrue(integ_time == T)
    
    def test_setup_of_ooa(self):
        dg = ooa()
        (present_pops, integration_times, nus, migration_matrices, frozen_pops,
            selfing_rates, events) = demography.integration.get_moments_arguments(dg)
        
        Ne = dg.Ne
        
        current_gen = 0
        T_elapsed = 0
        integ_time = 0
        
        # first Ns
        Ns, T_elapsed, current_gen, integ_time = demography.tmrcas.get_pop_sizes(Ne, 
            nus[0], T_elapsed, integration_times[0], current_gen, integ_time)
        self.assertTrue(len(Ns[0]) == 0) # no integration of root
        # second Ns
        Ns, T_elapsed, current_gen, integ_time = demography.tmrcas.get_pop_sizes(Ne, 
            nus[1], T_elapsed, integration_times[1], current_gen, integ_time)
        self.assertTrue(Ns[0][0] == Ne*nus[1][0])
        self.assertTrue(integ_time == sum(integration_times[:2]))
        # third epoch
        Ns, T_elapsed, current_gen, integ_time = demography.tmrcas.get_pop_sizes(Ne, 
            nus[2], T_elapsed, integration_times[2], current_gen, integ_time)
        self.assertTrue(np.allclose([N[0] for N in Ns], [Ne*nu for nu in nus[2]]))
        # third epoch
        Ns, T_elapsed, current_gen, integ_time = demography.tmrcas.get_pop_sizes(Ne, 
            nus[3], T_elapsed, integration_times[3], current_gen, integ_time)
        self.assertTrue(np.isclose(Ns[0][0], nus[3][0][0]*Ne, atol=2))
        self.assertTrue(np.isclose(Ns[1][0], nus[3][1][0]*Ne, atol=2))
        self.assertTrue(len(Ns[0]) == len(Ns[1]))
        self.assertTrue(len(Ns[0]) == len(Ns[2]))
        self.assertTrue(np.all(Ns[2] == nus[3][2]*Ne))

    def test_evolve_t(self):
        Ne = 1000
        order = 2
        mig_mat = [[0]]
        frozen = [False]
        Tmrcas = np.array([2*Ne*(4*Ne-1), 2*Ne])
        Ns = [[Ne] * 20]
        Tmrcas2 = demography.tmrcas.evolve_t(Tmrcas, order, Ns, mig_mat, frozen)
        self.assertTrue(np.all(Tmrcas == Tmrcas2))
    
    def test_split(self):
        pop_ids = ['A']
        pop_from = 'A'
        pops_to = ['B','C']
        order = 2
        Tmrcas = [1,2]
        split_Tmrcas, new_pop_ids = demography.tmrcas.split(Tmrcas, pop_from, 
                                                            pops_to,
                                                            pop_ids, order)
        self.assertTrue(new_pop_ids[0] == 'B')
        self.assertTrue(new_pop_ids[1] == 'C')
        self.assertTrue(np.allclose(Tmrcas*3, split_Tmrcas))
        
    
    def test_reorder_pops(self):
        old_pop_ids = ['pop1', 'pop2', 'pop3']
        new_pop_ids = ['pop1', 'pop3', 'pop2']
        order = 2
        num_vals = len(demography.tmrcas.tmrca_vector(order, len(old_pop_ids)))
        T = np.linspace(1, num_vals, num_vals)
        T_new,ids = demography.tmrcas.reorder_pops(T, order, old_pop_ids, new_pop_ids)
        self.assertTrue(T[0] == T_new[0])
        self.assertTrue(np.all([id1 == id2 for id1, id2 in zip(ids, new_pop_ids)]))
        T_back,ids = demography.tmrcas.reorder_pops(T_new, order, new_pop_ids, old_pop_ids)
        self.assertTrue(np.all(T == T_back))
        self.assertTrue(np.all([id1 == id2 for id1, id2 in zip(ids, old_pop_ids)]))

    def test_pass(self):
        Tmrcas = [1]
        pop_from = 'A'
        pop_to = 'B'
        pop_ids = ['A']
        Tmrcas, pop_ids = demography.tmrcas.pass_pop(Tmrcas, pop_from, pop_to,
                                                     pop_ids)
        self.assertTrue(len(pop_ids) == 1)
        self.assertTrue(pop_ids[0] == 'B')

    def test_ooa_integration(self):
        pop_ids = ['YRI','CEU','CHB']
        order = 2
        dg = ooa()
        Ne = dg.Ne
        Tmrcas = demography.tmrcas.integrate_tmrca(dg, Ne, order, pop_ids)
        

suite = unittest.TestLoader().loadTestsFromTestCase(TestTmrcaFunctions)

if __name__ == '__main__':
    unittest.main()
