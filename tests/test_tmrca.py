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


def dg_with_marginalize():
    Ne = 100
    G = nx.DiGraph()
    G.add_node('root', nu=1, T=0)
    G.add_node('A', nu=1, T=0.2)
    G.add_node('B0', nu=1, T=0.1)
    G.add_node('B', nu=1, T=0.1)
    G.add_node('C', nu=1, T=0.05)
    G.add_edges_from([('root','A'), ('root','B0'), ('B0','B'), ('B0','C')])
    return demography.DemoGraph(G, Ne=Ne)


def dg_with_pulse():
    Ne = 100
    G = nx.DiGraph()
    G.add_node('root', nu=1, T=0)
    G.add_node('A', nu=1, T=0.1, pulse={('B', 0.5, 0.1)})
    G.add_node('B', nu=1, T=0.1)
    G.add_edges_from([('root','A'), ('root','B')])
    return demography.DemoGraph(G, Ne=Ne)


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
        self.assertTrue(tmrcas[0] == f'T{order}_0_0')
        self.assertTrue(tmrcas[1] == f'T{order}_0_1')
        self.assertTrue(tmrcas[-1] == f'T1_{num_pops-1}_{num_pops-1}')

    def test_steady_state(self):
        order = 2
        Ne = 1000
        P = demography.tmrcas.transition(order, [Ne], [[1]], [False])
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
        P = demography.tmrcas.transition(order, [np.inf]*num_pops, m,
                                         [False]*num_pops)
        s = [1] * (num_pops * (num_pops+1) // 2)
        for i in range(1, order):
            s.extend([s[-1]+2**i] * (num_pops * (num_pops+1) // 2))
        self.assertTrue(np.allclose(np.sum(P, axis=1), s[::-1]))
    
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
        T = [1,2]
        split_Tmrcas, new_pop_ids = demography.tmrcas.split(T, pop_from, 
                                                            pops_to,
                                                            pop_ids, order)
        self.assertTrue(new_pop_ids[0] == 'B')
        self.assertTrue(new_pop_ids[1] == 'C')
        self.assertTrue(np.allclose([T[0],T[0],T[0],T[1],T[1],T[1]], split_Tmrcas))
        
    
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
        Tmrcas2 = dg.tmrca(pop_ids, order=order)
        self.assertTrue(np.all(Tmrcas==Tmrcas2))

    def test_steady_state_orders(self):
        dg = dg_without_selfing()
        Ne = 1000
        T1 = demography.tmrcas.steady_state_tmrca(dg, Ne, 1)
        T2 = demography.tmrcas.steady_state_tmrca(dg, Ne, 2)
        T3 = demography.tmrcas.steady_state_tmrca(dg, Ne, 3)
        T4 = demography.tmrcas.steady_state_tmrca(dg, Ne, 4)
        self.assertTrue(T1[0] == T2[-1] == T3[-1] == T4[-1])
        self.assertTrue(T2[-2] == T3[-2] == T4[-2])
        self.assertTrue(T3[-3] == T4[-3])

    def test_pulse_admixture(self):
        pids = ['A','B']
        Tmrcas = np.array([1,1,1])
        order = 1
        pop_from = 'B'
        pop_to = 'A'
        f = 0.2
        
        Pulse = demography.tmrcas.pulse_matrix(len(pids), order, 1, 0, f)
        self.assertTrue(np.allclose(np.sum(Pulse, axis=1), 1.))
        
        Tmrcas, current_pop_ids = demography.tmrcas.pulse_migrate(Tmrcas,
            pop_from, pop_to, f, pids, order)
        self.assertTrue(np.allclose(np.sum(Pulse, axis=1), 1.))
        
        dg = dg_with_pulse()
        order = 2
        Tmrcas = dg.tmrca(['A','B'], order=order)

    def test_marginalize(self):
        curr_pids = ['A','B','C']
        num_pops = len(curr_pids)
        order = 1
        pop_ids = ['A','B']
        Tmrcas = np.arange(1,num_pops*(num_pops+1)/2+1)
        Tmrcas,curr_pids = demography.tmrcas.marginalize(Tmrcas, 'C', curr_pids, order)
        self.assertTrue(len(curr_pids) == 2)
        self.assertTrue(curr_pids[0] == 'A')
        self.assertTrue(curr_pids[1] == 'B')

        dg = dg_with_marginalize()
        order = 2
        Tmrcas = dg.tmrca(['A','B'], order=order)
        self.assertTrue(len(Tmrcas) == 6)

    def test_frozen(self):
        order = 2
        Ne = 100
        G = nx.DiGraph()
        G.add_node('root', nu=1, T=0)
        G.add_node('A', nu=1, T=.1, frozen=True)
        G.add_node('B', nu=1, T=.1, frozen=True)
        G.add_edges_from([('root','A'),('root','B')])
        dg = demography.DemoGraph(G, Ne=Ne)

        gens = 10
        Ns = [[Ne] * gens]
        Tmrcas_init = demography.tmrcas.steady_state_tmrca(dg, Ne, order)
        Tmrcas = demography.tmrcas.evolve_t(Tmrcas_init, order, Ns, [[1]], [True])
        self.assertTrue(Tmrcas[1] == Tmrcas_init[1]+gens)
        self.assertTrue(np.isclose(Tmrcas[0]-Tmrcas[1]**2,
                                   Tmrcas_init[0]-Tmrcas_init[1]**2))

        Tmrcas = dg.tmrca(['A','B'], order=order)
        self.assertTrue(np.isclose(Tmrcas[0], Tmrcas[1]))
        self.assertTrue(np.isclose(Tmrcas[0], Tmrcas[2]))
        self.assertTrue(np.isclose(Tmrcas[3], Tmrcas[4]))
        self.assertTrue(np.isclose(Tmrcas[3], Tmrcas[5]))
        
        order = 2
        G = nx.DiGraph()
        Ne = 100
        G.add_node('root', nu=1, T=0)
        G.add_node('A', nu=1, T=.1, frozen=True)
        G.add_node('B', nu=1, T=.1)
        G.add_edges_from([('root','A'),('root','B')])
        dg = demography.DemoGraph(G, Ne=100)
        Tmrcas = dg.tmrca(['A','B'], order=order)
        self.assertTrue(np.isclose(Tmrcas[0], Tmrcas[1]))
        self.assertTrue(np.isclose(Tmrcas[0], Tmrcas[2]+8400))
        self.assertTrue(np.isclose(Tmrcas[3], Tmrcas[4]))
        self.assertTrue(np.isclose(Tmrcas[3], Tmrcas[5]+20))

        order = 2
        G = nx.DiGraph()
        Ne = 100
        G.add_node('root', nu=1, T=0)
        G.add_node('A', nu=2, T=.1, frozen=True)
        G.add_node('B', nu=2, T=.1)
        G.add_edges_from([('root','A'),('root','B')])
        dg = demography.DemoGraph(G, Ne=100)
        Tmrcas = dg.tmrca(['A','B'], order=order)
        self.assertTrue(np.isclose(Tmrcas[3], 2*Ne*(1+dg.G.nodes['A']['T'])))
        self.assertTrue(np.isclose(Tmrcas[4], 2*Ne*(1+dg.G.nodes['A']['T'])))
        self.assertTrue(Tmrcas[5] > 2*Ne)

    def test_result_against_H(self):
        dg = ooa()
        H = dg.LD(['YRI','CEU','CHB'])
        Tmrcas = dg.tmrca(['YRI','CEU','CHB'], order=1)
        self.assertTrue(np.allclose(H[-1] * 2 * dg.Ne / Tmrcas, 1., rtol=0.0001))

        dg = dg_with_marginalize()
        H = dg.LD(['A','B'])
        Tmrcas = dg.tmrca(['A','B'], order=1)
        self.assertTrue(np.allclose(H[-1] * 2 * dg.Ne / Tmrcas, 1., rtol=0.0001))

        dg = dg_with_pulse()
        H = dg.LD(['A','B'])
        Tmrcas = dg.tmrca(['A','B'], order=1)
        self.assertTrue(np.allclose(H[-1] * 2 * dg.Ne / Tmrcas, 1., rtol=0.0001))

    def test_tmrca_vector_selfing(self):
        order = 3
        num_pops = 4
        names = demography.tmrcas.tmrca_vector_selfing(order, num_pops)
        self.assertTrue(len(names) == order * (num_pops + num_pops * (num_pops+1)//2))

    def test_add_selfing_rates(self):
        dg = dg_without_selfing()
        pself = 0.1
        dg_self = demography.tmrcas.add_selfing_rates(dg, pself)
        for node in dg.G.nodes:
            self.assertTrue(dg_self.G.nodes[node]['selfing'] == pself)

    def test_equilibrium_selfing(self):
        Ne = 1000
        pself = 1./Ne
        order = 1
        P = demography.tmrcas.transition_selfing(1, [Ne], [[1]], [False], [pself])
        ss = np.linalg.inv(np.eye(2*order) - P).dot(np.ones(2*order))
        
        dg = dg_without_selfing()
        ss_noself = demography.tmrcas.steady_state_tmrca(dg, Ne, order)
        
        self.assertTrue(np.allclose(ss, ss_noself))

        pself = 0.0
        P = demography.tmrcas.transition_selfing(1, [Ne], [[1]], [False], [pself])
        ss = np.linalg.inv(np.eye(2*order) - P).dot(np.ones(2*order))
        self.assertTrue(np.allclose(ss[0], ss[1] + 1))
        
        pself = 1.0
        P = demography.tmrcas.transition_selfing(1, [Ne], [[1]], [False], [pself])
        ss = np.linalg.inv(np.eye(2*order) - P).dot(np.ones(2*order))
        self.assertTrue(np.isclose(ss[0], 2))
        self.assertTrue(np.isclose(ss[1], Ne+1))

        # two pops
        N = [Ne, Ne]
        order = 1
        p_self = 1./Ne
        m = 1./2/Ne
        P = demography.tmrcas.transition_selfing(1, N, [[1-m, m],[m, 1-m]], 
                                                 [False, False], [p_self, p_self])
        ss = np.linalg.inv(np.eye(len(P))-P).dot(np.ones(len(P)))
        
        P_rand = demography.tmrcas.transition(1, N, [[1-m, m],[m, 1-m]], 
                                              [False, False])
        ss_rand = np.linalg.inv(np.eye(len(P_rand))-P_rand).dot(np.ones(len(P_rand)))

    def test_integration_get_selfing_rates(self):
        Ne = 1000
        pself = 0.0
        dg = dg_without_selfing()
        dg_self = demography.tmrcas.add_selfing_rates(dg, pself)
        selfing = demography.integration.get_selfing_rates(dg_self.G, ['A','B'])
        self.assertTrue(selfing[0] == pself)
        self.assertTrue(selfing[1] == pself)

    def test_integrate_with_default_selfing(self):
        dg = dg_without_selfing()
        order = 1
        Ne = 1000
        pop_ids = ['A','B']
        pself = 1./Ne
        Ts = demography.tmrcas.integrate_tmrca_selfing(dg, Ne, order, pop_ids, pself)
        self.assertTrue(np.isclose(Ts[0], Ts[2]))
        self.assertTrue(np.isclose(Ts[1], Ts[4]))
        
        pself = 1.0
        Ts = demography.tmrcas.integrate_tmrca_selfing(dg, Ne, order, pop_ids, pself)
        self.assertTrue(np.isclose(Ts[0], Ts[1]))
        self.assertTrue(np.isclose(Ts[0], 2.0))
        self.assertTrue(np.isclose(Ts[2], Ts[4]))
        self.assertTrue(np.isclose(Ts[2], 1001))


suite = unittest.TestLoader().loadTestsFromTestCase(TestTmrcaFunctions)

if __name__ == '__main__':
    unittest.main()
