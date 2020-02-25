"""
Tests for creating DemoGraph object, and that we catch errors in input topologies.
"""
import unittest
import numpy as np
import networkx as nx
import demography
from demography.util import InvalidGraph
import copy
import moments

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

def example_three_split():
    G = nx.DiGraph()
    G.add_node('root', nu=1, T=0)
    G.add_node('pop1', nu=1, T=1)
    G.add_node('pop2', nu=1, T=1)
    G.add_node('pop3', nu=1, T=1)
    G.add_edges_from([('root','pop1'), ('root','pop2'), ('root','pop3')])
    return demography.DemoGraph(G)

class TestMomentsIntegration(unittest.TestCase):
    """
    Tests parsing the DemoGraph object to pass to moments.LD
    """
    def test_expected_sfs(self):
        pop_id = 'root'
        ns0 = 20
        nu = 1.
        theta = 1.0
        fs_moments = moments.Spectrum(
            moments.LinearSystem_1D.steady_state_1D(ns0, gamma=0, h=0.5),
            pop_ids=[pop_id])
        fs = demography.integration.moments_fs_root_equilibrium(ns0, nu, theta, pop_id)
        self.assertTrue(np.allclose(fs_moments, fs))
        self.assertTrue(fs.pop_ids == [pop_id])

    def test_maximum_pops_allowable(self):
        G = nx.DiGraph()
        G.add_node('root', nu=1, T=0)
        for i in ['A','B','C','D']:
            G.add_node('pop{0}'.format(i), nu=1, T=1)
        for i in range(6):
            G.add_node('pop{0}'.format(i), nu=1, T=max(5-i,1))
        G.add_edges_from([('root','popA'),('root','pop0'),('popA','pop1'),
            ('popA','popB'),('popB','pop2'),('popB','popC'),('popC','pop3'),
            ('popC','popD'),('popD','pop4'),('popD','pop5')])
        dg = demography.DemoGraph(G)
        (present_pops, integration_times, nus, migration_matrices, frozen_pops,
            selfing_rates, events) = demography.integration.get_moments_arguments(dg)
        self.assertRaises(AssertionError, demography.integration.check_max_num_pops, 
                          present_pops, 5)

    def test_lineages_needed(self):
        G = test_graph()
        dg = demography.DemoGraph(G)
        (present_pops, integration_times, nus, migration_matrices, frozen_pops,
            selfing_rates, events) = demography.integration.get_moments_arguments(dg)
        pop_ids = ['pop1','pop2']
        sample_sizes = [10, 20]
        lineages = demography.integration.get_number_needed_lineages(dg, 
            pop_ids, sample_sizes, events)
        self.assertTrue(lineages['pop1'] == sample_sizes[0])
        self.assertTrue(lineages['pop2'] == sample_sizes[1])
        self.assertTrue(lineages['root'] == 40)

    def test_sfs_pass(self):
        fs = moments.Demographics2D.snm([20,20])
        fs.pop_ids = ['pop1', 'pop2']
        child = 'child'
        self.assertTrue(fs.pop_ids[1] == 'pop2')
        fs = demography.integration.moments_pass(fs, 'pop2', child)
        self.assertTrue(fs.pop_ids[1] == child)

    def test_sfs_split(self):
        lineages = {'pop0':40, 'pop1':20, 'pop2':20, 'child1':15, 
                    'child2':5, 'child3':2, 'child4':2}
        fs = moments.Demographics1D.snm([40])
        fs.pop_ids = ['pop0']
        fs2 = moments.Manips.split_1D_to_2D(fs, 20, 20)
        pops_to = ['pop1','pop2']
        fs2.pop_ids = pops_to
        fs2_dg = demography.integration.moments_split(fs, 'pop0', ['pop1', 'pop2'],
                                                      lineages)
        self.assertTrue(fs2.pop_ids == fs2_dg.pop_ids)
        fs3_1 = moments.Manips.split_2D_to_3D_1(fs2, 15, 5)
        fs3_1.pop_ids = ['child1','pop2','child2']
        fs3_2 = moments.Manips.split_2D_to_3D_2(fs2, 15, 5)
        fs3_2.pop_ids = ['pop1','child1','child2']
        fs3_1_dg = demography.integration.moments_split(copy.deepcopy(fs2_dg), 'pop1', ['child1', 'child2'],
                                                      lineages)
        fs3_2_dg = demography.integration.moments_split(copy.deepcopy(fs2_dg), 'pop2', ['child1', 'child2'],
                                                      lineages)
        self.assertTrue(fs3_1.pop_ids == fs3_1_dg.pop_ids)
        self.assertTrue(fs3_2.pop_ids == fs3_2_dg.pop_ids)
        fs3 = moments.Manips.split_2D_to_3D_2(fs2, 10, 10)
        fs3.pop_ids = ['pop1','child1','child2']
        fs4 = demography.integration.moments_split(copy.deepcopy(fs3_2_dg), 'pop1', ['child3', 'child4'],
                                                      lineages)
        self.assertTrue(fs4.pop_ids == ['child3','child1','child2','child4'])
        fs4 = demography.integration.moments_split(copy.deepcopy(fs3_2_dg), 'child1', ['child3', 'child4'],
                                                      lineages)
        self.assertTrue(fs4.pop_ids == ['pop1','child3','child2','child4'])
        fs4 = demography.integration.moments_split(copy.deepcopy(fs3_2_dg), 'child2', ['child3', 'child4'],
                                                      lineages)
        self.assertTrue(fs4.pop_ids == ['pop1','child1','child3','child4'])

    def test_sfs_merge(self):
        # fill in
        pass

    def test_reorder_pops(self):
        # fill in
        pass

    def test_moments_output(self):
        G = nx.DiGraph()
        G.add_node('root', nu=1, T=0)
        G.add_node('pop1', nu=1, T=0)
        G.add_edge('root','pop1')
        dg = demography.DemoGraph(G)
        fs = dg.SFS(theta=1.0, pop_ids=['pop1'], sample_sizes=[10])
        self.assertTrue(np.allclose(moments.Demographics1D.snm([10]), fs))
        G = test_graph()
        dg = demography.DemoGraph(G)
        fs = dg.SFS(theta=1, pop_ids=dg.leaves, sample_sizes=[10,10])
        # test that the output is correct against implemented moments model for this

    def test_moments_integration_results(self):
        G = test_graph()
        dg = demography.DemoGraph(G)
        fs = dg.SFS(theta=1.0, pop_ids=['pop1', 'pop2'], sample_sizes=[20,20])
        fs2 = moments.Demographics1D.snm([60])
        fs2.integrate([2.0], 0.5)
        fs2 = moments.Manips.split_1D_to_2D(fs2, 20, 40)
        fs2.integrate([1.5, 1.], 0.1)
        fs2 = moments.Manips.split_2D_to_3D_2(fs2, 20, 20)
        nu_func = lambda t: [1.5, 1.0, 0.5*np.exp(np.log(3.0/0.5)*t/0.15)]
        fs2.integrate(nu_func, 0.1)
        fs2 = moments.Manips.admix_into_new(fs2, 0, 1, 20, 0.7, 0)
        nu2 = nu_func(0.1)[2]
        nu_func = lambda t: [1.0, nu2*np.exp(np.log(3.0/nu2)*t/0.05)]
        fs2.integrate(nu_func, 0.05)
        self.assertTrue(np.allclose(fs.data, fs2.data))
        
    def test_lineages_needed_for_pulse(self):
        for (f, n_from, n_to) in [(.01, 10, 10), (.05, 10, 100), (.1, 50, 40), (.4, 30, 30)]:
            n_from_pre = demography.integration.num_lineages_pulse_pre(n_from, n_to, f)
            n_from_post = demography.integration.num_lineages_pulse_post(n_from_pre, n_to, f)
            self.assertTrue(n_from == n_from_post)

    def test_pulses(self):
        G = nx.DiGraph()
        G.add_node('root', nu=1, T=0)
        G.add_node('pop1', nu=1, T=.5)
        G.add_node('pop2', nu=1, T=.5, pulse={('pop1', .1, .05), ('pop1', .5, 0.1)})
        G.add_edges_from([('root','pop1'),('root','pop2')])
        dg = demography.DemoGraph(G)
        (present_pops, integration_times, nus, migration_matrices, frozen_pops,
            selfing_rates, events) = demography.integration.get_moments_arguments(dg)
        lineages = demography.integration.get_number_needed_lineages(dg, ['pop1','pop2'], [10, 10], events)
        fs = dg.SFS(['pop1','pop2'], [10,10])

    def test_extending_frozen_pops(self):
        G = nx.DiGraph()
        G.add_node('root', nu=1, T=0)
        G.add_node('pop1', nu=1, T=.5)
        G.add_node('pop2', nu=1, T=.3)
        G.add_edges_from([('root','pop1'),('root','pop2')])
        dg = demography.DemoGraph(G)
        dg_aug = demography.integration.augment_with_frozen(dg, ['pop1','pop2'])
        acc_times = demography.util.get_accumulated_times(dg_aug)
        self.assertTrue(np.all(np.array(list(acc_times.values()))-list(acc_times.values())[0] < 1e-12))
        
    def test_frozen_pops_sfs(self):
        G = nx.DiGraph()
        G.add_node('root', nu=1, T=0)
        G.add_node('pop1', nu=1, T=.5)
        G.add_node('pop2', nu=1, T=.3)
        G.add_edges_from([('root','pop1'),('root','pop2')])
        dg = demography.DemoGraph(G)
        fs = dg.SFS(['pop1','pop2'],[20,20], theta=1)
        fs2 = moments.Demographics1D.snm([40])
        fs2 = moments.Manips.split_1D_to_2D(fs2, 20, 20)
        fs2.integrate([1,1], .3)
        fs2.integrate([1,1], .2, frozen=[False,True])
        self.assertTrue(np.allclose(fs.data, fs2.data))

suite = unittest.TestLoader().loadTestsFromTestCase(TestMomentsIntegration)

if __name__ == '__main__':
    unittest.main()
