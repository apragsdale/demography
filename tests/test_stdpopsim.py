import unittest
from demography.msprime_functions import msprime_from_graph, graph_from_msprime
from demography import DemoGraph
import networkx as nx
import stdpopsim


def simple_model():
    # demography with splits and non-symmetric migration rates and varying sizes
    G = nx.DiGraph()
    G.add_node('root', nu=1, T=0)
    G.add_node('A', nu=.5, T=.1, m={'pop1':1})
    G.add_node('pop1', nu=2, T=.2, m={'A':2, 'pop2':3, 'pop3':4})
    G.add_node('pop2', nu=3, T=.1, m={'pop1':5, 'pop3':6})
    G.add_node('pop3', nu=4, T=.1, m={'pop1':7, 'pop2':8})
    G.add_edges_from([('root','A'), ('root','pop1'), ('A','pop2'), ('A','pop3')])
    dg = DemoGraph(G, Ne=1000)
    return dg


class TestStdpopsimModels(unittest.TestCase):
    """
    Check that we can convert stdpopsim models to graphs and back again.
    """
    def test_model_equality(self):
        for model in stdpopsim.all_demographic_models():
            dg = graph_from_msprime(
                    model.demographic_events,
                    model.migration_matrix,
                    model.population_configurations,
                    [pop.id for pop in model.populations])
            pc, mm, de = msprime_from_graph(dg)
            model2 = stdpopsim.DemographicModel(
                    id=model.id,
                    description=model.description,
                    long_description=model.long_description,
                    generation_time=model.generation_time,
                    populations=model.populations,
                    population_configurations=pc,
                    migration_matrix=mm,
                    demographic_events=de,
                    )
            try:
                model.verify_equal(model2)
            except stdpopsim.UnequalModelsError:
                print(f"{model.id}")
                raise


    def test_conversions_simple(self):
        dg = simple_model()
        pc, mm, de = dg.msprime_inputs()
        dg2 = graph_from_msprime(de, mm, pc, ['root','A','pop1','pop2','pop3'])
        # ensure we recover the same nodes
        for node in dg.G.nodes():
            self.assertTrue(node in dg2.G.nodes())
        for node in dg2.G.nodes():
            self.assertTrue(node in dg.G.nodes())
        # are their migration rates, times, and sizes all correct?
        for node in dg.G.nodes():
            self.assertTrue(np.isclose(dg.G.nodes[node]['nu'], dg2.G.nodes[node]['nu']))
            self.assertTrue(np.isclose(dg.G.nodes[node]['T'], dg2.G.nodes[node]['T']))
            for node_to, m_to in dg.G.nodes[node]['m']:
                self.assertTrue(np.isclose(dg2.G.nodes[node]['m'][node_to], m_to))

suite = unittest.TestLoader().loadTestsFromTestCase(TestStdpopsimModels)


if __name__ == '__main__':
    unittest.main()
