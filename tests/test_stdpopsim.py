import unittest
from demography.msprime_functions import msprime_from_graph, graph_from_msprime

import stdpopsim


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


suite = unittest.TestLoader().loadTestsFromTestCase(TestStdpopsimModels)


if __name__ == '__main__':
    unittest.main()
