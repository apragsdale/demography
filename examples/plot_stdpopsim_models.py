import demography
import stdpopsim
import matplotlib.pylab as plt

for model in stdpopsim.all_demographic_models():
    populations = [pop.id for pop in model.populations]
    dg = demography.msprime_functions.graph_from_msprime(
                    model.demographic_events,
                    model.migration_matrix,
                    model.population_configurations,
                    populations)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    demography.plotting.plot_graph(dg, ax=ax1, leaf_order=populations)
    demography.plotting.plot_demography(
            dg, ax=ax2, gen=model.generation_time,
            leaf_order=populations)
    fig.suptitle(model.id)
    fig.tight_layout()
    fig.savefig(f"figures/stdpopsim_{model.id}.pdf")
