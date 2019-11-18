"""
Functions to create inputs for msprime simulation and to run simulation from
the DemoGraph object.

Most basic feature returns population_configurations, migration_matrix, and 
demographic_events. Also can run simulation if the DemoGraph object has
samples, sequence length, recombination rate or genetic map, or if those items
are passed to the function.
"""

try:
    import msprime
    msprime_installed = 1
except ImportError:
    msprime_installed = 0

import numpy as np
from . import integration

def msprime_from_graph(dg, Ne=None):
    """
    Takes a DemoGraph object and returns the population configurations,
    migration matrix, and demographic events needed to simulate under msprime.
    Each node in the graph corresponds to a population in 
    population_configurations, and we don't reuse populations, so if two
    populations merge, we send all lineages from each to the new population,
    instead of send all from one into the other.

    One subtlety that I'm not sure how I want to handle yet is Ne vs nus: in
    moments, we define relative population sizes for each population (relative
    to the ancestral/reference Ne), and times are measured in units of 2Ne
    generations. In msprime, we want to define actual population sizes and
    measure time in units of generations.

    For now, I have the graph object working with relative sizes and pop-size
    scaled time, and the Ne is an attribute of the whole graph. In this
    function, we can pass Ne, and if it isn't set, we take Ne from dg.Ne, and
    if that is set to None, we print a warning that Ne is not set and we are
    assuming that Ne is equal to the size of the root (which is often set to
    one, but we could set to anything, and then in this case, times and
    migration rates are scaled by this population size).
    """
    if Ne is None:
        if dg.Ne is not None:
            Ne = dg.Ne
        else:
            print("warning: set Ne to size of root, may cause scaling issues")
            Ne = dg.G.node['root']['nu']

    # we'll use the outputs from parsing for moments integration to walk
    # through the events as they occur backward in time
    # some of these events are unused (frozen_pops, selfing_rates)
    (present_pops, integration_times, nus, migration_matrices, frozen_pops,
        selfing_rates, events) = integration.get_moments_arguments(dg)

    growth_rates = get_population_growth_rates(dg, Ne)

    population_configurations, pop_indexes = get_population_configurations(
        dg, present_pops[-1], growth_rates, Ne)

    migration_matrix = get_migration_matrix(
        dg, present_pops[-1], pop_indexes, Ne)

    demographic_events = get_demographic_events(dg, pop_indexes, present_pops,
        integration_times, nus, events, Ne, growth_rates)

    return population_configurations, migration_matrix, demographic_events


def get_population_growth_rates(dg, Ne):
    rates = {}
    for pop in dg.G.nodes:
        if 'nu' in dg.G.node[pop]:
            rates[pop] = 0
        else:
            nuF = dg.G.node[pop]['nuF']
            nu0 = dg.G.node[pop]['nu0']
            T = dg.G.node[pop]['T']
            r = np.log(nuF/nu0) / T / 2 / Ne
            rates[pop] = r
    return rates


def get_population_configurations(dg, contemporary_pops, growth_rates, Ne):
    pop_configs = []
    pop_indexes = {}
    for ii,pop in enumerate(dg.G.nodes):
        pop_indexes[pop] = ii
        if 'nu' in dg.G.node[pop]:
            pop_configs.append(
                msprime.PopulationConfiguration(
                    initial_size=dg.G.node[pop]['nu']*Ne))
        elif 'nuF' in dg.G.node[pop]:
            pop_configs.append(
                msprime.PopulationConfiguration(
                    initial_size=dg.G.node[pop]['nuF']*Ne))
            if pop in contemporary_pops:
                pop_configs[-1].growth_rate = growth_rates[pop]
        else:
            print("oops")

    return pop_configs, pop_indexes


def get_migration_matrix(dg, contemporary_pops, pop_indexes, Ne):
    ### Note need to check direction of migration matrix indexes
    num_pops = len(pop_indexes)
    M = [[0 for i in range(num_pops)] for j in range(num_pops)]
    for pop in contemporary_pops:
        if 'm' in dg.G.node[pop]:
            ind_from = pop_indexes[pop]
            for pop_to in dg.G.node[pop]['m']:
                ind_to = pop_indexes[pop_to]
                scaled_rate = dg.G.node[pop]['m'][pop_to] / 2 / Ne
                M[ind_from][ind_to] = scaled_rate
    return M


def get_demographic_events(dg, pop_indexes, present_pops, integration_times, 
                           nus, events, Ne, growth_rates):
    demo_events = []
    elapsed_time = 0
    for es, it, pops in zip(events[::-1], integration_times[:0:-1],
                            present_pops[-2::-1]):
        # update time for this set of events
        elapsed_time += 2*Ne*it
        # append events
        for e in es:
            demo_events = demo_event_at(elapsed_time, e, pop_indexes,
                                        demo_events)
        # set population sizes for present pops
        # here, we just have to turn on growth rates, if they need to be
        # note that we have already set the initial sizes
        demo_events = update_population_sizes(elapsed_time, pops, growth_rates,
                                              pop_indexes, demo_events)
        # set migration rates for present pops
        demo_events = update_migration_rates(dg, elapsed_time, pops,
                                             pop_indexes, demo_events)
    return demo_events


def demo_event_at(t, e, pop_indexes, demo_events):
    # t is the time of the event
    # e is the event (split, merger, pulse, or pass), with the needed info
    # takes demo_events and appends events to end
    if e[0] == 'pass':
        demo_events.append(
            msprime.MassMigration(time=t, source=pop_indexes[e[2]], 
                    destination=pop_indexes[e[1]], proportion=1.))
    elif e[0] == 'split':
        demo_events.append(
            msprime.MassMigration(time=t, source=pop_indexes[e[2]],
                    destination=pop_indexes[e[1]], proportion=1.))
        demo_events.append(
            msprime.MassMigration(time=t, source=pop_indexes[e[3]],
                    destination=pop_indexes[e[1]], proportion=1.))
    elif e[0] == 'merger':
        demo_events.append(
            msprime.MassMigration(time=t, source=pop_indexes[e[3]],
                    destination=pop_indexes[e[1][0]], proportion=e[2][0]))
        demo_events.append(
            msprime.MassMigration(time=t, source=pop_indexes[e[3]],
                    destination=pop_indexes[e[1][1]], proportion=1.))
    elif e[0] == 'pulse':
        demo_events.append(
            msprime.MassMigration(time=t, source=pop_indexes[e[2]],
                    destination=pop_indexes[e[1]], proportion=e[3]))
    return demo_events


def update_migration_rates(dg, t, current_pops, pop_indexes, demo_events):
    # turns off all migration, and then adds rates only between populations
    # listed in current_pops
    demo_events.append(msprime.MigrationRateChange(time=t, rate=0))
    for pop in current_pops:
        if 'm' in dg.G.node[pop]:
            for pop_to in dg.G.node[pop]['m']:
                rate = dg.G.node[pop]['m'][pop_to]
                demo_events.append(msprime.MigrationRateChange(time=t, 
                    rate=rate,
                    matrix_index=(pop_indexes[pop], pop_indexes[pop_to])))
    return demo_events


def update_population_sizes(t, current_pops, growth_rates, pop_indexes,
                            demo_events):
    # sets growth rate for each population in current_pops
    # by default when populations are first configured, they are set to zero
    # unless they are a contemporary pop
    for pop in current_pops:
        demo_events.append(msprime.PopulationParametersChange(
            time=t, population_id=pop_indexes[pop],
            growth_rate=growth_rates[pop]))
    return demo_events


def get_samples(dg, pop_ids, sample_sizes):
    """
    Get the samples list for the given population names and sample sizes.
    Samples can only be taken from populations that are leaves, and we assume
    that the sampling occurs at the end of that node in the graph.

    To get the time of the end of each leaf, we get all leaves accumulated
    end times since the root, take the max over those accumulated times, and
    subtract each leaf's time from the max.

    Need to have the pop_indexes that the population configurations
    """
    pop_configs, pop_indexes = get_population_configurations(dg, [], {}, 1)
    leaf_times = get_accumulated_times(dg)
    max_leaf_time = max(leaf_times.values())
    samples = []
    for pop,ns in zip(pop_ids, sample_sizes):
        assert pop in dg.leaves, "samples can only be taken from leaves"
        pop_time = max_leaf_time - leaf_times[pop]
        samples.extend([msprime.Sample(pop_indexes[pop], time=pop_time)] * ns)
    return samples


def get_one_parent(dg, child):
    if hasattr(dg.predecessors[child], "__len__"): # from a merger, just pick one parent
        parent = dg.predecessors[child][0]
    else:
        parent = dg.predecessors[child]
    return parent


def get_accumulated_times(dg):
    leaf_times = {}
    for leaf in dg.leaves:
        t = dg.G.nodes[leaf]['T']
        parent = get_one_parent(dg, leaf)
        while parent is not dg.root:
            t += dg.G.nodes[parent]['T']
            parent = get_one_parent(dg, parent)
        leaf_times[leaf] = t
    return leaf_times


"""
Run msprime simulation
"""

