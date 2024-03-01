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

import sys
import math
import itertools
import networkx as nx
import numpy as np
import demography
from . import integration
from . import util

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
            Ne = dg.G.nodes['root']['nu']

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
        if 'nu' in dg.G.nodes[pop]:
            rates[pop] = 0
        else:
            nuF = dg.G.nodes[pop]['nuF']
            nu0 = dg.G.nodes[pop]['nu0']
            T = dg.G.nodes[pop]['T']
            r = np.log(nuF/nu0) / T / 2 / Ne
            rates[pop] = r
    return rates


def get_population_configurations(dg, contemporary_pops, growth_rates, Ne):
    pop_configs = []
    pop_indexes = {}
    for ii,pop in enumerate(dg.G.nodes):
        pop_indexes[pop] = ii
        if 'nu' in dg.G.nodes[pop]:
            pop_configs.append(
                msprime.PopulationConfiguration(
                    initial_size=dg.G.nodes[pop]['nu']*Ne,
                    metadata={'label': pop})
            )
        elif 'nuF' in dg.G.nodes[pop]:
            pop_configs.append(
                msprime.PopulationConfiguration(
                    initial_size=dg.G.nodes[pop]['nuF']*Ne,
                    metadata={'label': pop}))
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
        if 'm' in dg.G.nodes[pop]:
            ind_from = pop_indexes[pop]
            for pop_to in dg.G.nodes[pop]['m']:
                if pop_to in contemporary_pops:
                    ind_to = pop_indexes[pop_to]
                    scaled_rate = dg.G.nodes[pop]['m'][pop_to] / 2 / Ne
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
        # append events (in reverse order backward in time)
        for e in es[::-1]:
            demo_events = demo_event_at(elapsed_time, e, pop_indexes,
                                        demo_events)
        # set population sizes for present pops
        # here, we just have to turn on growth rates, if they need to be
        # note that we have already set the initial sizes
        demo_events = update_population_sizes(elapsed_time, pops, growth_rates,
                                              pop_indexes, demo_events)
        # set migration rates for present pops
        demo_events = update_migration_rates(dg, elapsed_time, pops,
                                             pop_indexes, demo_events, Ne)
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
        for source_pop in e[2:]:
            demo_events.append(msprime.MassMigration(time=t,
                                       source=pop_indexes[source_pop],
                                       destination=pop_indexes[e[1]],
                                       proportion=1.))
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


def update_migration_rates(dg, t, current_pops, pop_indexes, demo_events, Ne):
    # turns off all migration, and then adds rates only between populations
    # listed in current_pops
    demo_events.append(msprime.MigrationRateChange(time=t, rate=0))
    for pop in current_pops:
        if 'm' in dg.G.nodes[pop]:
            for pop_to in dg.G.nodes[pop]['m']:
                # if pop_to is present
                if pop_to not in current_pops:
                    continue
                rate = dg.G.nodes[pop]['m'][pop_to]
                demo_events.append(msprime.MigrationRateChange(time=t, 
                    rate=rate / 2 / Ne,
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
    leaf_times = util.get_accumulated_times(dg)
    max_leaf_time = max(leaf_times.values())
    samples = []
    for pop,ns in zip(pop_ids, sample_sizes):
        assert pop in dg.leaves, "samples can only be taken from leaves"
        pop_time = max_leaf_time - leaf_times[pop]
        samples.extend([msprime.Sample(pop_indexes[pop], time=pop_time)] * ns)
    return samples


def graph_from_ddb_graph(dict_graph, node_attrs, node_labels, dot=False):
    """
    Add attributes to a dict-of-dicts graph ``(dict_graph, node_attrs)`` that
    can be used to import the graph as a DemoGraph().
    """
    # Get the nodes in the dict-of-dicts graph.
    from_nodes = set(dict_graph.keys())
    to_nodes = set(itertools.chain(*(d.keys() for d in dict_graph.values())))
    all_nodes = from_nodes | to_nodes
    root_set = from_nodes - to_nodes
    assert len(root_set) == 1
    root = root_set.pop()
    Ne_ref = node_attrs[root]["start_size"]

    epoch_name_suffix = True
    if len(node_labels) == len(all_nodes):
        epoch_name_suffix = False

    # New node attributes dict.
    dg_attrs = dict()

    for node, node_attr in node_attrs.items():
        epoch_j, pop_k = node
        name = node_labels[pop_k]
        if epoch_name_suffix and epoch_j > 0:
            name += f"/{epoch_j}"

        # Save the node identifier from the msprime.DemographyDebugger
        # dict-of-dicts graph.
        dg_attr = dict(msprime_node=node)

        start_time = node_attr["start_time"]
        end_time = node_attr["end_time"]
        start_size = node_attr["start_size"]
        end_size = node_attr["end_size"]

        if math.isclose(start_size, end_size):
            dg_attr.update(nu=start_size / Ne_ref)
        else:
            dg_attr.update(nu0=end_size / Ne_ref, nuF=start_size / Ne_ref)

        # Get pulses and make the times a fraction of the epoch length.
        # The pulse_in node attribute uses a backwards-time convention,
        # which corresponds to pulses out of the node in the forwards-time
        # convention used by DemoGraph.
        pulse_out = node_attr["pulse_in"]
        if len(pulse_out) > 0:
            pulse = set()
            for ((epoch_j, pop_k), time), proportion in pulse_out.items():
                source = node_labels[pop_k]
                if epoch_name_suffix and epoch_j > 0:
                    source += f"/{epoch_j}"
                t_frac = (end_time - time) / (end_time - start_time)
                pulse.add((source, t_frac, proportion))
            dg_attr.update(pulse=pulse)

        def get_source(node_attrs, node):
            """
            Return ``node`` or its oldest descendent in ``node_attrs``.
            """
            a = node[0]
            while a >= 0:
                node = (a, node[1])
                if node in node_attrs:
                    break
                a -= 1
            if a < 0:
                node = None
            return node

        # Migration rates.
        # Like for migration pulses, the M_in node attribute corresponds to the
        # migrants out of the node in a forward-time convention.
        M_out = node_attr["M_in"]
        if any(M_out > 0):
            m_dict = dict()
            for pop_i, m in enumerate(M_out):
                if m > 0:
                    source_node = get_source(node_attrs, (epoch_j, pop_i))
                    if source_node is None:
                        raise ValueError(f"No source for node {node}.")
                    source = node_labels[pop_i]
                    if epoch_name_suffix and source_node[0] > 0:
                        source += f"/{source_node[0]}"
                    m_dict[source] = m * 2 * Ne_ref
            dg_attr.update(m=m_dict)

        if node == root:
            T = 0
        else:
            T = (end_time - start_time) / (2 * Ne_ref)
        dg_attr.update(T=T)

        # Make informative label for graphviz diagrams.
        label = name
        for k in ("T", "nu", "nu0", "nuF"):
            if k in dg_attr:
                label += f"\n{k}={dg_attr[k]:.3g}"
        if "pulse" in dg_attr:
            for source, t, prop in dg_attr["pulse"]:
                label += f"\npulse[{source}]: t_frac={t:.3g}, m={prop:.3g}"
        if "m" in dg_attr:
            for source, prop in dg_attr["m"].items():
                label += f"\nm[{source}]={prop:.3g}"
        dg_attr.update(label=label)
        dg_attrs[name] = dg_attr

    G = nx.DiGraph(dict_graph)
    assert nx.is_directed_acyclic_graph(G)
    assert nx.is_tree(G)
    # Rename nodes.
    name_map = {attr["msprime_node"]: node for node, attr in dg_attrs.items()}
    nx.relabel_nodes(G, name_map, copy=False)
    # Apply the node attributes.
    for node, node_attr in dg_attrs.items():
        G.nodes[node].update(**node_attr)

    if dot:
        # Output graphviz dot file.        
        A = nx.nx_agraph.to_agraph(G)
        # Place all nodes for a given epoch at the same height.
        for epoch_j, nodes in itertools.groupby(G.nodes, lambda n: G.nodes[n]["msprime_node"][0]):
            A.add_subgraph(list(nodes), name=str(epoch_j), rank="same")
        A.write(sys.stdout)

    return demography.DemoGraph(G, Ne=Ne_ref)


def graph_from_msprime(
        demographic_events, migration_matrix, population_configurations,
        populations=None, leaves=None, dot=False):
    """
    Construct a DemoGraph from msprime `demographic_events` and
    `population_configurations`. An optional list of population names may
    be provided in `populations`, in the same order as population_configurations.
    If `leaves` is None, all populations will be treated as leaves. However,
    if some populations should be internal nodes, then `leaves` should be a
    list of the leaf populations.
    """
    if populations is None:
        populations = [f"pop{i}" for i in range(len(population_configurations))]
    if leaves is not None:
        # Convert leaves to integer population IDs.
        pop = {p: i for i, p in enumerate(populations)}
        leaves = [pop[i] for i in leaves]
    ddb = msprime.DemographyDebugger(
            demographic_events=demographic_events,
            migration_matrix=migration_matrix,
            population_configurations=population_configurations)
    dict_graph, node_attrs = ddb.as_graph(leaves=leaves)
    dg = graph_from_ddb_graph(dict_graph, node_attrs, populations, dot=dot)
    return dg
