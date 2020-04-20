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

import math
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
                    initial_size=dg.G.nodes[pop]['nu']*Ne))
        elif 'nuF' in dg.G.nodes[pop]:
            pop_configs.append(
                msprime.PopulationConfiguration(
                    initial_size=dg.G.nodes[pop]['nuF']*Ne))
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


def graph_from_msprime(demographic_events, migration_matrix,
                       population_configurations, populations=None):
    """
    Construct a DemoGraph from msprime `demographic_events` and
    `population_configurations`. An optional list of population names may
    be provided in `populations`, in the same order as population_configurations.
    """

    if populations is None:
        n_pops = len(population_configurations)
        populations = [str(i) for i in range(1, n_pops+1)]

    G = nx.DiGraph()

    for j, pc in enumerate(population_configurations):
        attr = dict(
                end_time=0, end_size=pc.initial_size,
                growth_rate=pc.growth_rate)

        # initial migrations
        m_dict = dict()
        for k, m in enumerate(migration_matrix[j]):
            pop_k = populations[k]
            if m != 0:
                m_dict[pop_k] = m
        if len(m_dict) > 0:
            attr.update(m=m_dict)

        G.add_node(populations[j], **attr)

        """
        # distinguish leaf nodes in graphviz diagrams
        attr.update(color="red", shape="rect")
        """

    def get_parent(G, x):
        edges = G.in_edges(x)
        if len(edges) == 0:
            return None
        assert len(edges) == 1, f"{x} has too many parent nodes"
        parent, _ = next(iter(edges))
        return parent

    def top_of_lineage(G, x):
        y = get_parent(G, x)
        while y is not None:
            x = y
            y = get_parent(G, x)
        return x

    def start_size(G, x, time):
        attr = G.nodes[x]
        end_time = attr["end_time"]
        end_size = attr["end_size"]
        assert end_time is not None, f"{x}: {attr}"
        assert end_size is not None, f"{x}: {attr}"
        growth_rate = attr.get("growth_rate")
        if growth_rate is not None:
            start_size = end_size * math.exp(growth_rate * (end_time - time))
        else:
            start_size = end_size
        return start_size

    prev_event_time = 0
    for i, event in enumerate(demographic_events):
        if event.time < prev_event_time:
            raise ValueError(
                    "demographic events must be sorted in time-ascending order")
        prev_event_time = event.time

        if isinstance(event, msprime.MassMigration):
            source = top_of_lineage(G, populations[event.source])
            dest = top_of_lineage(G, populations[event.dest])

            if event.proportion == 1:
                t_dest = G.nodes[dest].get("end_time")
                if t_dest < event.time:
                    dest_start_size = start_size(G, dest, event.time)
                    G.nodes[dest].update(
                            start_time=event.time,
                            start_size=dest_start_size)
                    new = dest + "/^"
                    G.add_node(
                            new, end_time=event.time,
                            end_size=dest_start_size,
                            growth_rate=G.nodes[dest].get("growth_rate"))
                    G.add_edge(new, dest)
                    dest = new
                G.nodes[source].update(
                        start_time=event.time,
                        start_size=start_size(G, source, event.time))
                G.add_edge(dest, source)
            else:
                pulse = G.nodes[source].get("pulse", set())
                pulse.add((dest, event.time, event.proportion))
                G.nodes[source]["pulse"] = pulse

        elif isinstance(event, msprime.PopulationParametersChange):
            pop = top_of_lineage(G, populations[event.population])
            pop_start_size = start_size(G, pop, event.time)
            if event.initial_size is not None:
                end_size = event.initial_size
            else:
                end_size = pop_start_size
            t_pop = G.nodes[pop].get("end_time")
            if t_pop < event.time:
                G.nodes[pop].update(
                    start_time=event.time,
                    start_size=pop_start_size)
                new = pop + "/^"
                G.add_node(
                        new, end_time=event.time,
                        end_size=end_size,
                        growth_rate=event.growth_rate)
                G.add_edge(new, pop)
            else:
                G.nodes[pop].update(
                        end_size=end_size,
                        growth_rate=event.growth_rate)

        elif isinstance(event, msprime.MigrationRateChange):
            m = event.rate
            if event.matrix_index is not None:
                j, k = event.matrix_index
                dest = top_of_lineage(G, populations[j])
                source = top_of_lineage(G, populations[k])
                t_dest = G.nodes[dest].get("end_time")
                if t_dest < event.time:
                    dest_start_size = start_size(G, dest, event.time)
                    G.nodes[dest].update(
                            start_time=event.time,
                            start_size=dest_start_size)
                    new = dest + "/^"
                    G.add_node(
                            new, end_time=event.time,
                            end_size=dest_start_size,
                            growth_rate=G.nodes[dest].get("growth_rate"),
                            m={source: m})
                    G.add_edge(new, dest)
                else:
                    G.nodes[dest].update(m={source: m})
            else:
                # all populations have migration rates changed
                current_pops = set()
                for pop in populations:
                    current_pops.add(top_of_lineage(G, pop))
                for pop in current_pops:
                    m_dict_cur = G.nodes[pop].get("m", dict())
                    m_dict = dict()
                    new_epoch = False
                    if m == 0 and len(m_dict_cur) > 0:
                        new_epoch = True
                    elif m > 0:
                        for source in current_pops:
                            if pop == source:
                                continue
                            m_dict[source] = m
                            if m != m_dict_cur.get(source, 0):
                                new_epoch = True
                    if new_epoch:
                        t_pop = G.nodes[pop].get("end_time")
                        if t_pop < event.time:
                            pop_start_size = start_size(G, pop, event.time)
                            G.nodes[pop].update(
                                    start_time=event.time,
                                    start_size=pop_start_size)
                            new = pop + "/^"
                            attr = dict(
                                    end_time=event.time,
                                    end_size=pop_start_size,
                                    growth_rate=G.nodes[pop].get("growth_rate"))
                            if len(m_dict) > 0:
                                attr.update(m=m_dict)
                            G.add_node(new, **attr)
                            G.add_edge(new, pop)
                        else:
                            # whew! just update existing nodes here
                            if len(m_dict) > 0:
                                G.nodes[pop].update(m=m_dict)
                            else:
                                del G.nodes[pop]["m"]

    assert nx.is_directed_acyclic_graph(G), \
            "Cycle detected. Please report this bug, and include the " \
            "demographic model that triggered this error."

    # Non-treeness is probably an internal error. But its possible the user
    # provided a demographic model corresponding to multiple isolated subgraphs.
    assert nx.is_tree(G)

    root = next(nx.topological_sort(G))
    G = nx.relabel_nodes(G, {root: "root"})
    root = "root"

    if G.nodes[root].get("start_size") is None:
        G.nodes[root].update(
                start_time=G.nodes[root]["end_time"],
                start_size=G.nodes[root]["end_size"])
    Ne_ref = G.nodes[root].get("start_size")

    for node in nx.topological_sort(G):
        attr = G.nodes[node]
        start_size = attr.get("start_size")
        end_size = attr.get("end_size")
        start_time = attr.get("start_time")
        end_time = attr.get("end_time")

        assert start_size is not None, f"{node} has no start_size"
        assert end_size is not None, f"{node} has no end_size"
        assert start_time is not None, f"{node} has no start_time"
        assert end_time is not None, f"{node} has no end_time"

        if math.isclose(start_size, end_size, rel_tol=1e-3):
            attr.update(nu=start_size/Ne_ref)
        else:
            attr.update(nu0=start_size/Ne_ref, nuF=end_size/Ne_ref)

        T = (start_time - end_time) / (2 * Ne_ref)
        attr.update(T=T)

        # fix pulse times to be a fraction of the epoch length
        pulses = attr.get("pulse")
        if pulses is not None:
            pulses2 = set()
            for source, time, proportion in pulses:
                t_frac = (start_time - time) / (start_time - end_time)
                pulses2.add((source, t_frac, proportion))
            attr.update(pulse=pulses2)

        # scale migration rates by Ne
        m_dict = attr.get("m")
        if m_dict is not None:
            for source in m_dict.keys():
                m_dict[source] *= 2 * Ne_ref

        """
        # make informative label for graphviz diagrams
        label = node
        #for k in ("start_time", "end_time", "start_size", "end_size"):
        for k in ("T", "nu", "nu0", "nuF"):
            if k in attr:
                label += f"\n{k}={attr[k]:.3f}"
        if "pulse" in attr:
            for source, t, prop in attr["pulse"]:
                label += f"\npulse[{source}]: t_frac={t:.3f}, m={prop:.3g}"
        if "m" in attr:
            for source, prop in attr["m"].items():
                label += f"\nm[{source}]={prop:.3g}"
        attr.update(label=label)
        """

    return demography.DemoGraph(G, Ne=Ne_ref)
