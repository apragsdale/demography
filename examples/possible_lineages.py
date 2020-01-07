"""
This takes msprime inputs (demographic_events, migration_matrix, and
population_configurations), as well as the sampling configuration, to
determine at what time epochs lineages are allowed in each population defined
in the population_configurations.

This only works if we set up samples separately as a list of samples, instead
of specifying them through population_configurations.
"""

import msprime
import demography
from collections import defaultdict
import copy

def get_sampling_times(samples):
    """
    takes a list of samples and returns times and locations of sampling events
    
    sampling_times stores the time of sampling events, and for which demes we
    are taking samples from at that time, as a list
    """
    sampling_times = defaultdict(list)
    for sample in samples:
        if sample.population not in sampling_times[sample.time]:
            sampling_times[sample.time].append(sample.population)
    
    return sampling_times

def get_reachable_pops_migration(mig_mat):
    """
    Assuming mig_mat[i][j] is the rate of migration of lineages from pop i
    to pop j looking *backward in time*. So if lineages are in pop i, they can
    be moved to pop j if mig_mat[i][j] is positive
    
    I'm sure there are much better algorithms for doing this... didn't take 
    the time to look though.
    """
    num_pops = len(mig_mat)
    reachable = defaultdict(set)
    # directly reachable
    for pop_from in range(num_pops):
        for pop_to in range(num_pops):
            if pop_from == pop_to:
                reachable[pop_from].add(pop_from)
            elif mig_mat[pop_from][pop_to] > 0:
                reachable[pop_from].add(pop_to)
    
    # indirectly reachable
    num_iters = 0
    any_added = 1
    while num_iters < len(mig_mat) and any_added == 1:
        num_iters += 1
        any_added = 0
        for pop_from in range(num_pops):
            for reached in reachable[pop_from]:
                for pop_to in reachable[reached]:
                    if pop_to not in reachable[pop_from]:
                        reachable[pop_from].add(pop_to)
                        any_added = 1
    
    return reachable


def get_lineage_configs(pop_config, mig_mat, demo_events, sampling_times):
    """
    A population is only turned off (set to zero) if no lineages are possible
    in that population. This can only happen if there is a mass migration
    event that takes all lineages to a different population.
    A population is turned on (set to one) if there are migration events
    (whether continuous or discrete) or sampling events that could place
    lineages in that population.
    The order that we check things is 1) sampling events, 2) discrete movement
    of lineages, and 3) migration matrix.
    """
    check_times = [de.time for de in demo_events] + list(sampling_times.keys())
    check_times = sorted(list(set(check_times)))
    
    # to make it easier to grab events for a given time
    divided_events = defaultdict(list)
    for de in demo_events:
        divided_events[de.time].append(de)
    
    # initial configuration, set all to zero
    lineage_config = [0 for i in pop_config]
    
    configs = {}
    
    for ct in check_times:
        # turn on pops from sampling lineages
        for pop_id in sampling_times[ct]:
            lineage_config[pop_id] = 1
        
        # loop through demographic events for this time, update migration
        # matrix or apply mass migrations
        for de in divided_events[ct]:
            if de.type == 'mass_migration':
                pop_from = de.source
                pop_to = de.dest
                prop = de.proportion
                if lineage_config[pop_from] == 1:
                    # lineages move into pop_to if there were any in pop_from
                    lineage_config[pop_to] = 1
                if prop == 1.0:
                    # if all lineages move, we turn off pop_from
                    lineage_config[pop_from] = 0
            if de.type == 'migration_rate_change':
                if de.matrix_index == None:
                    # all rates are set to de.rate
                    # reset mig_mat
                    mig_mat = [[de.rate for i in range(len(mig_mat))] 
                               for j in range(len(mig_mat))]
                else:
                    (i,j) = de.matrix_index
                    mig_mat[i][j] = de.rate
        
        # update lineage_config based on reachable nodes through migration mat
        reachable_pops = get_reachable_pops_migration(mig_mat)
        for pop_from in range(len(pop_config)):
            if lineage_config[pop_from] == 1:
                for pop_to in reachable_pops[pop_from]:
                    lineage_config[pop_to] = 1
        
        configs[ct] = copy.copy(lineage_config)
    
    return configs

def get_epochs(pop_config, mig_mat, demo_events, sampling_times):
    configs = get_lineage_configs(pop_config, mig_mat, demo_events,
                                  sampling_times)
    # collapse configs
    thinned_configs = {}
    for time in sorted(list(configs.keys())):
        if time == min(configs.keys()):
            thinned_configs[time] = configs[time]
        else:
            if configs[time] != thinned_configs[max(thinned_configs.keys())]:
                thinned_configs[time] = configs[time]
    
    times = sorted(list(thinned_configs.keys())) + [-1]
    epochs = [(t0, t1) for t0,t1 in zip(times[:-1],times[1:])]
    epoch_configs = {}
    for e in epochs:
        epoch_configs[e] = thinned_configs[e[0]]
    return epoch_configs

if "__name__" == "__main__":
    ## your simulation inputs here
    import homo_sapiens
    dg = homo_sapiens.ooa_gutenkunst()
    
    pop_config, mig_mat, demo_events = dg.msprime_inputs()
    
    ## set up the samples you want
    # in the OOA model as defined here, YRI is pop 3, CEU is 4, and CHB is 5
    # let's take 10 samples from each
    samples = [
        msprime.Sample(population=3, time=0) for i in range(10)
    ] + [
        msprime.Sample(population=4, time=0) for i in range(10)
    ]+ [
        msprime.Sample(population=5, time=0) for i in range(10)
    ]
    
    sampling_times = get_sampling_times(samples)
    
    epoch_configs = get_epochs(pop_config, mig_mat, demo_events,
                               sampling_times)

