import numpy as np
import copy


def all_rates(pop_ids, pop_indexes, pop_config, mig_mat, demo_events, gens):
    """
    
    """
    rates = {}
    for ii,pop1 in enumerate(pop_ids):
        for pop2 in pop_ids[ii:]:
            ind1 = pop_indexes[pop1]
            ind2 = pop_indexes[pop2]
            rates[(pop1, pop2)] = get_rates(ind1, ind2, pop_config,
                                            mig_mat, demo_events, gens)
    return rates


def get_rates(ind1, ind2, pop_config, mig_mat, demo_events, gens):
    """
    
    """
    num_pops = len(pop_config)
    
    # intitialize states
    y = np.zeros(num_pops*(num_pops+1)//2)
    inds = set_up_inds(num_pops)
    y[inds[(ind1,ind2)]] = 1
    
    # initialize vector of rates over number of generations
    rate = np.zeros(gens)
    
    # get N(t) for all time epochs
    N_t = pop_sizes(pop_config, demo_events)
    
    # get migration matrices for all epochs from demographic events
    mig_mats = migration_matrices(pop_config, mig_mat, demo_events)
    
    # get pulse migration events from demographic events
    pulses = pulse_events(pop_config, demo_events)
    
    this_N_t = {}
    drift_rate = np.zeros(num_pops*(num_pops+1)//2)
    for gen in range(gens):
        # 1. set/reset drift and migration rates
        if gen in N_t:
            for pop_ind in N_t[gen]:
                this_N_t[pop_ind] = N_t[gen][pop_ind]
        update_drift_rate(this_N_t, gen, drift_rate)
        
        if gen in mig_mats:
            m = mig_mats[gen]
            migration_rate = get_mig_transition(m)
        
        # 2. apply pulse events if any occur this generation
        if gen in pulses:
            for pulse in pulses[gen]:
                y = pulse.dot(y)
            
        # 3. apply migration events
        y = migration_rate.dot(y)
        
        # 4. compute coal rate
        # 4a. compute prob of coal in any lineage this generation
        # 4b. compute prob that coal has not occured by now
        p_coal = sum(drift_rate*y)
        p_not_yet_coaled = sum(y)
        rate[gen] = p_coal/p_not_yet_coaled
        
        # 5. apply drift events
        y = (1-drift_rate)*y
    
    return rate


def set_up_inds(num_pops):
    inds = {}
    c = 0
    for i in range(num_pops):
        for j in range(i,num_pops):
            inds[(i,j)] = c
            c += 1
    return inds    


def pop_sizes(pop_config, demo_events):
    """
    Returns pop size functions, indexed by when we start applying that func.
    Indexed by ordered indexing of populations in pop_config.
    """
    N_t = {}
    N_t[0] = {}
    for pop_ind,pc in enumerate(pop_config):
        N0 = pc.initial_size
        r = pc.growth_rate
        N_t[0][pop_ind] = drift_rate_func(N0, r, 0)
    
    for de in demo_events:
        if de.get_ll_representation()['type'] == 'population_parameters_change':
            pop_ind = de.population
            t0 = de.time
            if de.initial_size is None:
                # scroll through to get size from last function
                for t in sorted(N_t.keys()):
                    if pop_ind in N_t[t]:
                        N0 = N_t[t][pop_ind](t0)
            else:
                N0 = de.initial_size
            
            r = de.growth_rate
            gen = int(np.ceil(t0))
            N_t.setdefault(gen, {})
            N_t[gen][pop_ind] = drift_rate_func(N0, r, t0)

    return N_t


def drift_rate_func(N0, r, t0):
    return lambda t, N0=N0, r=r, t0=t0: N0 * np.exp(-r * (t-t0))


def update_drift_rate(this_N_t, gen, drift_rate):
    num_pops = len(this_N_t)
    curr_ind = 0
    for pop_ind in range(num_pops):
        drift_rate[curr_ind] = 1./(2*this_N_t[pop_ind](gen))
        curr_ind += (num_pops-pop_ind)


def migration_matrices(pop_config, mig_mat, demo_events):
    ms = {}
    m = np.array(mig_mat)
    ms[0] = copy.copy(m)
    for de in demo_events:
        gen = int(np.ceil(de.time))
        if de.get_ll_representation()['type'] == 'migration_rate_change':
            if de.matrix_index is None:
                m *= de.rate
                ms[gen] = copy.copy(m)
            else:
                i,j = de.matrix_index[0], de.matrix_index[1]
                m[i,j] = de.rate
                ms[gen] = copy.copy(m)
    return ms


def get_mig_transition(m):
    num_pops = len(m)
    inds = set_up_inds(num_pops)
    M = np.eye(num_pops*(num_pops+1)//2)
    for ii in range(num_pops):
        for jj in range(num_pops):
            if m[ii][jj] != 0: # nonzero rate of lineage from ii to jj
                for kk in range(num_pops):
                    pair_from = tuple(sorted([ii,kk]))
                    if kk == ii:
                        pair_to_single = tuple(sorted([jj,kk]))
                        pair_to_double = tuple(sorted([jj,jj]))
                        M[inds[pair_from], inds[pair_from]] -= 2*m[ii][jj]*(1-m[ii][jj])
                        M[inds[pair_to_single], inds[pair_from]] += 2*m[ii][jj]*(1-m[ii][jj])
                        M[inds[pair_from], inds[pair_from]] -= m[ii][jj] ** 2
                        M[inds[pair_to_double], inds[pair_from]] += m[ii][jj] ** 2
                    else:
                        pair_to = tuple(sorted([jj,kk]))
                        M[inds[pair_from], inds[pair_from]] -= m[ii][jj]
                        M[inds[pair_to], inds[pair_from]] += m[ii][jj]
    return M

def pulse_events(pop_config, demo_events):
    num_pops = len(pop_config)
    inds = set_up_inds(num_pops)
    pulses = {}
    for de in demo_events:
        if de.get_ll_representation()['type'] == 'mass_migration':
            P = np.eye(num_pops*(num_pops+1)//2)
            gen = int(np.ceil(de.time))
            ind_from = de.source
            ind_to = de.dest
            frac = de.proportion
            for ii in range(num_pops):
                for jj in range(ii,num_pops):
                    if ii == jj:
                        if ii == ind_from:
                            # prob move both = frac ** 2
                            P[inds[(ii,jj)],inds[(ii,jj)]] -= frac ** 2
                            P[inds[(ind_to,ind_to)],inds[(ii,jj)]] += frac ** 2
                            # prob move one = 2*frac*(1-frac)
                            P[inds[(ii,jj)],inds[(ii,jj)]] -= 2*frac*(1-frac)
                            pair_to = tuple(sorted([ii,ind_to]))
                            P[inds[pair_to],inds[(ii,jj)]] += 2*frac*(1-frac)
                            # prob move neither = (1-frac) ** 2
                    else:
                        if ii == ind_from:
                            # prob move the lineage = frac
                            P[inds[(ii,jj)],inds[(ii,jj)]] -= frac
                            pair_to = tuple(sorted([ind_to,jj]))
                            P[inds[pair_to],inds[(ii,jj)]] += frac
                            # prob don't move the lineage = 1-frac
                        elif jj == ind_from:
                            # prob move the lineage = frac
                            P[inds[(ii,jj)],inds[(ii,jj)]] -= frac
                            pair_to = tuple(sorted([ii,ind_to]))
                            P[inds[pair_to],inds[(ii,jj)]] += frac
                            # prob don't move the lineage = 1-frac
            pulses.setdefault(gen, [])
            pulses[gen].append(P)
    return pulses
