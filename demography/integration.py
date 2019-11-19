"""
Functions to integrate moments from the Demography.
Note that moments can handle up to five populations at any given time, so we
have a check that ensures that there are no more than five populations at all
times along the graph.
moments also requires samples to be defined.
Samples can only be defined for leave nodes, and the sampling happens at the
end of the node.

Some of these functions will be also used for integrating moments.LD ans dadi,
so we'll break those out into Integration.py.
"""

import numpy as np
import copy
from collections import defaultdict

try:
    import moments
    moments_installed = 1
except ImportError:
    moments_installed = 0

try:
    import moments.LD
    momentsLD_installed = 1
except ImportError:
    momentsLD_installed = 0

try:
    import dadi
    dadi_installed = 1
except:
    dadi_installed = 0


tol = 1e-14

"""
Functions shared between multiple simulation engines.

Events that are consistent between moments, moments.LD, and dadi:
Split
Merge
Pulse migrate
Marginalize
"""

def get_pulse_events(G):
    # get all the pulse events for each branch, their times, pop_to, and weights 
    pulses = {}
    for pop_from in G.nodes:
        if 'pulse' in G.nodes[pop_from]:
            pulses[pop_from] = []
            for pulse_event in G.nodes[pop_from]['pulse']:
                pulse_time = pulse_event[1] * G.nodes[pop_from]['T']
                pop_to = pulse_event[0]
                weight = pulse_event[2]
                pulses[pop_from].append([pulse_time, pop_to, weight])
            # order them chronologically
            pulses[pop_from] = sorted(pulses[pop_from])
    return pulses


def update_pulse_migration_events(pulse_migration_events, new_pops, t_epoch):
    for pop in new_pops:
        if pop in pulse_migration_events:
            for pulse_event in pulse_migration_events[pop]:
                pulse_event[0] -= t_epoch
    return pulse_migration_events


def get_pop_size_function(nus):
    """
    Every entry in nus is either a value (int/float) or list of length 2, as
        [nu0, growth_rate]
    growth_rate is found as r = np.log(nuF/nu0)/T, nu_func=nu0*exp(r*t)
    """
    nu = []
    any_fn = False
    for pop_nu in nus:
        if hasattr(pop_nu, '__len__'):
            any_fn = True
    
    if any_fn:
        for pop_nu in nus:
            if hasattr(pop_nu, '__len__'):
                any_fn = True
                nu0 = pop_nu[0]
                growth_rate = pop_nu[1]
                nu.append( lambda t, nu0=nu0, growth_rate=growth_rate: nu0 * np.exp(
                                                                        growth_rate*t) )
            else:
                nu.append( lambda t, pop_nu=pop_nu: pop_nu )
        return lambda t: [nu_func(t) for nu_func in nu]
    else:
        return nus


def get_migration_matrix(G, new_pops):
    # get the migration matrix for the list of new_pops
    M = np.zeros((len(new_pops), len(new_pops)))
    for ii, pop_from in enumerate(new_pops):
        if 'm' in G.nodes[pop_from]:
            for jj, pop_to in enumerate(new_pops):
                if pop_from == pop_to:
                    continue
                else:
                    if pop_to in G.nodes[pop_from]['m']:
                        M[ii][jj] = G.nodes[pop_from]['m'][pop_to]
    return M


def get_frozen_pops(G, new_pops):
    # get the list of frozen pops for new_pops
    frozen = []
    for pop in new_pops:
        if 'frozen' in G.nodes[pop]:
            if G.nodes[pop]['frozen'] == True:
                frozen.append(True)
            else:
                frozen.append(False)
        else:
            frozen.append(False)
    return frozen


def get_selfing_rates(G, new_pops):
    # get the selfing rates for the list of new_pops
    selfing = []
    for pop in new_pops:
        if 'selfing' in G.nodes[pop]:
            selfing.append(G.nodes[pop]['selfing'])
        else:
            selfing.append(0)
    if set(selfing) == {0}:
        return None
    else:
        return selfing


def reorder_events(new_events):
    """
    Place marginalize events at end of events
    """
    new_events_reordered = []
    for event in new_events:
        if event[0] != 'marginalize' and event[0] != 'pass':
            new_events_reordered.append(event)
    for event in new_events:
        if event[0] == 'pass':
            new_events_reordered.append(event)
    for event in new_events:
        if event[0] == 'marginalize':
            new_events_reordered.append(event)
    return new_events_reordered


def add_size_to_nus(G, pop, time_left):
    """
    adds either nu, or [nu0, growth_rate], where nu0 is the size at the beginning
    of the epoch use time_left to set nu0 to the size at the beginning of the epoch
    """
    if 'nu' in G.nodes[pop]:
        return G.nodes[pop]['nu']
    else:
        tt = G.nodes[pop]['T'] - time_left
        if 'nu0' in G.nodes[pop] and 'nuF' in G.nodes[pop]:
            growth_rate = np.log(G.nodes[pop]['nuF']/G.nodes[pop]['nu0']) / G.nodes[pop]['T']
            nu0 = G.nodes[pop]['nu0'] * np.exp(growth_rate * tt)
            return [nu0, growth_rate]
        elif 'growth_rate' in G.nodes[pop] and 'nuF' in G.nodes[pop]:
            nu0_pop = G.nodes[pop]['nuF'] * np.exp(-G.nodes[pop]['growth_rate']*G.nodes[pop]['T'])
            nu0 = nu0_pop * np.exp(growth_rate * tt)
            return [nu0, G.nodes[pop]['growth_rate']]
        elif 'growth_rate' in G.nodes[pop] and 'nu0' in G.nodes[pop]:
            nu0 = G.nodes[pop]['nu0'] * np.exp(G.nodes[pop]['growth_rate'] * tt)
            return [nu0, G.nodes[pop]['growth_rate']]


def get_new_pulse_times(new_pops, pulse_migration_events):
    new_pulse_times = []
    for this_pop in new_pops:
        if this_pop not in pulse_migration_events:
            new_pulse_times.append(1e10)
        else:
            temp_time = 1e10
            for pulse_event in pulse_migration_events[this_pop]:
                if pulse_event[0] > 0:
                    temp_time = min(temp_time, pulse_event[0])
            new_pulse_times.append(temp_time)
    return new_pulse_times


def get_next_events(dg, pulse_migration_events, time_left, present_pops):
    new_pops = [] # records populations present after any events
    new_times = [] # records time left to integrate of these pops
    new_nus = [] # records new pop sizes, and growth rate if given
    new_events = [] # records events to apply at end of epoch
    
    # if any population is at an end, apply events
    # else update its integration time left and pop size
    for ii,pop_time_left in enumerate(time_left):
        this_pop = present_pops[-1][ii]
        if pop_time_left < tol:
            if this_pop in dg.successors:
                # if it has children (1 or 2), carry over or split
                # check if children already in new_pops, 
                # if so, it's a merger (with weights)
                for child in dg.successors[this_pop]:
                    if child not in new_pops:
                        new_pops += dg.successors[this_pop]
                        new_times += [dg.G.nodes[child]['T'] for child in dg.successors[this_pop]]
                        new_nus += [add_size_to_nus(dg.G, child, dg.G.nodes[child]['T']) for child in dg.successors[this_pop]]
                if len(dg.successors[this_pop]) == 2:
                    child1, child2 = dg.successors[this_pop]
                    new_events.append( ('split', this_pop, child1, child2) )
                else:
                    child = dg.successors[this_pop][0]
                    # if the one child is a merger, need to specify,
                    # otherwise, event is a pass on
                    if dg.predecessors[child] == [this_pop]:
                        new_events.append( ('pass', this_pop, child ) )
                    else:
                        parent1, parent2 = dg.predecessors[child]
                        # check if we've already recorded this event 
                        # from the other parent
                        event_recorded = 0
                        for event in new_events:
                            if event[0] == 'merger' and event[1] == (parent1, parent2):
                                event_recorded = 1
                        if event_recorded == 1:
                            continue
                        else:
                            weights = (dg.G.get_edge_data(parent1,child)['weight'], dg.G.get_edge_data(parent2,child)['weight'])
                            new_events.append( ('merger', (parent1, parent2), weights, child) )
            else: # else no children and we eliminate it
                new_events.append( ('marginalize', this_pop) )
                continue
        else:
            new_pops += [this_pop]
            new_times += [pop_time_left]
            
            new_nus += [add_size_to_nus(dg.G, this_pop, pop_time_left)]
    
    # for previous pops, check if any have a pulse occuring now
    # we'll update times directly in the pulse_migration_events dictionary
    for this_pop in present_pops[-1]:
        if this_pop in pulse_migration_events:
            for pulse_event in pulse_migration_events[this_pop]:
                if pulse_event[0] < 0: # this pulse already occurred
                    continue
                elif pulse_event[0] < tol: # this pulse occurs now
                    new_events.append( ('pulse', this_pop, pulse_event[1], pulse_event[2]) )

    return new_pops, new_times, new_nus, new_events

### a lot of this should be used by moments as well
def get_moments_arguments(dg):
    """
    takes the demography object dg and returns present_pops, integration_times,
    nus, migration_matrices, frozen_pops, selfing_rates, and events
    
    For purposes of integration, we set time=0 to be the pre-event time in the
    ancestral population. Then every event (split, param change, merger) is a
    time since this reference time. The last time in the returned list is
    the stopping time for integration.

    This function goes from root to contemporary time, tracking time to
    integrate along each branch, and updating with each pulse migration event,
    split, merger, or marginalization. Each one of these events corresponds
    to an epoch, separated by events.
    """
    present_pops = [[dg.root]]
    integration_times = [dg.G.nodes[dg.root]['T']]
    nus = [[dg.G.nodes[dg.root]['nu']]]
    migration_matrices = [[0]]
    frozen_pops = [[False]]
    try:
        selfing_rates = [[dg.G.nodes[dg.root]['selfing']]]
    except KeyError:
        selfing_rates = [None]
    events = []

    # tracks time left on each current branch to integrate
    time_left = [dg.G.nodes[dg.root]['T']]
    
    # get all pulse migration events from the tree
    pulse_migration_events = get_pulse_events(dg.G)

    advance = True
    while advance == True:
        # if no pop has any time left and all pops are leaves, end it
        if (np.all(np.array(time_left) < tol) and 
                np.all([p not in dg.successors for p in present_pops[-1]])):
            advance = False
        else:
            new_pops, new_times, new_nus, new_events = get_next_events(dg, 
                pulse_migration_events, time_left, present_pops)
            
            # for new pops, get the times to the next pulse (ones that are positive)
            # (we already set negative the times to pulse if they have occured)
            new_pulse_times = get_new_pulse_times(new_pops, pulse_migration_events)
            
            # set integration time of this epoch to next pulse or end of population
            time_left = new_times
            t_epoch = min(min(time_left), min(new_pulse_times))
            integration_times.append(t_epoch)
            
            # update times left to next events
            time_left = [pop_time_left - t_epoch for pop_time_left in time_left]
            pulse_migration_events = update_pulse_migration_events(
                pulse_migration_events, new_pops, t_epoch)
            
            present_pops.append(new_pops)
            
            nus.append(new_nus)
            
            # get the migration matrix for this next epoch
            migration_matrices.append(get_migration_matrix(dg.G, new_pops))
            
            # get the list of frozen pops for this next epoch
            frozen_pops.append(get_frozen_pops(dg.G, new_pops))
            
            # get selfing rates for this nextepoch
            selfing_rates.append(get_selfing_rates(dg.G, new_pops))
                        
            # rearrange new events so that marginalize happens last
            events.append(reorder_events(new_events))
    
    return present_pops, integration_times, nus, migration_matrices, frozen_pops, selfing_rates, events


"""
Functions to evolve moments.LD to get LD statistics (no maximum number of pops)
"""

#### fix this
def ld_root_equilibrium(nu, theta, rho, pop_id, selfing=None):
    ss = moments.LD.Numerics.steady_state(theta=theta, rho=rho)
    Y = moments.LD.LDstats(ss, pop_ids=[pop_id], num_pops=1)
    ## XXX: would rather have this be analytic in moments.LD
    ## XXX: also need to include selfing
    if nu != 1.:
        Y.integrate([nu], 40, theta=theta, rho=rho)
    return Y


def evolve_ld(dg, rho=None, theta=None, pop_ids=None):
    """
    integrates moments.LD along the demography, which returns an LDStats
    object, for the given rhos, where rho=4*Ne*r.
    Note that theta in this model is 4*Ne*u, and not scaled by L, so it would
    be on the order of 0.001 instead of 1 (for example.
    """
    if theta == None:
        theta = dg.theta
    # check that theta is reasonable - warning if not
    
    # get the features from the dg
    # this ignores the features of the root used for initialization
    (present_pops, integration_times, nus, migration_matrices, frozen_pops,
        selfing_rates, events) = get_moments_arguments(dg)

    # initialize the LD stats at the root of the demography
    Y = ld_root_equilibrium(dg.G.nodes[dg.root]['nu'], theta, rho, dg.root)
    
    # step through the list of integrations and events
    for ii, (pops, T, nu, mig_mat, frozen, selfing) in enumerate(zip(
                present_pops, integration_times, nus, migration_matrices,
                frozen_pops, selfing_rates)):
        # first get the nu_function for this epoch
        nu_epoch = get_pop_size_function(nu)

        # integrate this epoch
        Y.integrate(nu_epoch, T, rho=rho, theta=theta, m=mig_mat,
                    selfing=selfing, frozen=frozen)

        # apply events
        if ii < len(events): ## want to change this to allow events at very end of simulation
            Y = ld_apply_events(Y, events[ii], present_pops[ii+1])

    # at the end, make sure the populations are in the right order
    if pop_ids is not None:
        Y = ld_rearrange_pops(Y, pop_ids)

    return Y


def ld_apply_events(Y, epoch_events, next_present_pops):
    """
    takes the LDstats object and applied events (such as splits, mergers,
    pulse migrations, and marginalizations)
    """
    if len(epoch_events) > 0:
        for e in epoch_events:
            if e[0] == 'pass':
                Y = ld_pass(Y, e[1], e[2])
            elif e[0] == 'split':
                Y = ld_split(Y, e[1], e[2], e[3])
            elif e[0] == 'merger':
                Y = ld_merge(Y, e[1], e[2], e[3]) # pops_from, weights, pop_to
            elif e[0] == 'pulse':
                Y = ld_pulse(Y, e[1], e[2], e[3]) # pop_from, pop_to, f
            elif e[0] == 'marginalize':
                Y = Y.marginalize(Y.pop_ids.index(e[1])+1)
    # make sure correct order of pops for the next epoch
    Y = ld_rearrange_pops(Y, next_present_pops)
    return Y


def ld_pass(Y, pop_from, pop_to):
    # just pass on populations, make sure keeping correct order of pop_ids
    new_ids = []
    for pid in Y.pop_ids:
        if pid == pop_from:
            new_ids.append(pop_to)
        else:
            new_ids.append(pid)
    Y.pop_ids = new_ids
    return Y


def ld_split(Y, parent, child1, child2):
    ids_from = Y.pop_ids
    Y = Y.split(ids_from.index(parent)+1)
    ids_to = ids_from + [child2]
    ids_to[ids_from.index(parent)] = child1
    Y.pop_ids = ids_to
    return Y


def ld_merge(Y, pops_to_merge, weights, pop_to):
    """
    Two populations (pops_to_merge = [popA, popB]) merge (with given weights)
    and form new population (pop_to).
    """
    pop1,pop2 = pops_to_merge
    ids_from = Y.pop_ids
    ids_to = copy.copy(ids_from)
    ids_to.pop(ids_to.index(pop1))
    ids_to.pop(ids_to.index(pop2))
    ids_to.append(pop_to)
    pop1_ind = ids_from.index(pop1)
    pop2_ind = ids_from.index(pop2)
    if pop1_ind < pop2_ind:
        Y = Y.merge(pop1_ind+1, pop2_ind+1, weights[0])
    else:
        Y = Y.merge(pop2_ind+1, pop1_ind+1, weights[1])
    Y.pop_ids = ids_to
    return Y


def ld_pulse(Y, pop_from, pop_to, pulse_weight):
    """
    A pulse migration event
    Different from merger, where the two parental populations are replaced by the 
    admixed population.
    Here, pulse events keep the current populations in place.
    """
    if pop_to in Y.pop_ids:
        ind_from = Y.pop_ids.index(pop_from)
        ind_to = Y.pop_ids.index(pop_to)
        Y = Y.pulse_migrate(ind_from+1, ind_to+1, pulse_weight)
    else:
        print("warning: pop_to in pulse_migrate isn't in present pops")
    return Y


def ld_rearrange_pops(Y, pop_order):
    current_order = Y.pop_ids
    for i in range(len(pop_order)):
        if current_order[i] != pop_order[i]:
            j = current_order.index(pop_order[i])
            while j > i:
                Y = Y.swap_pops(j,j+1)
                current_order = Y.pop_ids
                j -= 1
    if list(current_order) != list(pop_order):
        print("population ordering messed up")
        print(current_order)
        print(pop_order)
    return Y


"""
Functions to evolve using moments to get the frequency spectrum (max 5 pops)
"""

def moments_fs_root_equilibrium(ns0, nu, theta, pop_id, gamma=None, h=0.5):
    if gamma is not None:
        ss = moments.LinearSystem_1D.steady_state_1D(ns0, gamma=gamma, h=h) * theta
    else:
        ss = moments.LinearSystem_1D.steady_state_1D(ns0) * theta
    fs = moments.Spectrum(ss, pop_ids=[pop_id])
    if nu != 1.:
        fs.integrate([nu], 40, theta=theta, rho=rho, gamma=gamma, h=h)
    return fs


def check_max_five_pops(present_pops):
    for pps in present_pops:
        assert len(pps) <= 5, """to run moments, we can't have more than five populations at any given time"""
        
"""
For pulse events, we have enough lineages so that the pulse migration
approximation in moments.Manips.admix_inplace is accurate. If population 1
pulses into population 2 with expected proportion f, and we want n1 an n2
lineages left *after* the event, we need to keep n1_0 from before the event.
n1_0 = n1 + E[# lineages moved] + 2*stdev(# lineages moved)
     = n1 + f*n2 + 2*np.sqrt(n2*f*(1-f))
and take the ceiling of this.
"""

def num_lineages_pulse_pre(n_from, n_to, f):
    # n_from and n_to are the number of lineages after pulse
    #n_from_pre = n_from + np.ceil(f*n_to + 2*np.sqrt(n_to*f*(1-f))) # multiplied by two because one still raised warnings
    n_from_pre = n_from + np.ceil(n_to)
    return int(n_from_pre)


def num_lineages_pulse_post(n_from, n_to, f):
    #n_from_post = n_from - np.ceil(f*n_to + 2*np.sqrt(n_to*f*(1-f)))
    n_from_post = n_from - np.ceil(n_to)
    return int(n_from_post)


def get_number_needed_lineages(dg, pop_ids, sample_sizes, events):
    # for each population in dg, gives how many samples are needed in that population:
    # splits: add children to get parents
    # mergers: each parent same as child
    # pulse events: moments.Manips.admix_inplace vs moments.Manips.admix_into_new?
    # pass on: same as child
    # for pulse events, need a rule about how many inplace lineages remain each time,
    # probably based on the pulse probability
    lineages = defaultdict(int)
    for pop, ns in zip(pop_ids, sample_sizes):
        lineages[pop] = ns
    for leaf in dg.leaves:
        if leaf not in lineages:
            lineages[leaf]

    for epoch_events in events[::-1]:
        for e in epoch_events[::-1]:
            if e[0] == 'split':
                parent = e[1]
                child1 = e[2]
                child2 = e[3]
                lineages[parent] += lineages[child1] + lineages[child2]
            elif e[0] == 'merger':
                parent1 = e[1][0]
                parent2 = e[1][1]
                child = e[3]
                lineages[parent1] += lineages[child]
                lineages[parent2] += lineages[child]
            elif e[0] == 'pulse':
                pop_from = e[1]
                pop_to = e[2]
                f = e[3]
                n_from = lineages[pop_from]
                n_to = lineages[pop_to]
                n_from_needed = num_lineages_pulse_pre(n_from, n_to, f)
                lineages[pop_from] = n_from_needed
            elif e[0] == 'pass':
                parent = e[1]
                child = e[2]
                lineages[parent] = lineages[child]

    return lineages

def evolve_sfs_moments(dg, theta=None, pop_ids=None, 
                       sample_sizes=None, gamma=None, h=0.5):
    """
    integrates moments.LD along the demography, which returns an LDStats
    object, for the given rhos, where rho=4*Ne*r.
    Note that theta in this model is 4*Ne*u, and not scaled by L, so it would
    be on the order of 0.001 instead of 1 (for example.

    pop_ids and sample_sizes must be of same length, and in same order 
    """
    if theta == None:
        theta = dg.theta

    # get the features from the dg
    # this ignores the features of the root used for initialization
    (present_pops, integration_times, nus, migration_matrices, frozen_pops,
        selfing_rates, events) = get_moments_arguments(dg)

    check_max_five_pops(present_pops)

    num_lineages = get_number_needed_lineages(dg, pop_ids, sample_sizes, events)
    # initialize the LD stats at the root of the demography
    fs = moments_fs_root_equilibrium(num_lineages[dg.root], nus[0][0], theta, dg.root,
                                     gamma=gamma, h=h)

    # step through the list of integrations and events
    for ii, (pops, T, nu, mig_mat, frozen) in enumerate(zip(
                present_pops, integration_times, nus, migration_matrices, frozen_pops)):
        # first get the nu_function for this epoch
        nu_epoch = get_pop_size_function(nu)

        # integrate this epoch
        fs.integrate(nu_epoch, T, theta=theta, m=mig_mat,
                     frozen=frozen, gamma=gamma, h=h)

        # apply events
        if ii < len(events):
            fs = moments_apply_events(fs, events[ii], present_pops[ii+1], num_lineages)

    # at the end, make sure the populations are in the right order
    if pop_ids is not None:
        fs = moments_rearrange_pops(fs, pop_ids)

    return fs


def moments_apply_events(fs, epoch_events, next_present_pops, lineages):
    """
    takes the LDstats object and applied events (such as splits, mergers,
    pulse migrations, and marginalizations)
    """
    if len(epoch_events) > 0:
        for e in epoch_events:
            if e[0] == 'pass':
                fs = moments_pass(fs, e[1], e[2])
            elif e[0] == 'split':
                fs = moments_split(fs, e[1], e[2], e[3], lineages)
            elif e[0] == 'merger':
                fs = moments_merge(fs, e[1], e[2], e[3], lineages) # pops_from, weights, pop_to
            elif e[0] == 'pulse':
                fs = moments_pulse(fs, e[1], e[2], e[3]) # pop_from, pop_to, f
            elif e[0] == 'marginalize':
                fs = fs.marginalize(fs.pop_ids.index(e[1])+1)
    # make sure correct order of pops for the next epoch
    fs = moments_rearrange_pops(fs, next_present_pops)
    return fs


def moments_pass(fs, pop_from, pop_to):
    # just pass on populations, make sure keeping correct order of pop_ids
    new_ids = []
    for pid in fs.pop_ids:
        if pid == pop_from:
            new_ids.append(pop_to)
        else:
            new_ids.append(pid)
    fs.pop_ids = new_ids
    return fs


def moments_split(fs, parent, child1, child2, lineages):
    ids_from = fs.pop_ids
    data = copy.copy(fs)
    data.pop_ids = None
    if data.ndim == 1:
        fs_to = moments.Manips.split_1D_to_2D(data,
                    lineages[child1], lineages[child2])
    elif data.ndim == 2:
        if ids_from[0] == parent:
            fs_to = moments.Manips.split_2D_to_3D_1(data,
                        lineages[child1], lineages[child2])
        else:
            fs_to = moments.Manips.split_2D_to_3D_2(data,
                        lineages[child1], lineages[child2])
    elif data.ndim == 3:
        if ids_from[0] == parent:
            data = np.swapaxes(data, 0, 2)
            fs_to = moments.Manips.split_3D_to_4D_3(data,
                        lineages[child1], lineages[child2])
            data = np.swapaxes(data, 0, 2)
        elif ids_from[1] == parent:
            data = np.swapaxes(data, 1, 2)
            fs_to = moments.Manips.split_3D_to_4D_3(data,
                        lineages[child1], lineages[child2])
            data = np.swapaxes(data, 1, 2)
        elif ids_from[2] == parent:
            fs_to = moments.Manips.split_3D_to_4D_3(data,
                        lineages[child1], lineages[child2])
    elif data.ndim == 4:
        if ids_from[0] == parent:
            data = np.swapaxes(data, 0, 2)
            fs_to = moments.Manips.split_4D_to_5D_3(data,
                        lineages[child1], lineages[child2])
            data = np.swapaxes(data, 0, 2)
        elif ids_from[1] == parent:
            data = np.swapaxes(data, 1, 2)
            fs_to = moments.Manips.split_4D_to_5D_3(data,
                        lineages[child1], lineages[child2])
            data = np.swapaxes(data, 1, 2)
        elif ids_from[2] == parent:
            fs_to = moments.Manips.split_4D_to_5D_3(data,
                        lineages[child1], lineages[child2])
        elif ids_from[3] == parent:
            fs_to = moments.Manips.split_4D_to_5D_4(data,
                        lineages[child1], lineages[child2])

    ids_to = ids_from + [child2]
    ids_to[ids_from.index(parent)] = child1
    fs_to.pop_ids = ids_to
    return fs_to


def moments_merge(fs, pops_to_merge, weights, pop_to, lineages):
    """
    Two populations (pops_to_merge = [popA, popB]) merge (with given weights)
    and form new population (pop_to).
    """
    data = copy.copy(fs)
    data.pop_ids = None
    pop1,pop2 = pops_to_merge
    ids_from = fs.pop_ids
    ids_to = copy.copy(ids_from)
    ids_to.pop(ids_to.index(pop1))
    ids_to.pop(ids_to.index(pop2))
    ids_to.append(pop_to)
    pop1_ind = ids_from.index(pop1)
    pop2_ind = ids_from.index(pop2)
    # use admix_into_new, and then marginalize the parental populations
    if pop1_ind < pop2_ind:
        data = moments.Manips.admix_into_new(data, pop1_ind, pop2_ind,
                    lineages[pop_to], weights[0])
    else:
        data = moments.Manips.admix_into_new(data, pop2_ind, pop1_ind,
                    lineages[pop_to], weights[1])
    data.pop_ids = ids_to
    return data


def moments_pulse(fs, pop_from, pop_to, pulse_weight):
    """
    A pulse migration event
    Different from merger, where the two parental populations are replaced by
    the admixed population.
    Here, pulse events keep the current populations in place.
    """
    if pop_to in fs.pop_ids:
        ind_from = fs.pop_ids.index(pop_from)
        ind_to = fs.pop_ids.index(pop_to)
        n_from = fs.shape[ind_from] - 1
        n_to = fs.shape[ind_to] - 1
        n_from_post = num_lineages_pulse_post(n_from, n_to, pulse_weight)
        fs = moments.Manips.admix_inplace(fs, ind_from, ind_to, n_from_post, pulse_weight)
    else:
        print("warning: pop_to in pulse_migrate isn't in present pops")
    return fs


def moments_rearrange_pops(fs, pop_order):
    curr_order = fs.pop_ids
    for i in range(len(pop_order)):
        if curr_order[i] != pop_order[i]:
            j = curr_order.index(pop_order[i])
            while j > i:
                fs = np.swapaxes(fs,j-1,j)
                curr_order[j-1], curr_order[j] = curr_order[j], curr_order[j-1]
                j -= 1
    fs.pop_ids = curr_order
    if list(curr_order) != list(pop_order):
        print("population ordering messed up")
        print(curr_order)
        print(pop_order)
    return fs

"""
Functions to evolve using dadi to get the frequency spectrum (max 3 pops)
"""

