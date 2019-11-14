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
                nu.append( lambda t, nu0=nu0, growth_rate=growth_rate: nu0 * np.exp(growth_rate*t) )
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
        if event[0] != 'marginalize' and event[0] != 'pass:
            new_events_reordered.append(event)
    for event in new_events:
        if event[0] == 'pass:
            new_events_reordered.append(event)
    for event in new_events:
        if event[0] == 'marginalize':
            new_events_reordered.append(event)
    return new_events_reordered


def add_size_to_nus(G, pop, time_left):
    """
    adds either nu, or [nu0, growth_rate], where nu0 is the size at the beginning of the epoch
    use time_left to set nu0 to the size at the beginning of the epoch
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


def get_next_events(dg, pulse_migration_events):
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

"""
Functions to evolve moments.LD to get LD statistics (no maximum number of pops)
"""


### a lot of this should be used by moments as well
def get_moments_ld_arguments(dg):
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

    note to self: I could do a lot of work to modularize and test this function 
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
            new_pops, new_times, new_nus, new_events = get_next_events(dg, pulse_migration_events)
            
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


### ^^^^ all to do with parsing the graph for moments/momentsLD

#### fix this
def root_equilibrium(nu, theta, rho, pop_id, selfing=None):
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
        selfing_rates, events) = get_moments_ld_arguments(dg)

    # initialize the LD stats at the root of the demography
    Y = root_equilibrium(dg.G.nodes[dg.root]['nu'], theta, rho, dg.root)
    
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
            Y = apply_events(Y, events[ii], present_pops[ii+1])

    # at the end, make sure the populations are in the right order
    if pop_ids is not None:
        Y = rearrange_pops(Y, pop_ids)

    return Y


def apply_events(Y, epoch_events, next_present_pops):
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
    Y = rearrange_pops(Y, next_present_pops)
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


def rearrange_pops(Y, pop_order):
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


"""
Functions to evolve using dadi to get the frequency spectrum (max 3 pops)
"""

