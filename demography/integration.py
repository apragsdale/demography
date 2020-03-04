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
from . import util
import demography
import networkx as nx

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


def get_pop_size_function(nus, engine='moments'):
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
        if engine == 'moments':
            return lambda t: [nu_func(t) for nu_func in nu]
        elif engine == 'dadi':
            return nu
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
                        M[jj][ii] = G.nodes[pop_from]['m'][pop_to]
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
                if len(dg.successors[this_pop]) >= 2:
                    children = dg.successors[this_pop]
                    new_events.append( tuple(['split', this_pop] + children ) )
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


def augment_with_frozen(dg, sampled_pops):
    """
    For populations in sampled pops, check if all accumulated times are within
    tol of each other. If not, extend the populations that fall short with
    frozen populations, adding a new frozen pops that we pass.
    What do we do with names? A: make the ancient population <name>_pre, and
    set the frozen pop to <name>.
    """
    dg_out = copy.deepcopy(dg)
    accumulated_times = util.get_accumulated_times(dg)
    max_time = max(accumulated_times.values())
    if np.all([abs(accumulated_times[pop]-max_time) < tol for pop in sampled_pops]):
        return dg_out
    else:   
        G = dg_out.G
        for pop in sampled_pops:
            # check if this pop is contemporary
            if abs(accumulated_times[pop]-max_time) < tol:
                continue
            else:
                time_left = max_time - accumulated_times[pop]
                # relabel the population node
                pop_pre = pop + '_pre'
                assert pop_pre not in G.nodes, "naming issue..."
                mapping = {pop: pop_pre}
                G = nx.relabel_nodes(G, mapping)
                # add frozen population
                G.add_node(pop, nu=1, T=time_left, frozen=True)
                # add edge between ancient and frozen population
                G.add_edge(pop_pre, pop)
        # create dg object and add relevant attributes that were present on dg
        dg_out = demography.DemoGraph(G, Ne=dg.Ne, mutation_rate=dg.mutation_rate)
        return dg_out

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


def evolve_ld(dg, rho=None, theta=None, pop_ids=None, augment=True):
    """
    integrates moments.LD along the demography, which returns an LDStats
    object, for the given rhos, where rho=4*Ne*r.
    Note that theta in this model is 4*Ne*u, and not scaled by L, so it would
    be on the order of 0.001 instead of 1.
    pop_ids specifies the leaf populations we want to predict data for.
    """
    assert momentsLD_installed, "moments.LD is not installed"

    if theta == None:
        theta = dg.theta
    # check that theta is reasonable - warning if not
    
    if augment is True:
        dg_sim = augment_with_frozen(dg, pop_ids)
    else:
        dg_sim = dg
    
    # get the features from the dg
    # this ignores the features of the root used for initialization
    (present_pops, integration_times, nus, migration_matrices, frozen_pops,
        selfing_rates, events) = get_moments_arguments(dg_sim)

    # initialize the LD stats at the root of the demography
    Y = ld_root_equilibrium(dg_sim.G.nodes[dg_sim.root]['nu'], theta, rho, dg_sim.root)
    
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
    Y = ld_marginalize_nonpresent(Y, pop_ids)

    if pop_ids is not None:
        Y = ld_rearrange_pops(Y, pop_ids)

    return Y


def ld_marginalize_nonpresent(Y, pop_ids):
    pops_to_marginalize = []
    for pop in Y.pop_ids:
        if pop not in pop_ids:
            pops_to_marginalize.append(pop)
    
    marge_indexes = [Y.pop_ids.index(pop)+1 for pop in pops_to_marginalize]
    if len(marge_indexes) > 0:
        Y = Y.marginalize(marge_indexes)
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
                Y = ld_split(Y, e[1], e[2:]) # pass list of children (typically 2, but could be more)
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


def ld_split(Y, parent, children):
    ids_from = Y.pop_ids
    for ii in range(len(children)-1):
        Y = Y.split(ids_from.index(parent)+1)
    ids_to = ids_from + list(children[1:])
    ids_to[ids_from.index(parent)] = children[0]
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
        print(f"warning: pop_to ({pop_to}) in pulse_migrate isn't in present pops {Y.pop_ids}")
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

### this should be fixed for when nu != 1 to not have to integrate out
def moments_fs_root_equilibrium(ns0, nu, theta, pop_id, gamma=None, h=0.5,
                                reversible=False):
    if reversible is False:
        if gamma is not None:
            ss = moments.LinearSystem_1D.steady_state_1D(ns0, gamma=gamma, h=h,
                                                         theta=theta)
        else:
            ss = moments.LinearSystem_1D.steady_state_1D(ns0, theta=theta)
        fs = moments.Spectrum(ss, pop_ids=[pop_id])
    else:
        assert h==0.5, "finite genome integration can only be done with h=1/2"
        if gamma is None:
            gamma = 0.0
        ss = moments.LinearSystem_1D.steady_state_1D_reversible(ns0,
                                                                gamma=gamma,
                                                                theta_fd=theta,
                                                                theta_bd=theta)
        fs = moments.Spectrum(ss, pop_ids=[pop_id], mask_corners=False)

    if nu != 1.:
        if reversible is False:
            fs.integrate([nu], 40, theta=theta, gamma=gamma, h=h)
        else:
            fs.integrate([nu], 40, gamma=gamma, h=h,
                         finite_genome=True, theta_fd=theta, theta_bd=theta)

    return fs


def check_max_num_pops(present_pops, num):
    for pps in present_pops:
        assert len(pps) <= num, """to run this method, we can't have more \
                                than five populations at any given time"""
        
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
                children = e[2:]
                lineages[parent] += sum([lineages[child] for child in children])
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
                       sample_sizes=None, gamma=None, h=0.5,
                       reversible=False, augment=True):
    """
    pop_ids and sample_sizes must be of same length, and in same order 
    """
    assert moments_installed, "moments is not installed"

    if theta == None:
        if dg.theta is not None:
            theta = dg.theta
        else:
            theta = 1
    
    if gamma is None:
        gamma = 0.0

    if augment is True:
        dg_sim = augment_with_frozen(dg, pop_ids)
    else:
        dg_sim = dg

    # get the features from the dg
    # this ignores the features of the root used for initialization
    (present_pops, integration_times, nus, migration_matrices, frozen_pops,
        selfing_rates, events) = get_moments_arguments(dg_sim)

    check_max_num_pops(present_pops, 5)

    num_lineages = get_number_needed_lineages(dg_sim, pop_ids, sample_sizes,
                                              events)
    # initialize the sfs at the root of the demography
    fs = moments_fs_root_equilibrium(num_lineages[dg_sim.root], nus[0][0], 
                                     theta, dg_sim.root, gamma=gamma, h=h,
                                     reversible=reversible)

    # step through the list of integrations and events
    for ii, (pops, T, nu, mig_mat, frozen) in enumerate(zip(
                present_pops, integration_times, nus, migration_matrices,
                frozen_pops)):
        # first get the nu_function for this epoch
        nu_epoch = get_pop_size_function(nu)

        # integrate this epoch
        if reversible is False:
            fs.integrate(nu_epoch, T, theta=theta, m=mig_mat,
                         frozen=frozen, gamma=[gamma]*len(nu), h=[h]*len(nu))
        elif reversible is True:
            fs.integrate(nu_epoch, T, m=mig_mat,
                         frozen=frozen, gamma=[gamma]*len(nu), h=[h]*len(nu),
                         finite_genome=reversible,
                         theta_fd=theta, theta_bd=theta)
        
        # apply events
        if ii < len(events):
            fs = moments_apply_events(fs, events[ii], present_pops[ii+1],
                                      num_lineages)

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
                fs = moments_split(fs, e[1], e[2:], lineages)
            elif e[0] == 'merger':
                fs = moments_merge(fs, e[1], e[2], e[3], lineages) # pops_from, weights, pop_to
            elif e[0] == 'pulse':
                fs = moments_pulse(fs, e[1], e[2], e[3]) # pop_from, pop_to, f
            elif e[0] == 'marginalize':
                fs = fs.marginalize([fs.pop_ids.index(e[1])])
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

def moments_split(fs, parent, children, lineages):
    fs.pop_ids[fs.pop_ids.index(parent)] = children[0]
    for jj,child2 in enumerate(children[1:]):
        temp_lineages = {child2: lineages[child2]}
        temp_lineages[children[0]] = lineages[children[0]]
        if jj+2 < len(children):
            temp_lineages[children[0]] += sum([lineages[c] for c in children[jj+2:]])
        fs = moments_split_2_way(fs, children[0], children[0], child2, temp_lineages)
    return fs

def moments_split_2_way(fs, parent, child1, child2, lineages):
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
            fs_to = moments.Manips.split_3D_to_4D_1(data,
                        lineages[child1], lineages[child2])
        elif ids_from[1] == parent:
            fs_to = moments.Manips.split_3D_to_4D_2(data,
                        lineages[child1], lineages[child2])
        elif ids_from[2] == parent:
            fs_to = moments.Manips.split_3D_to_4D_3(data,
                        lineages[child1], lineages[child2])
    elif data.ndim == 4:
        if ids_from[0] == parent:
            fs_to = moments.Manips.split_4D_to_5D_1(data,
                        lineages[child1], lineages[child2])
        elif ids_from[1] == parent:
            fs_to = moments.Manips.split_4D_to_5D_2(data,
                        lineages[child1], lineages[child2])
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
    # use admix_into_new
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
        print(f"warning: pop_to ({pop_to}) in pulse_migrate isn't in present pops {fs.pop_ids}")
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

def evolve_sfs_dadi(dg, pts, theta=None, pop_ids=None, 
                       sample_sizes=None, gamma=None, h=0.5):
    """
    pop_ids and sample_sizes must be of same length, and in same order 
    pts: either integer for number of points to use in integration (using
         default grid), or list of three points to extrapolate over
    
    """
    assert dadi_installed, "dadi is not installed"
    
    check_pts_sample_size(pts, sample_sizes, buffer=5)

    if theta == None:
        if dg.theta is not None:
            theta = dg.theta
        else:
            theta = 1

    dg_sim = augment_with_frozen(dg, pop_ids)

    # get the features from the dg
    # this ignores the features of the root used for initialization
    (present_pops, integration_times, nus, migration_matrices, frozen_pops,
        selfing_rates, events) = get_moments_arguments(dg_sim)

    check_max_num_pops(present_pops, 3)

    # initialize phi(s) at the root of the demography
    grid = get_dadi_grid(pts)
    phi = dadi_root_equilibrium(pts, grid, nu=nus[0][0], theta=theta,
                                gamma=gamma, h=h)

    # step through the list of integrations and events
    for ii, (pops, T, nu, mig_mat, frozen) in enumerate(zip(
                present_pops, integration_times, nus, migration_matrices,
                frozen_pops)):
        # first get the nu_function for this epoch
        nu_epoch = get_pop_size_function(nu, engine='dadi')

        # integrate this epoch
        phi = integrate_phis(phi, grid, pts, nu_epoch, T, theta=theta,
                             m=mig_mat, frozen=frozen, gamma=gamma, h=h)

        # apply events
        # question: does 
        if ii < len(events):
            phi = dadi_apply_events(phi, grid, pts, events[ii], 
                                    present_pops[ii], present_pops[ii+1])

    # rearrange so that present_pops[-1] == pop_ids
    phi = dadi_rearrange_pops(phi, pts, present_pops[-1], pop_ids)
    spectrum = sample_dadi(phi, grid, pts, sample_sizes)
    spectrum.pop_ids = pop_ids

    return spectrum


def get_dadi_grid(pts):
    if hasattr(pts, "__len__"):
        xx = [dadi.Numerics.default_grid(pt) for pt in pts]
    else:
        xx = dadi.Numerics.default_grid(pts)
    return xx

def dadi_root_equilibrium(pts, xx, nu=1.0, theta=1.0, gamma=0.0, h=0.5):
    if hasattr(pts, "__len__"):
        phi = [dadi.PhiManip.phi_1D(x, nu=nu, theta0=theta, gamma=gamma, h=h)
               for x in xx]
    else:
        phi = dadi.PhiManip.phi_1D(xx, nu=nu, theta0=theta, gamma=gamma, h=h)
    return phi


def check_pts_sample_size(pts, sample_sizes, buffer=10):
    max_ns = np.max(sample_sizes)
    if hasattr(pts, "__len__") is False:
        pts = [pts]
    assert np.all([pt >= max_ns + buffer for pt in pts]), "pts too small"


# call different Integration functions depending on the dimension
def integrate_dadi(this_phi, this_xx, T, nu, m, gamma, h, theta, frozen):
    if this_phi.ndim == 1:
        this_phi = dadi.Integration.one_pop(this_phi, this_xx, T, nu=nu[0],
                                 gamma=gamma, h=h, theta0=theta, frozen=frozen[0])
    elif this_phi.ndim == 2:
        this_phi = dadi.Integration.two_pops(this_phi, this_xx, T, nu1=nu[0],
                                  nu2=nu[1], m12=m[0][1], m21=m[1][0],
                                  gamma1=gamma, gamma2=gamma, h1=h, h2=h,
                                  theta0=theta, frozen1=frozen[0],
                                  frozen2=frozen[1])
    elif this_phi.ndim == 3:
        this_phi = dadi.Integration.three_pops(this_phi, this_xx, T, nu1=nu[0],
                                  nu2=nu[1], nu3=nu[2], m12=m[0][1], 
                                  m13=m[0][2], m21=m[1][0], m23=m[1][2], 
                                  m31=m[2][0], m32=m[2][1], gamma1=gamma,
                                  gamma2=gamma, gamma3=gamma, h1=h, h2=h,
                                  h3=h, theta0=theta, frozen1=frozen[0], 
                                  frozen2=frozen[1], frozen3=frozen[2])
    else:
        raise Exception("phi dimension cannot be larger than three")
    return this_phi

def integrate_phis(phi, xx, pts, nu, T, theta=1., m=None, frozen=None,
                   gamma=0, h=0.5):
    if hasattr(pts, "__len__"):
        for ii, (this_phi, this_xx) in enumerate(zip(phi, xx)):
            phi[ii] = integrate_dadi(this_phi, this_xx, T, nu, m, gamma, h,
                                     theta, frozen)
    else:
        phi = integrate_dadi(phi, xx, T, nu, m, gamma, h, theta, frozen)
    return phi


def sample_dadi(phi, grid, pts, ns):
    if hasattr(pts, "__len__"):
        spectra = [dadi.Spectrum.from_phi(this_phi, ns, 
                                          tuple([this_grid]*this_phi.ndim)) 
                   for (this_phi, this_grid) in zip(phi,grid)]
        fs = dadi.Numerics.quadratic_extrap(spectra, [x[1] for x in grid])
    else:
        fs = dadi.Spectrum.from_phi(phi, ns, tuple([grid]*phi.ndim))
    return fs


def dadi_apply_events(phi, grid, pts, epoch_events, prev_present_pops,
                      next_present_pops):
    """
    takes the LDstats object and applied events (such as splits, mergers,
    pulse migrations, and marginalizations)
    """
    if len(epoch_events) > 0:
        for e in epoch_events:
            if e[0] == 'pass':
                phi = dadi_pass(phi, pts, e[1], e[2], prev_present_pops,
                               next_present_pops)
            elif e[0] == 'split':
                phi = dadi_split(phi, grid, pts, e[1], e[2], e[3],
                                 prev_present_pops, next_present_pops)
            elif e[0] == 'merger':
                phi = dadi_merge(phi, grid, pts, e[1], e[2], e[3],
                                 prev_present_pops, next_present_pops)
            elif e[0] == 'pulse':
                phi = dadi_pulse(phi, grid, pts, e[1], e[2], e[3],
                                 prev_present_pops, next_present_pops)
            elif e[0] == 'marginalize':
                phi = dadi_marginalize(phi, grid, pts, e[1],
                            prev_present_pops, next_present_pops)
    return phi


def dadi_marginalize(phi, grid, pts, pop_to_remove, prev_present_pops,
                     next_present_pops):
    index_to_remove = prev_present_pops.index(pop_to_remove)
    new_ids = []
    for pid in prev_present_pops:
        if pid != pop_to_remove:
            new_ids.append(pid)

    if hasattr(pts, "__len__"):
        phi = [dadi.PhiManip.remove_pop(this_phi, this_grid, index_to_remove+1)
               for (this_phi, this_grid) in zip(phi, grid)]
    else:
        phi = dadi.PhiManip.remove_pop(phi, grid, index_to_remove+1)
    phi = dadi_rearrange_pops(phi, pts, new_ids, next_present_pops)
    return phi


def dadi_pass(phi, pts, pop_from, pop_to, prev_present_pops, next_present_pops):
    # just pass on populations, make sure keeping correct order of pop_ids
    new_ids = []
    for pid in prev_present_pops:
        if pid == pop_from:
            new_ids.append(pop_to)
        else:
            new_ids.append(pid)

    phi = dadi_rearrange_pops(phi, pts, new_ids, next_present_pops)
    return phi


def dadi_split(phi, grid, pts, parent, child1, child2, prev_present_pops,
               next_present_pops):
    if hasattr(pts, "__len__"):
        if phi[0].ndim == 1:
            phi = [dadi.PhiManip.phi_1D_to_2D(this_grid, this_phi) for 
                   this_grid, this_phi in zip(grid,phi)]
            new_ids = [child1, child2]
        elif phi[0].ndim == 2:
            if prev_present_pops[0] == parent:
                phi = [dadi.PhiManip.phi_2D_to_3D_split_1(this_grid, this_phi)
                       for this_grid, this_phi in zip(grid,phi)]
            else:
                phi = [dadi.PhiManip.phi_2D_to_3D_split_2(this_grid, this_phi)
                       for this_grid, this_phi in zip(grid,phi)]
            new_ids = prev_present_pops + [child2]
            new_ids[prev_present_pops.index(parent)] = child1
    else:
        if phi.ndim == 1:
            phi = dadi.PhiManip.phi_1D_to_2D(grid, phi)
            new_ids = [child1, child2]
        elif phi.ndim == 2:
            if prev_present_pops[0] == parent:
                phi = dadi.PhiManip.phi_2D_to_3D_split_1(grid, phi)
            else:
                phi = dadi.PhiManip.phi_2D_to_3D_split_2(grid, phi)
            new_ids = prev_present_pops + [child2]
            new_ids[prev_present_pops.index(parent)] = child1

    phi = dadi_rearrange_pops(phi, pts, new_ids, next_present_pops)
    return phi


def dadi_merge(phi, grid, pts, pops_to_merge, weights, pop_to,
               prev_present_pops, next_present_pops):
    """
    Two populations (pops_to_merge = [popA, popB]) merge (with given weights)
    and form new population (pop_to).

    If two populations, create the third pop and and then marginalize the
    first two.

    If three populations to start, places new merged pop in place of one of
    the populations, and we marginalize the other.
    """
    pop1,pop2 = pops_to_merge
    if len(prev_present_pops) == 2:
        if prev_present_pops[0] == pop1:
            f = weights[0]
        elif prev_present_pops[0] == pop2:
            f = weights[1]
        # end up with one pop
        if hasattr(pts, "__len__"):
            for ii,(this_phi, this_grid) in enumerate(zip(phi, grid)):
                this_phi = dadi.PhiManip.phi_2D_to_3D_admix(this_phi, f,
                                    this_grid, this_grid, this_grid)
                this_phi = dadi.PhiManip.remove_pop(this_phi, this_grid, 1)
                this_phi = dadi.PhiManip.remove_pop(this_phi, this_grid, 1)
                phi[ii] = this_phi
        else:
            phi = dadi.PhiManip.phi_2D_to_3D_admix(phi, f,
                                    grid, grid, grid)
            phi = dadi.PhiManip.remove_pop(phi, grid, 1)
            phi = dadi.PhiManip.remove_pop(phi, grid, 1)
    elif len(prev_present_pops) == 3:
        # need to admix one into the other and then marginalize the unadmixed
        # we'll admix pop2 into pop1 with weight weights[1]
        # then we'll remove pop2 index, left with merged pop in place of pop1
        # use dadi_pulse for the first part

        # pulse pop2 into pop1, leaving order unchanged
        phi = dadi_pulse(phi, grid, pts, pop2, pop1, weights[1],
                         prev_present_pops, prev_present_pops)

        # get the new_ids after marginalizing pop2
        ind_to_remove = prev_present_pops.index(pop2)
        new_ids = [pop for pop in prev_present_pops if pop != pop2]

        # marginalize
        phi = dadi_marginalize(phi, grid, pts, pop2, prev_present_pops,
                     new_ids)

        new_ids[new_ids.index(pop1)] = pop_to        
        # reorder so the new ids are in correct order for next_present_pops
        phi = dadi_rearrange_pops(phi, pts, new_ids, next_present_pops)
    return phi


def dadi_pulse(phi, grid, pts, pop_from, pop_to, pulse_weight,
                                 prev_present_pops, next_present_pops):
    """
    A pulse migration event
    Different from merger, where the two parental populations are replaced by
    the admixed population.
    Here, pulse events keep the current populations in place.
    """
    assert np.all(prev_present_pops == next_present_pops)
    if pop_to in prev_present_pops:
        ind_from = prev_present_pops.index(pop_from)
        ind_to = prev_present_pops.index(pop_to)
        if len(prev_present_pops) == 2:
            if ind_from == 0 and ind_to == 1: 
                # pulse 1 into 2
                if hasattr(pts, "__len__"):
                    phi = [dadi.PhiManip.phi_2D_admix_1_into_2(this_phi,
                                pulse_weight, this_grid, this_grid)
                           for this_phi, this_grid in zip(phi,grid)]
                else:
                    phi = dadi.PhiManip.phi_2D_admix_1_into_2(phi,
                                pulse_weight, grid, grid)
            elif ind_from == 1 and ind_to == 0:
                # pulse 2 into 1
                if hasattr(pts, "__len__"):
                    phi = [dadi.PhiManip.phi_2D_admix_2_into_1(this_phi,
                                pulse_weight, this_grid, this_grid)
                           for this_phi, this_grid in zip(phi,grid)]
                else:
                    phi = dadi.PhiManip.phi_2D_admix_2_into_1(phi,
                                pulse_weight, grid, grid)
        elif len(prev_present_pops) == 3:
            if ind_from == 0 and ind_to == 1:
                # pulse 1 into 2
                # (phi, f1,f3, xx,yy,zz), f1=pulse_weight, f3=0
                if hasattr(pts, "__len__"):
                    phi = [dadi.PhiManip.phi_3D_admix_1_and_3_into_2(this_phi,
                                pulse_weight, 0, this_grid, this_grid,
                                this_grid)
                           for this_phi, this_grid in zip(phi,grid)]
                else:
                    phi = dadi.PhiManip.phi_3D_admix_1_and_3_into_2(phi,
                                pulse_weight, 0, grid, grid, grid)
            elif ind_from == 0 and ind_to == 2:
                # pulse 1 into 3
                # (phi, f1,f2, xx,yy,zz), f1=pulse_weight, f2=0
                if hasattr(pts, "__len__"):
                    phi = [dadi.PhiManip.phi_3D_admix_1_and_2_into_3(this_phi,
                                pulse_weight, 0, this_grid, this_grid,
                                this_grid)
                           for this_phi, this_grid in zip(phi,grid)]
                else:
                    phi = dadi.PhiManip.phi_3D_admix_1_and_2_into_3(phi,
                                pulse_weight, 0, grid, grid, grid)
                
            elif ind_from == 1 and ind_to == 0:
                # pulse 2 into 1
                # (phi, f2,f3, xx,yy,zz), f2=pulse_weight, f3=0
                if hasattr(pts, "__len__"):
                    phi = [dadi.PhiManip.phi_3D_admix_2_and_3_into_1(this_phi,
                                pulse_weight, 0, this_grid, this_grid,
                                this_grid)
                           for this_phi, this_grid in zip(phi,grid)]
                else:
                    phi = dadi.PhiManip.phi_3D_admix_2_and_3_into_1(phi,
                                pulse_weight, 0, grid, grid, grid)
                
            elif ind_from == 1 and ind_to == 2:
                # pulse 2 into 3
                # (phi, f1,f2, xx,yy,zz), f1=0, f2=pulse_weight
                if hasattr(pts, "__len__"):
                    phi = [dadi.PhiManip.phi_3D_admix_1_and_2_into_3(this_phi,
                                0, pulse_weight, this_grid, this_grid,
                                this_grid)
                           for this_phi, this_grid in zip(phi,grid)]
                else:
                    phi = dadi.PhiManip.phi_3D_admix_1_and_2_into_3(phi,
                                0, pulse_weight, grid, grid, grid)
                
            elif ind_from == 2 and ind_to == 0:
                # pulse 3 into 1
                # (phi, f2,f3, xx,yy,zz), f2=0, f3=pulse_weight
                if hasattr(pts, "__len__"):
                    phi = [dadi.PhiManip.phi_3D_admix_2_and_3_into_1(this_phi,
                                0, pulse_weight, this_grid, this_grid,
                                this_grid)
                           for this_phi, this_grid in zip(phi,grid)]
                else:
                    phi = dadi.PhiManip.phi_3D_admix_2_and_3_into_1(phi,
                                0, pulse_weight, grid, grid, grid)
                
            elif ind_from == 2 and int_to == 1:
                # pulse 3 into 2
                # (phi, f1,f3, xx,yy,zz), f1=0, f3=pulse_weight
                if hasattr(pts, "__len__"):
                    phi = [dadi.PhiManip.phi_3D_admix_1_and_3_into_2(this_phi,
                                0, pulse_weight, this_grid, this_grid,
                                this_grid)
                           for this_phi, this_grid in zip(phi,grid)]
                else:
                    phi = dadi.PhiManip.phi_3D_admix_1_and_3_into_2(phi,
                                0, pulse_weight, grid, grid, grid)
                
    else:
        print(f"warning: {pop_to} isn't in present pops, cannot pulse migrate")
    
    return phi


def dadi_rearrange_pops(phi, pts, curr_order, pop_order):
    """
    swap axes to get this pop order to match next_pop_order
    """
    for i in range(len(pop_order)):
        if curr_order[i] != pop_order[i]:
            j = curr_order.index(pop_order[i])
            while j > i:
                if hasattr(pts, "__len__"):
                    phi = [np.swapaxes(this_phi, j-1, j) for this_phi in phi]
                else:
                    phi = np.swapaxes(phi, j-1, j)
                curr_order[j-1], curr_order[j] = curr_order[j], curr_order[j-1]
                j -= 1
    if list(curr_order) != list(pop_order):
        print("population ordering messed up in dadi_rearrange_pops")
        print(curr_order)
        print(pop_order)
    return phi
