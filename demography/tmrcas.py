import numpy as np
import copy
from . import integration
from itertools import combinations_with_replacement
from scipy.special import gammaln


def compute_tmrcas(dg, pop_ids, Ne, order, selfing):
    """
    selfing is the default selfing rate. If None, it is ignored. It does not
    override selfing rates specified per branch.
    """
    # check if any branches have selfing rates set
    any_selfing = check_selfing(dg)
    # if default selfing rate is specified, ensure we simulate under selfing
    if selfing is not None: 
        any_selfing = True
    # if some branches have selfing rates and default selfing rate is not
    # set, it defaults to 0.0
    if any_selfing and selfing is None:
        selfing = 0.0
    
    if any_selfing:
        T = integrate_tmrca_selfing(dg, Ne, order, selfing, pop_ids)
    else:
        T = integrate_tmrca(dg, Ne, order, pop_ids)
    
    return T


def check_selfing(dg):
    any_selfing = False
    for node in dg.G.nodes:
        if 'selfing' in dg.G.nodes[node]:
            any_selfing = True
    return any_selfing

###
### functions for integrating TMRCAs
###


def _choose(n, i):
    return np.exp(gammaln(n+1)- gammaln(n-i+1) - gammaln(i+1))


def gens_in_interval_as_t(Ne, T_elapsed, T, current_gen, integ_time):
    tt = []
    while current_gen/2/Ne < integ_time+T:
        tt.append(T_elapsed-integ_time)
        T_elapsed += 1./2/Ne
        current_gen += 1
    integ_time += T
    return np.array(tt), T_elapsed, current_gen, integ_time

def get_pop_sizes(Ne, nus, T_elapsed, T, current_gen, integ_time):
    N_func = []
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
                N_func.append( lambda t, nu0=nu0, growth_rate=growth_rate: 
                                Ne * nu0 * np.exp(growth_rate*t) )
            else:
                N_func.append( lambda t, pop_nu=pop_nu:
                                Ne * pop_nu * np.ones(len(t)) )

        # get times for this epoch
        tt, T_elapsed, current_gen, integ_T = gens_in_interval_as_t(Ne,
            T_elapsed, T, current_gen, integ_time)

        Ns = [N_func[i](tt) for i in range(len(N_func))]
    else:
        # get times for this epoch
        tt, T_elapsed, current_gen, integ_time = gens_in_interval_as_t(Ne,
            T_elapsed, T, current_gen, integ_time)

        Ns = [Ne*nu*np.ones(len(tt)) for nu in nus]
    
    return Ns, T_elapsed, current_gen, integ_time


###
### randomly mating populations (no selfing)
### we track (T_{i,j}^n for all (i,j) pop pairs, for all moments 1, ..., n
###


def tmrca_vector(order, num_pops):
    """
    [T_{1,1}^n, T_{1,2}^n, ..., T_{p,p}^n, T_{1,1}^{n-1}, 
        T_{1,2}^{n-2}, ..., T_{p,p}^{n-1}, ... T_{p,p}]
    """
    tmrcas = []
    pairs = list(combinations_with_replacement(range(num_pops), 2))
    for i in range(1, order+1)[::-1]: # from largest to smallest order
        for pair in pairs:
            tmrcas.append(f'T{i}_{pair[0]}_{pair[1]}')
    return tmrcas


def integrate_tmrca(dg, Ne, order, pop_ids):
    """
    Block upper diagonal transition matrix for one generation
    """
    # get events
    (present_pops, integration_times, nus, migration_matrices, frozen_pops,
        selfing_rates, events) = integration.get_moments_arguments(dg)
    
    current_pop_ids = present_pops[0]
    
    # initial steady state
    Tmrcas = steady_state_tmrca(dg, Ne, order)

    T_elapsed = 0
    current_gen = 0
    integ_time = 0
    
    # loop through epochs
    for ii, (pops, T, nu, mig_mat, frozen, selfing) in enumerate(zip(
                present_pops, integration_times, nus, migration_matrices,
                frozen_pops, selfing_rates)):
        
        # get total elapsed T, evolve while less than 2*Ne*T generations and
        # get list of Nes for each of these generations, or a single set
        # of Nes if all pop sizes are constant
        Ns, T_elapsed, current_gen, integ_time = get_pop_sizes(Ne, nus[ii],
            T_elapsed, T, current_gen, integ_time)
        
        # rescale m
        mig_mat = 1./2/Ne * np.array(mig_mat)
        
        # evolve over generations in this epoch
        Tmrcas = evolve_t(Tmrcas, order, Ns, mig_mat, frozen)
        
        # apply events (splits, mergers, pulses, marginalizations)
        if ii < len(events):
            Tmrcas, current_pop_ids = apply_events(Tmrcas, events[ii],
                                                   current_pop_ids,
                                                   present_pops[ii+1], order)
    
    Tmrcas, current_pop_ids = marginalize_nonpresent(Tmrcas, pop_ids,
                                                     current_pop_ids, order)
    
    Tmrcas, current_pop_ids = reorder_pops(Tmrcas, order, current_pop_ids,
                                           pop_ids)
    return Tmrcas


def steady_state_tmrca(dg, Ne, order):
    """
    Get the steady state T = -inv(P).dot(ones)
    """
    P = transition(order, [Ne], [[1]], [False])
    return np.linalg.inv(np.eye(order) - P).dot(np.ones(order))


def transition(order, N, m, frozen):
    """
    Moment vector has order (T_{1,1}^n, T_{1,1}^{n-1}, ..., T_{1,2}^n, ...)
    """
    num_pops = len(N)
    for i,freeze in enumerate(frozen):
        if freeze is True:
            N[i] = np.inf
    tmrcas = tmrca_vector(order, num_pops)
    pairs = list(combinations_with_replacement(range(num_pops), 2))
    P = np.zeros((order*len(pairs), order*len(pairs)))
    for ii,(i,j) in enumerate(pairs):
        for n in range(1, order+1)[::-1]: # order of moment for given pair
            this_idx = tmrcas.index(f'T{n}_{i}_{j}')
            if i == j:
                for jj,(k,l) in enumerate(pairs): # loop over all pairs
                    if k == l:
                        m_ik = m[i][k]
                        for kk in range(1, n+1): # loop over all
                            m_ik = m[i][k] # migration rate from 
                            P[this_idx, tmrcas.index(f'T{kk}_{k}_{l}')] += m_ik**2*(1-1/2/N[k])*_choose(n,kk)
                    else:
                        m_ik = m[i][k]
                        m_il = m[i][l]
                        for kk in range(1, n+1):
                            P[this_idx, tmrcas.index(f'T{kk}_{k}_{l}')] += 2*m_ik*m_il*_choose(n,kk)
            else:
                for jj,(k,l) in enumerate(pairs):
                    if k == l:
                        m_ik = m[i][k]
                        m_jk = m[j][k]
                        for kk in range(1, n+1): # loop over all
                            m_ik = m[i][k] # migration rate from 
                            P[this_idx, tmrcas.index(f'T{kk}_{k}_{l}')] += m_ik*m_jk*(1-1/2/N[k])*_choose(n,kk)
                    else:
                        m_ik = m[i][k]
                        m_jl = m[j][l]
                        for kk in range(1, n+1):
                            P[this_idx, tmrcas.index(f'T{kk}_{k}_{l}')] += m_ik*m_jl*_choose(n,kk)
                        m_jk = m[j][k]
                        m_il = m[i][l]
                        for kk in range(1, n+1):
                            P[this_idx, tmrcas.index(f'T{kk}_{k}_{l}')] += m_jk*m_il*_choose(n,kk)
    return P


def evolve_t(Tmrcas, order, Ns, mig_mat, frozen):
    if len(Ns[0]) == 0:
        return Tmrcas
    else:
        # if diagonals of mig_mat are zero and row doesn't sum to one, set diags
        mig_mat = set_prob_nomig(mig_mat, frozen)
        # set frozen pops by setting mig_mat[i] to zero (including mig_mat[i][i])
        last_Ns = [-1] * len(Ns)
        for i in range(len(Ns[0])):
            current_Ns = [Ns[j][i] for j in range(len(Ns))]
            if not np.all(current_Ns == last_Ns): 
                P = transition(order, current_Ns, mig_mat, frozen)
            Tmrcas = P.dot(Tmrcas) + 1
        return Tmrcas


def set_prob_nomig(M, frozen):
    for ii in range(len(M)):
        if frozen[ii] == True:
            for jj in range(len(M)):
                M[ii][jj] = 0
                M[ii][ii] = 1
        if np.sum(M[ii]) != 1.:
            if M[ii][ii] != 0.:
                raise("ValueError", "Migration matrix issues")
            else:
                M[ii][ii] = 1 - np.sum(M[ii])
    return M


def rename_moment(mom):
    m, p1, p2 = mom.split('_')
    if int(p1) > int(p2):
        return '_'.join([m, p2, p1])
    else:
        return mom


def apply_events(Tmrcas, events, current_pop_ids, next_pop_ids, order):
    for event in events:
        if event[0] == 'split':
            Tmrcas, current_pop_ids = split(Tmrcas, event[1],
                                            [event[2], event[3]],
                                            current_pop_ids, order) 
        if event[0] == 'pass':
            Tmrcas, current_pop_ids = pass_pop(Tmrcas, event[1], event[2],
                                               current_pop_ids)
        if event[0] == 'marginalize':
            Tmrcas, current_pop_ids = marginalize(Tmrcas, event[1],
                                                  current_pop_ids, order)
        if event[0] == 'pulse':
            Tmrcas, current_pop_ids = pulse_migrate(Tmrcas, event[1], event[2],
                                               event[3], current_pop_ids, order)
    assert len(current_pop_ids) == len(next_pop_ids)
    assert np.all([pid in next_pop_ids for pid in current_pop_ids])
    Tmrcas, current_pop_ids = reorder_pops(Tmrcas, order, current_pop_ids,
                                           next_pop_ids)
    assert np.all([id1 == id2 for id1, id2 in zip(current_pop_ids, next_pop_ids)])
    return Tmrcas, current_pop_ids

def reorder_pops(Tmrcas, order, old_pop_ids, new_pop_ids):
    new_Tmrcas = np.zeros(len(Tmrcas))
    names = tmrca_vector(order, len(old_pop_ids))
    mapping = {}
    for ii,pop in enumerate(new_pop_ids):
        mapping[ii] = old_pop_ids.index(pop)
    for ii,new_moment in enumerate(names):
        mom = new_moment.split("_")[0]
        pop1 = int(new_moment.split("_")[1])
        pop2 = int(new_moment.split("_")[2])
        xx = [mom,mapping[pop1],mapping[pop2]]
        yy = [str(x) for x in xx]
        old_moment = f'{"_".join(yy)}'
        new_Tmrcas[ii] = Tmrcas[names.index(rename_moment(old_moment))]
    return new_Tmrcas, new_pop_ids


def split(Tmrcas, pop_from, pops_to, pop_ids, order):
    old_names = tmrca_vector(order, len(pop_ids))
    new_names = tmrca_vector(order, len(pop_ids)+1)
    from_idx = pop_ids.index(pop_from)
    to1_idx = from_idx
    to2_idx = len(pop_ids)
    new_Tmrcas = np.zeros(len(new_names))
    for ii,new_mom in enumerate(new_names):
        mom, p1, p2 = new_mom.split('_')
        p1 = int(p1)
        p2 = int(p2)
        if p1 in [to1_idx, to2_idx]:
            p1 = from_idx
        if p2 in [to1_idx, to2_idx]:
            p2 = from_idx
        mom_from = rename_moment('_'.join([mom, str(p1), str(p2)]))
        new_Tmrcas[ii] = Tmrcas[old_names.index(mom_from)]
    pop_ids[to1_idx] = pops_to[0]
    pop_ids.append(pops_to[1])
    return new_Tmrcas, pop_ids


def pass_pop(Tmrcas, pop_from, pop_to, pop_ids):
    pop_ids[pop_ids.index(pop_from)] = pop_to
    return Tmrcas, pop_ids


def marginalize(Tmrcas, pop_out, pop_ids, order):
    old_names = tmrca_vector(order, len(pop_ids))
    out_idx = pop_ids.index(pop_out)
    marg_Tmrcas = []
    for ii,mom in enumerate(old_names):
        p1 = int(mom.split('_')[1])
        p2 = int(mom.split('_')[2])
        if p1 != out_idx and p2 != out_idx:
            marg_Tmrcas.append(Tmrcas[ii])
    marg_Tmrcas = np.array(marg_Tmrcas)
    pop_ids.pop(pop_ids.index(pop_out))
    return marg_Tmrcas, pop_ids


def marginalize_nonpresent(Tmrcas, pop_ids, current_pop_ids, order):
    for pop in current_pop_ids:
        if pop not in pop_ids:
            Tmrcas, current_pop_ids = marginalize(Tmrcas, pop_out,
                                                  current_pop_ids, order)
    return Tmrcas, current_pop_ids


def pulse_migrate(Tmrcas, pop_from, pop_to, f, current_pop_ids, order):
    num_pops = len(current_pop_ids)
    idx_from = current_pop_ids.index(pop_from)
    idx_to = current_pop_ids.index(pop_to)
    Pulse = pulse_matrix(num_pops, order, idx_from, idx_to, f)
    Tmrcas = Pulse.dot(Tmrcas)
    return Tmrcas, current_pop_ids


def pulse_matrix(num_pops, order, idx_from, idx_to, f):
    names = tmrca_vector(order, num_pops)
    P = np.zeros((len(names), len(names)))
    for i, mom in enumerate(names):
        x = mom.split('_')[0]
        p1 = int(mom.split('_')[1])
        p2 = int(mom.split('_')[2])
        if p1 == p2 == idx_to:
            mom_from = rename_moment('_'.join([x, str(idx_from), str(idx_from)]))
            P[i, names.index(mom_from)] = f**2
            mom_from = rename_moment('_'.join([x, str(idx_to), str(idx_from)]))
            P[i, names.index(mom_from)] = 2*f*(1-f)
            P[i, names.index(mom)] = (1-f)**2
        elif (p1 == idx_to and p2 == idx_from) or (p2 == idx_to and p1 == idx_from):
            mom_from = rename_moment('_'.join([x, str(idx_from), str(idx_from)]))
            P[i, names.index(mom_from)] = f
            P[i, names.index(mom)] = (1-f)
        elif p1 == idx_to:
            mom_from = rename_moment('_'.join([x, str(idx_from), str(p2)]))
            P[i, names.index(mom_from)] = f
            P[i, names.index(mom)] = (1-f)
        elif p2 == idx_to:
            mom_from = rename_moment('_'.join([x, str(idx_from), str(p1)]))
            P[i, names.index(mom_from)] = f
            P[i, names.index(mom)] = (1-f)
        else:
            P[i, i] = 1
    return P


###
### selfing populations
### we track TMRCAs for sampled lineages both within the same diploid and from
### different ploids in the same population, and the rate between lineages
### across populations. (Tw_{i,i}^n, Tb_{i,j}^n)
###


def steady_state_tmrca_selfing(dg, Ne, order, selfing):
    root_node = dg.root
    nu_root = dg.G[root_node]['nu']
    



