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
        T = steady_state_tmrca_selfing(dg, Ne, order, selfing)
        T = integrate_tmrca_selfing(dg, Ne, order, selfing)
    else:
        T = steady_state_tmrca(dg, Ne, order)
        T = integrate_tmrca(dg, Ne, order)
    
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


def gens_in_interval_as_t(Ne, T_elapsed, T, current_gen):
    tt = [0]
    TF = T_elapsed + T
    while (current_gen+1)/2/Ne < TF:
        T_elapsed += 1./2/Ne
        current_gen += 1
        tt.append(T_elapsed)
    return np.array(tt), T_elapsed, current_gen

def get_pop_sizes(Ne, nus, T_elapsed, T, current_gen):
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
                                 Ne * nu0 * np.exp(growth_rate*(t-T_elapsed)) )
            else:
                N_func.append( lambda t, pop_nu=pop_nu: Ne * pop_nu )

        # get times for this epoch
        tt, T_elapsed, current_gen = gens_in_interval_as_t(Ne, T_elapsed, T,
                                                           current_gen)

        Ns = [N_func(tt)]
    else:
        # get times for this epoch
        tt, T_elapsed, current_gen = gens_in_interval_as_t(Ne, T_elapsed, T,
                                                           current_gen)

        Ns = [Ne*nu for nu in nus]
    
    return Ns, T_elapsed, current_gen

###
### randomly mating populations (no selfing)
### we track (T_{i,j}^n for all (i,j) pop pairs, for all moments 1, ..., n
###


def tmrca_vector(order, num_pops):
    tmrcas = []
    pairs = list(combinations_with_replacement(range(num_pops), 2))
    for pair in pairs:
        for i in range(1, order+1)[::-1]:
            tmrcas.append(f'T{i}_{pair[0]}_{pair[1]}')
    return tmrcas


def integrate_tmrca(dg, Ne, order):
    # get events
    (present_pops, integration_times, nus, migration_matrices, frozen_pops,
        selfing_rates, events) = get_moments_arguments(dg)

    # initial steady state
    T = steady_state_tmrca(dg, Ne, order)

    T_elapsed = 0
    current_gen = 0
    
    # loop through epochs
    for ii, (pops, T, nu, mig_mat, frozen, selfing) in enumerate(zip(
                present_pops, integration_times, nus, migration_matrices,
                frozen_pops, selfing_rates)):
        
        # get total elapsed T, evolve while less than 2*Ne*T generations
        Ns, current_gen = get_pop_sizes(Ne, nus, T_elapsed, T, current_gen)
        T_elapsed += T
        
        # get list of Nes for each of these generations, or a single set
        # of Nes if all pop sizes are constant
        
        # evolve over generations in this epoch

def steady_state_tmrca(dg, Ne, order):
    """
    Get the steady state T = -inv(P).dot(ones)
    """
    P = transition(order, [Ne], [[1]])
    return np.linalg.inv(np.eye(2) - P).dot(np.ones(2))


def transition(order, N, m):
    """
    Moment vector has order (T_{1,1}^n, T_{1,1}^{n-1}, ..., T_{1,2}^n, ...)
    """
    num_pops = len(N)
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

###
### selfing populations
### we track TMRCAs for sampled lineages both within the same diploid and from
### different ploids in the same population, and the rate between lineages
### across populations. (Tw_{i,i}^n, Tb_{i,j}^n)
###


def steady_state_tmrca_selfing(dg, Ne, order, selfing):
    root_node = dg.root
    nu_root = dg.G[root_node]['nu']
    



