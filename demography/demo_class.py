"""
Class to define demography object, as a networkx DAG
"""

import networkx as nx
import numpy as np
from . import integration
from . import msprime_functions
from . import util
from . import coalescence_rates
from . import tmrcas
# from . import twolocus_tmrcas
import tskit


## check nx version (must be >= 2.1)
def check_nx_version():
    assert (nx.__version__ >= "2.2"), "networkx must be version 2.2 or \
                                       higher to use Demography"

try:
    import msprime
    msprime_installed = 1
except ImportError:
    msprime_installed = 0


class DemoGraph():
    """
    Class for Demography objects, which are represented as networkx directed
    acyclic graphs, whose nodes represent populations and edges represent
    splits and mergers between these populations.

    The Demography object can store attributes, such as reference Ne, mutation
    rate, recombination rate.

    Samples can be specified for any leaf population, and we assume that the
    sampling occurs at the end of that population (so either it's a
    contemporary sample, or an ancient sample at the end of that branch).

    Nodes, representing populations, can have a number of attributes. The
    required attributes are:
        Their sizes nu or size functions (nu0 and nuF, or nuF and growth_rate)
        relative to Ne
        Times T (how long the populations exist)
    Optional attributes are:
        Migration rates from the node to other populations (if they exist at 
        the time)
        Pulse migration events from the node to other present populations
        Selfing rates (default is 0)
        Frozen population (default is False)

    Edges define population splits or continuations (if we want to change an
    attribute) and then also define mergers, if two populations direct to the
    same child population. In the case of mergers, we also need to specify the
    weight of the two edges, and those weights need to sume to 1 (so we know
    the contributions from the admixture event).

    Ne : reference effective population size
    samples : 
    
    """
    def __init__(self, G, Ne=None, mutation_rate=None, recombination_rate=None, 
                 samples=None, sequence_length=None):
        check_nx_version()
        
        self.G = G

        self.leaves = self.get_leaves()
        self.root = self.get_root()
        self.successors = self.get_successors()
        self.predecessors = self.get_predecessors()

        # check that this is a valid graph, raises InvalidGraph exception if 
        # there are multiple roots, or if the times of splitting/merging
        # populations do not align, or if there are loops
        util.check_valid_demography(self)

        self.Ne = Ne
        self.mutation_rate = mutation_rate
        self.recombination_rate = recombination_rate
        self.samples = samples
        self.sequence_length = sequence_length
        self.theta = self.get_theta()

    """
    Functions to get relationships of populations within the graph
    """
    def get_root(self):
        # returns the root of the demography (the initial population)
        return next(nx.topological_sort(self.G))

    def get_leaves(self):
        # returns the leaf nodes - populations without successors
        return [x for x in self.G if len(self.G._adj[x]) == 0]

    def get_successors(self):
        # returns a dict of children populations for each node, if they exist
        return {node : list(adjs.keys()) for node, adjs in self.G._adj.items() if adjs.keys()}

    def get_predecessors(self):
        # returns a dict of parental populations for each node, if they exist
        return {x : list(self.G.predecessors(x)) for x in self.G if list(self.G.predecessors(x))}


    """
    Functions to simulate or evolve over the demography to get expected
    site frequency spectrum or linkage disequilibrium statistics.
    """

    def get_theta(self):
        """
        
        """
        if self.Ne is not None and self.mutation_rate is not None:
            if self.sequence_length is None:
                theta = 4*self.Ne*self.mutation_rate
            else:
                theta = 4*self.Ne*self.mutation_rate*self.sequence_length
        else:
            theta = 1.0
        return theta

    def LD(self, pop_ids, rho=None, r=None, theta=None, u=None, augment=True,
           time_series=False, time_series_resolution=10):
        """
        
        """
        # compute expected LD curves and heterozygosity statistics for
        # populations with given samples. uses moments.LD
        # rho = 4*Ne*r, where r is the per base recombination rate
        # rho is either None, a signle value, or a list of rhos
        assert not ((rho is not None) and (r is not None)), "Can only pass rho or r"
        if rho is None:
            if r is not None:
                if self.Ne is None:
                    raise ValueError("If r is given, the DemoGraph object must have Ne defined")
                else:
                    rho = 4*self.Ne*r
        
        assert not ((theta is not None) and (u is not None)), "Can only pass theta or u"
        if theta is None:
            if u is not None:
                if self.Ne is None:
                    raise ValueError("If u is given, the DemoGraph object must have Ne defined")
                else:
                    theta = 4*self.Ne*u
        
        y = integration.evolve_ld(self, rho=rho, theta=theta, pop_ids=pop_ids,
                                  augment=augment,
                                  time_series=time_series,
                                  time_series_resolution=time_series_resolution)
        return y

    def SFS(self, pop_ids, sample_sizes, engine='moments',
            theta=None, s=0, h=0.5, Ne=None, u=None, pts=None,
            reversible=False, augment=True):
        """
        Computes the expected frequency spectrum for the given populations and
            sample sizes.
        
        Inputs:
            engine (default moments, could also be dadi)
            pop_ids (list of leaf populations)
            sample_sizes (list of sample sizes for those leaf populations)
            theta (optional, defaults to 1 if u and Ne are not set)
            s (selection coefficient)
            h (dominance coefficient)
            
        """
        if theta is None:
            if u is not None:
                if Ne is None and self.Ne is None:
                    print("warning: u is given, but no Ne - setting theta to 1")
                    theta = 1
                if Ne is not None:
                    theta = 4*Ne*u
                if self.Ne is not None:
                    theta = 4*self.Ne*u
            else:
                theta = 1.

        # set population scaled selection coefficient, if given
        if s != 0:
            assert Ne is not None or self.Ne is not None, "Ne or dg.Ne must be set"
            gamma = 2 * self.Ne * s
        else:
            gamma = 0.0

        if engine == 'moments':
            if reversible is True:
                assert h == 0.5, "reversible model requires h=1/2"
            fs = integration.evolve_sfs_moments(self, theta=theta, 
                                                pop_ids=pop_ids,
                                                sample_sizes=sample_sizes,
                                                gamma=gamma, h=h,
                                                reversible=reversible,
                                                augment=augment)
        elif engine == 'dadi':
            assert reversible is False, "Can't simulate finite genome with dadi"
            util.max_two_successors(self)
            assert pts is not None, "pts must be given to integrate with dadi"
            fs = integration.evolve_sfs_dadi(self, pts, theta=theta, 
                                             pop_ids=pop_ids,
                                             sample_sizes=sample_sizes,
                                             gamma=gamma, h=h,
                                                augment=augment)
        return fs


    """
    Functions to handle msprime simulation
    """

    def msprime_inputs(self, Ne=None):
        """
        Input:
            Ne (optional)
        Outputs:
            population_configurations
            migration_matrix
            demographic_events

        Objects needed to run a simulation in msprime
        If Ne is not given, the function checks whether self.Ne is given, and
            if not, Ne is set to 1.
        """
        # get the population_configurations, migration_matrix, and demographic
        # events to run the msprime simulation
        (pop_config, mig_mat,
            demo_events) = msprime_functions.msprime_from_graph(self, Ne=Ne)
        return pop_config, mig_mat, demo_events

    def msprime_samples(self, pop_ids, sample_sizes):
        """
        Inputs:
            pop_ids is the list of populations to get samples from
            sample sizes using a list of number of haploid samples to take from
                the populations given in pop_ids, with same order
        Output:
            list of samples for msprime
        """
        assert np.all([pop in self.leaves for pop in pop_ids]), "invalid sampling population"
        assert len(pop_ids) == len(sample_sizes), "pop_ids and sample_sizes must have same length"
        samples = msprime_functions.get_samples(self, pop_ids, sample_sizes)
        return samples

    def simulate_msprime(self, model='hudson', Ne=None,
                         pop_ids=None, sample_sizes=None,
                         sequence_length=None, recombination_rate=None,
                         recombination_map=None, mutation_rate=None,
                         replicates=None, seed=None):
        """
        recombination map: pass file path+name, msprime will parse and set the 
            length
            
        """
        # check if recombination rate or genetic map is passed
        if recombination_rate == None:
            recombination_rate = self.recombination_rate
        if recombination_map is not None and recombination_rate is not None:
            print("recombination rate given, but using given map")
            recombination_rate = None
        
        if recombination_map is not None:
            recombination_map = msprime.RecombinationMap.read_hapmap(recombination_map)
            if sequence_length is not None:
                print("recombination map given, sequence length is map length")
            sequence_length = None
        
        pop_config, mig_mat, demo_events = self.msprime_inputs(Ne=Ne)
        samples = self.msprime_samples(pop_ids=pop_ids,
                                  sample_sizes=sample_sizes)

        ts = msprime.simulate(population_configurations=pop_config,
                              migration_matrix=mig_mat,
                              demographic_events=demo_events,
                              samples=samples,
                              model=model,
                              length=sequence_length,
                              recombination_rate=recombination_rate,
                              mutation_rate=mutation_rate,
                              recombination_map=recombination_map,
                              num_replicates=replicates,
                              random_seed=seed)

        return ts

    """
    Compute coalescence rates for specified populations. Uses msprime
    demographic events, migration matrix, and population configs to get
    coalescent rates between pairs of lineages within and between set of
    populations.
    """
    
    def coal_rates(self, pop_ids=None, Ne=None, gens=None):
        """
        gens defaults to 2*N
        """
        if Ne is None:
            if self.Ne is not None:
                Ne = self.Ne
            else:
                raise InputError("Please pass Ne")
        (pop_config, mig_mat, demo_events) = self.msprime_inputs(Ne=Ne)
        
        pop_indexes = {}
        for ii,pop in enumerate(self.G.nodes):
            pop_indexes[pop] = ii

        if gens is None:
            gens = 2*Ne
        rates = coalescence_rates.all_rates(pop_ids, pop_indexes, pop_config,
                                            mig_mat, demo_events, gens)
        return rates

    """
    Compute Tmrca's between populations, using a discrete recursion.
    """
    
    def tmrca(self, pop_ids, Ne=None, order=2, selfing=None, two_locus=False, r=None):
        """
        This function computes all (non-central?) moments of Tmrcas within
        and between populations in the demography.
         
        pop_ids: which leaf populations to compute 
        Ne: effective population size. Must be specified as argument or in
            the demography
        order: highest order moment to compute.
        selfing: if selfing is None, uses selfing rates defined in the
                 demography, if they are given. Otherwise, defaults to
                 random mating model. If some subset of the populations have
                 selfing rates set, we use those selfing rates for those
                 branches, and then uses the selfing rate defined here for the
                 rest of the branches. In this case, if selfing is None, it
                 defaults to 0, so that selfing is disallowed if not specified
                 in some populations.
        two_locus: if False (default), computes moments of the single locus Tmrca
                   distribution. If True, order must be set to tuple of length two
                   (order_x, order_y), and 
        """
        if Ne is None:
            Ne = self.Ne
            assert Ne is not None, "Ne must be specified to compute Tmrcas"

        if selfing is not None:
            assert 0 <= selfing <= 1, "selfing rate must be between 0 and 1"

        assert np.all([pid in self.leaves for pid in pop_ids]), "pop_ids must be leaves of the demography"

        # To do:
        # returns either a structured array or a dictionary with moments
        # also need to decide on the basis, central vs non-central moments, etc
        if two_locus is False:
            ETn = tmrcas.compute_tmrcas(self, pop_ids, Ne, order, selfing)
        else:
            if r is None:
                print("recombingation rate `r` not specified - setting r = 0.")
                r = 0
            if selfing is not None:
                print("only the random-mating model is implemented for two loci")
            ETn = twolocus_tmrcas.compute_twolocus_tmrcas(self, pop_ids, Ne, order[0],
                                                          order[1], r)
        
        return ETn
