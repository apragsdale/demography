"""
Class to define demography object, as a networkx DAG
"""

import networkx as nx
import numpy as np
from . import integration
from . import msprime_functions
import tskit

## check nx version (must be >= 2.1)
def check_nx_version():
    assert (float(nx.__version__) >= 2.2), "networkx must be version 2.2 or \
                                            higher to use Demography"

try:
    import msprime
    msprime_installed = 1
except ImportError:
    msprime_installed = 0

## exception raised if the input graph has an issue
class InvalidGraph(Exception):
    pass

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
        check_valid_demography(self)

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
    Functions to simulate or evolve over the demography to get expecte
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

    def LD(self, rho=None, theta=None, pop_ids=None):
        """
        
        """
        # compute expected LD curves and heterozygosity statistics for
        # populations with given samples. uses moments.LD
        # rho = 4*Ne*r, where r is the per base recombination rate
        # rho is either None, a signle value, or a list of rhos
        y = integration.evolve_ld(self, rho=rho, theta=theta, pop_ids=pop_ids)
        return y

    def SFS(self, engine='moments', pop_ids=None, sample_sizes=None, 
            theta=1., s=None, h=None, Ne=None, u=None):
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

        # check that there are at most 3/5 populations at any time...
        #if pop_ids is None:
            #set pops as leaves
        #if sample_sizes is None:
            #error

        # set population scaled selection coefficient, if given
        if s is not None:
            gamma = 2 * dg.Ne * s
            if h is None:
                h = 0.5
        else:
            gamma = None

        if engine == 'moments':
            fs = integration.evolve_sfs_moments(self, theta=theta, 
                                                pop_ids=pop_ids,
                                                sample_sizes=sample_sizes,
                                                gamma=gamma, h=h)

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

    def msprime_samples(self, pop_ids=None, sample_sizes=None):
        """
        Inputs:
            pop_ids is the list of populations to get samples from
            sample sizes using a list of number of haploid samples to take from
                the populations given in pop_ids, with same order
        Output:
            list of samples for msprime
        """
        assert np.all([pop in self.leaves for pop in pop_ids]), "invalid sampling population"
        samples = msprime_functions.get_samples(self, pop_ids, sample_sizes)
        return samples

    def simulate_msprime(self, model='hudson', Ne=None,
                         pop_ids=None, sample_sizes=None,
                         sequence_length=None, recombination_rate=None,
                         recombination_map=None, mutation_rate=None,
                         replicates=None):
        """
        Qs: how to handle recombination map (msprime format vs hapmap, both?)
            
        """
        # check if recombination rate or genetic map is passed
        if recombination_rate == None:
            recombination_rate = self.recombination_rate
        if recombination_map is not None and recombination_rate is not None:
            print("recombination rate given, but using given map")
            recombination_rate = None
        if sequence_length is None:
            print("no sequence length set, using genetic map if given")
            if recombination_map is not None:
                print("one is given - but need to fix this")
                # get end of recombination map
                sequence_length = 1.
            else:
                print("setting sequence length to 1")
                sequence_length = 1.
        
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
                              num_replicates=replicates)

        return ts

"""
Set of functions to check that the demography is specified properly

Note: could split these off into functions outside of class, for readibility
"""
def check_valid_demography(dg):
    # check that there is only a single root
    num_roots = sum([node not in dg.predecessors for node in dg.G.nodes] )
    if num_roots != 1:
        raise InvalidGraph('demography requires a single root')
    # check that there are no loops
    if len(list(nx.simple_cycles(dg.G))) > 0:
        raise InvalidGraph('demography cannot have any loops')
    # check that mergers are valid (proportions sum to one)
    any_mergers = False
    for pop in dg.predecessors:
        if len(dg.predecessors[pop]) != 1:
            if len(dg.predecessors[pop]) != 2:
                raise InvalidGraph('mergers can only be between two pops')
            else:
                total_weight = 0
                for parent in dg.predecessors[pop]:
                    if 'weight' not in dg.G.get_edge_data(parent, pop):
                        raise InvalidGraph('weights must be assigned for mergers')
                    else:
                        total_weight += dg.G.get_edge_data(parent, pop)['weight']
                if total_weight != 1.:
                    raise InvalidGraph('merger weights must sum to 1')
    # check that all times align
    if all_merger_times_align(dg) == False:
        raise InvalidGraph('splits/mergers do not align')

def all_merger_times_align(dg):
    # returns True if split and merger times align throughout the graph
    # note that this doesn't check if all leaves end at the same time
    all_align = True
    for child in dg.predecessors:
        if len(dg.predecessors[child]) == 2:
            # is a merger - travel up each side through all paths to root,
            # make sure times at common predecessors are equal
            all_paths = nx.all_simple_paths(dg.G, dg.root, child)
            # get corresponding times along each path
            nodes = []
            times = []
            for simple_path in all_paths:
                nodes.append(['root'])
                times.append([0])
                for this_node in simple_path[1:]:
                    nodes[-1].append(this_node)
                    times[-1].append(dg.G.nodes[this_node]['T'] + times[-1][-1])
            # silly loop, but it works...
            for i in range(len(nodes)-1):
                for n,t1 in zip(nodes[i], times[i]):
                    for j in range(i,len(nodes)):
                        if n in nodes[j]:
                            t2 = times[j][nodes[j].index(n)]
                            if t1 != t2:
                                all_align = False
                                return all_align
                    
        else:
            continue
    return all_align
