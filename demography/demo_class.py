"""
Class to define demography object, as a networkx DAG
"""

import networkx as nx
import numpy as np
from . import integration
#from . import msprime_functions
import tskit

## check nx version (must be >= 2.1)
def check_nx_version():
    assert (float(nx.__version__) >= 2.0), "networkx must be version 2.0 or \
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
    Class for Demography objects, which are represented as networkx directed acyclic
    graphs, whose nodes represent populations and edges represent splits and mergers
    between these populations.

    The Demography object can store attributes, such as reference Ne, mutation rate,
    recombination rate.

    Samples can be specified for any leaf population, and we assume that the sampling
    occurs at the end of that population (so either it's a contemporary sample, or an
    ancient sample at the end of that branch).

    Nodes, representing populations, can have a number of attributes. The required
    attributes are:
        Their sizes nu or size functions (nu0 and nuF, or nuF and growth_rate) relative
        to Ne
        Times T (how long the populations exist)
    Optional attributes are:
        Migration rates from the node to other populations (if they exist at the time)
        Pulse migration events from the node to other present populations
        Selfing rates (default is 0)
        Frozen population (default is False)

    Edges define population splits or continuations (if we want to change an attribute)
    and then also define mergers, if two populations direct to the same child population.
    In the case of mergers, we also need to specify the weight of the two edges, and
    those weights need to sume to 1 (so we know the contributions from the admixture
    event).

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

        # check that this is a valid graph
        # raises InvalidGraph exception if there are multiple roots, or if the times
        # of splitting/merging populations do not align, or if there are loops
        self.check_valid_demography()

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
    Set of functions to check that the demography is specified properly
    """
    def check_valid_demography(self):
        # check that there is only a single root
        num_roots = sum([node not in self.predecessors for node in self.G.nodes] )
        if num_roots != 1:
            raise InvalidGraph('demography requires a single root')
        # check that there are no loops
        if len(list(nx.simple_cycles(self.G))) > 0:
            raise InvalidGraph('demography cannot have any loops')
        # check that mergers are valid (proportions sum to one)
        any_mergers = False
        for pop in self.predecessors:
            if len(self.predecessors[pop]) != 1:
                if len(self.predecessors[pop]) != 2:
                    raise InvalidGraph('mergers can only be between two pops')
                else:
                    total_weight = 0
                    for parent in self.predecessors[pop]:
                        if 'weight' not in self.G.get_edge_data(parent, pop):
                            raise InvalidGraph('weights must be assigned for mergers')
                        else:
                            total_weight += self.G.get_edge_data(parent, pop)['weight']
                    if total_weight != 1.:
                        raise InvalidGraph('merger weights must sum to 1')
        # check that all times align
        if self.all_merger_times_align() == False:
            raise InvalidGraph('splits/mergers do not align')

    def all_merger_times_align(self):
        # returns True if split and merger times align throughout the graph
        # note that this doesn't check if all leaves end at the same time (contemp) 
        all_align = True
        for child in self.predecessors:
            if len(self.predecessors[child]) == 2:
                # is a merger - travel up each side through all paths to root, make sure
                # times at common predecessors are equal
                all_paths = nx.all_simple_paths(self.G, self.root, child)
                # get corresponding times along each path
                nodes = []
                times = []
                for simple_path in all_paths:
                    nodes.append(['root'])
                    times.append([0])
                    for this_node in simple_path[1:]:
                        nodes[-1].append(this_node)
                        times[-1].append(self.G.nodes[this_node]['T'] + times[-1][-1])
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

    """
    Functions to simulate or evolve over the demography
    """
    def get_theta(self):
        if self.Ne is not None and self.mutation_rate is not None:
            if self.sequence_length is None:
                theta = 4*self.Ne*self.mutation_rate
            else:
                theta = 4*self.Ne*self.mutation_rate*self.sequence_length
        else:
            theta = 1.0
        return theta

    def LD(self, rho=None, theta=None, pop_ids=None):
        # compute expected LD curves and heterozygosity statistics for populations with
        # given samples. uses moments.LD
        # rho = 4*Ne*r, where r is the per base recombination rate
        # rho is either None, a signle value, or a list of rhos
        
        y = integration.evolve_ld(self, rho=rho, theta=theta, pop_ids=pop_ids)
        return y

"""
    def SFS(self, engine='moments', s=None, h=None):
        # compute expected frequency spectrum for the given samples
        # engine could be either 'moments' or 'dadi'
        fs = integration.evolve(self, engine=engine, s=s, h=h)
        return fs

    def msprime_inputs(self):
        pop_config, samp, mig_mat, demo_events = msprime_from_graph(self)
        return pop_config, samp, mig_mat, demo_events
    
    def simulate_msprime(self, model='hudson'):
        # want to allow replicates? genetic map? what else?
        # model could be hudson or dtwf
        pop_config, samp, mig_mat, demo_events = msprime_from_graph(self)
        ts = msprime_from_graph.simulate(population_configurations=pop_config,
            sequence_length=self.sequence_length)
"""
