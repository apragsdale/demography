"""
Class to define demography object, as a networkx DAG
"""

import networkx as nx
import numpy as np

# networkx version needs to be >= 2.0
#if nx.__version__ < 2.0:
#    raise issue

# check imports of other software engines
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
    import msprime
    msprime_installed = 1
except ImportError:
    msprime_installed = 0


class demo_graph():
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
    """
    def __init__(self, G, Ne=None, mutation_rate=None, recombination_rate=None, 
                 samples=None):
        self.G = G
        self.Ne = Ne
        self.mutation_rate = mutation_rate
        self.recombination_rate = recombination_rate
        self.samples = samples

        self.leaves = self.get_leaves()
        self.root = self.get_root()
        self.successors = self.get_successors()
        self.predecessors = self.get_predecessors()

    def get_root(self):
        # returns the root of the demography (the initial population)
        return next(nx.topological_sort(self.G))

    def get_leaves(self):
        # returns the leaf nodes - populations without successors
        return [x for x in G if len(G._adj[x]) == 0]

    def get_successors(self):
        # returns a dict of children populations for each node, if they exist
        return {node : list(adjs.keys()) for node, adjs in G._adj.items() if adjs.keys()}

    def get_predecessors(self):
        # returns a dict of parental populations for each node, if they exist
        return {x : list(G.predecessors(x)) for x in G if list(G.predecessors(x))}


"""
Functions to integrate moments from the Demography.
Note that moments can handle up to five populations at any given time, so we have a 
check that ensures that there are no more than five populations at all times along
the graph.
moments also requires samples to be defined, and 
"""
