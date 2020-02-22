import networkx as nx


## exception raised if the input graph has an issue
class InvalidGraph(Exception):
    pass


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
    
def max_two_successors(dg):
    # check that at most two successors
    # want to relax this
    for pop in dg.successors:
        if len(dg.successors[pop]) > 2:
            raise InvalidGraph('can only split into maximum two populations')

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

"""

"""

def get_accumulated_times(dg):
    leaf_times = {}
    for leaf in dg.leaves:
        t = dg.G.nodes[leaf]['T']
        parent = get_one_parent(dg, leaf)
        while parent is not dg.root:
            t += dg.G.nodes[parent]['T']
            parent = get_one_parent(dg, parent)
        leaf_times[leaf] = t
    return leaf_times


def get_one_parent(dg, child):
    if hasattr(dg.predecessors[child], "__len__"): # from a merger, just pick one parent
        parent = dg.predecessors[child][0]
    else:
        parent = dg.predecessors[child]
    return parent

