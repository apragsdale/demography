"""
Module to handle plotting demographies. These can be used to visually debug a
demography object that has been defined, or to create quality visualizations of
the demography, either as a stand-alone plot or as a subplot axes object. 

Requires matplotlib version >= ??
"""

import numpy as np
import matplotlib.pylab as plt
import copy
import networkx as nx
import operator

# Font styles for plots
pop_label_style = dict(size=10, color='darkblue')
frac_style = dict(size=8, color='black')
text_style = dict(ha='center', va='center', fontsize=10)
text_box_style = dict(facecolor='none', edgecolor='darkblue', 
                      boxstyle='round,pad=0.25', linewidth=1)


def fill_intervals(dg, intervals, pop):
    for succ in dg.successors[pop]:
        if succ not in intervals:
            T_last = intervals[pop][1]
            intervals[succ] = (T_last, T_last+dg.G.nodes[succ]['T'])
            
            if succ not in dg.leaves:
                fill_intervals(dg, intervals, succ)


def rescale_intervals(intervals):
    max_t = max([i[1] for i in intervals.values()])
    for pop in intervals.keys():
        intervals[pop] = (intervals[pop][0]/max_t, intervals[pop][1]/max_t)
    return intervals


def get_relative_intervals(dg):
    # get relative time for each branch from root to end of last leaf
    intervals = {}
    pop = dg.root
    intervals[pop] = (-dg.G.nodes[dg.root]['T'], 0)
    fill_intervals(dg, intervals, pop)
    return rescale_intervals(intervals)


def get_leaf_order(dg):
    # if pop_order is not given, we want to cluster leaves with common parents
    # and that are in higher order clades, so that arrows don't cross too much
    # XXX To Do!! will take some thought...
    leaf_parents = [dg.predecessors[leaf] for leaf in dg.leaves]
    order = []
    focal_leaf = dg.leaves[0]


def draw_node(ax, x, y, node, dg, pops_drawn, pop_locations):
    # given x and y coordinates, draws the text box for the given node and
    # updates the bookkeeping
    ax.text(x, y, node, **text_style, bbox=text_box_style)
    pops_drawn[node] = True
    pop_locations[node] = (x, y)


def get_merger_order(node, dg, pop_locations):
    preds = dg.predecessors[node]
    distances = {preds[0]:[], preds[1]:[]}
    locs = []
    for leaf in dg.leaves:
        if pop_locations[leaf][0] < pop_locations[node][0]:
            locs.append(-1)
        elif pop_locations[leaf][0] > pop_locations[node][0]:
            locs.append(1)
        else:
            locs.append(0)
        for pred in preds:
            distances[pred].append(len(nx.shortest_path(dg.G.to_undirected(), pred, leaf)))
    vals = {}
    for pred in preds:
        vals[pred] = np.sum([l*d for l,d in zip(locs, distances[pred])])
    # if more positive, goes on left. if more negative, goes on right
    return sorted(vals, key=vals.get)[::-1]
    
def draw_successors(ax, node, dg, intervals, pops_drawn, pop_locations, padding):
    # if a child is not drawn, draw it
    # note that the leaves should already be drawn, because we don't want to
    # call this on the leaves
    if node in dg.leaves:
        return
    
    # make sure children are drawn. If not, draw them
    for child in dg.successors[node]:
        # don't draw if already drawn
        # otherwise we draw it's children if we need to, and then draw it
        if pops_drawn[child] is True:
            continue
        
        # draw the child's children
        draw_successors(ax, child, dg, intervals, pops_drawn, pop_locations, padding)
        
        # now we can draw this child node
        # if it's passed on (only one successor, which has only one parent),
        # we placed it directly above. if it's a split, halfway between.
        # if it's a merger, we have to guess at the order, and say if it's on
        # the left or right
        if len(dg.successors[child]) == 2:
            # split, take average
            x = np.mean([pop_locations[succ][0] for succ in dg.successors[child]])
        if len(dg.successors[child]) == 1:
            # check if passed on or merger
            if len(dg.predecessors[dg.successors[child][0]]) == 1:
                # pass on
                x = pop_locations[dg.successors[child][0]][0]
            else:
                # merger
                order = get_merger_order(dg.successors[child][0], dg, pop_locations)
                if order[0] == child: # fork to left
                    x = pop_locations[dg.successors[child][0]][0]-padding
                else: # fork to right
                    x = pop_locations[dg.successors[child][0]][0]+padding
        
        y = 1 - np.mean(intervals[child])
        draw_node(ax, x, y, child, dg, pops_drawn, pop_locations)


def draw_edge(ax, edge, dg, pop_locations, offset=0.005, buffer=0.03):
    # if it's directly above-below, we center the arrow. if it goes left or
    # right, we offset it slightly, based on offset dist
    # depending on plot size, we might need to adjust the buffer for font size,
    # since font sizes don't scale if you change the figure size
    node_from, node_to = edge
    if pop_locations[node_from][0] < pop_locations[node_to][0]: # diag right
        x_from = pop_locations[node_from][0] + offset
        x_to = pop_locations[node_to][0] - offset
    elif pop_locations[node_from][0] > pop_locations[node_to][0]: # diag left
        x_from = pop_locations[node_from][0] - offset
        x_to = pop_locations[node_to][0] + offset
    else: # straight down
        x_from = x_to = pop_locations[node_from][0]
    
    y_from = pop_locations[node_from][1] - buffer
    y_to = pop_locations[node_to][1] + buffer
    
    # get annotation if it's a merger
    if len(dg.predecessors[node_to]) == 2:
        weight = dg.G.get_edge_data(edge[0], edge[1])['weight']
        annot = True
    else:
        annot = False

    ax.annotate(
        '', xy=(x_to, y_to), xycoords='data',
        xytext=(x_from, y_from), textcoords='data',
        arrowprops={'arrowstyle': '->'})
    if annot == True:
        ax.annotate(
            f'{weight}', xy=(np.mean([x_from, x_to]), np.mean([y_from, y_to])), xycoords='data',
            xytext=(4, 0), textcoords='offset points', fontsize=8)


def draw_pulses(ax, dg, pop_locations, offset):
    for pop_from in dg.G.nodes:
        if 'pulse' in dg.G.nodes[pop_from]:
            for pulse_info in dg.G.nodes[pop_from]['pulse']:
                pop_to = pulse_info[0]
                weight = pulse_info[2]
                x_from, y_from = pop_locations[pop_from]
                x_to, y_to = pop_locations[pop_to]
                if x_from < x_to:
                    x_from += offset
                    x_to -= offset
                else:
                    x_from -= offset
                    x_to += offset
                ax.annotate(
                    '', xy=(x_to, y_to), xycoords='data',
                    xytext=(x_from, y_from), textcoords='data',
                    arrowprops={'arrowstyle': '->', 'ls': 'dashed'})
                ax.annotate(
                    f'{weight}', xy=(np.mean([x_from, x_to]), 
                                     np.mean([y_from, y_to])),
                    xycoords='data',
                    xytext=(0, 3), textcoords='offset points', fontsize=8)


def plot_graph(dg, fignum=1, leaf_order=None, leaf_locs=None, ax=None,
               show_pulses=True, show=False, offset=0.02, padding=0.05):
    """
    This function is mostly useful for debugging and visualizing the topology of
    the demography, with all populations labeled. For a nicer visualizing, use
    plot_demography below.

    Ignores population sizes and continuous migration rates, just plots the
    relationships between populations in the DemoGraph. Arrows indicate splits,
    mergers, and pulse migration events, with fraction of contributions shown
    if needed.

    Time is more-or-less ordered from bottom to top as present to past, but
    because populations can persist for different lengths of time, this plot
    isn't meant to be to scale or necessarily indicate the ordering of pulse
    migration events or splits along different lineages.

    If a node has a single parent, it is placed directly above. If a node has
    two children, it is placed directly between.

    If leaf_order is not given, we try to cluster, but this isn't guaranteed
    to give a good result.
    """
    assert leaf_order is not None, "specify leaf_order=[...]"
    assert [l in dg.leaves for l in leaf_order], "if leaf_order given, must include all leaves"
    assert not ax == None or show == False, "cannot show plot if passing axis"
    
    pops_drawn = {}
    pop_locations = {}
    for node in dg.G.nodes:
        pops_drawn[node] = False

    intervals = get_relative_intervals(dg)

    if ax == None:
        fig = plt.figure(fignum)
        fig.clf()
        ax = plt.subplot(1,1,1)

    ax.set_xlim([0,1])
    ax.set_ylim([0,1+np.mean(intervals[dg.root])+.1])
    
    # draw the root
    y = 1 - np.mean(intervals[dg.root])
    draw_node(ax, 0.5, y, dg.root, dg, pops_drawn, pop_locations)
            
    # reorder the leaves if not given
    if leaf_order is None:
        leaf_order = copy.copy(dg.leaves)
    
    # draw the leaves
    if leaf_locs is None:
        leaf_locs = [(i+1.)/(len(leaf_order)+1.) for i in range(len(leaf_order))]
 
    for i,l in enumerate(leaf_order):
        x = leaf_locs[i]
        if np.isclose(intervals[l][1], 1.0):
            y = 1 - np.max([np.mean(intervals[lll]) for lll in leaf_order])
        else:
            y = 1 - np.mean(intervals[l])
        draw_node(ax, x, y, l, dg, pops_drawn, pop_locations)
    
    # fill in to root
    draw_successors(ax, dg.root, dg, intervals, pops_drawn, pop_locations, padding)
    
    # annotate with arrows
    for edge in dg.G.edges:
        draw_edge(ax, edge, dg, pop_locations)
    
    if show_pulses:
        draw_pulses(ax, dg, pop_locations, offset)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    if show == False:
        return ax
    else:
        fig.tight_layout()
        plt.show()



def plot_demography(dg, fignum=1, leaf_order=None, ax=None,
                    show=False, padding=0.1, stacked=None):

    """
    The plot_graph function is mostly for viewing overarching topology. We'll
    use a lot of the same functions for this function, which draws shapes to
    represent the duration and size of each population, with arrows for pulse
    and continuous migration events (if option is turned on), with sizes drawn
    on a linear or log scale (sometimes we'll choose log scale if there are
    orders of magnitude differences in population sizes along the demography).

    stacked is a list of pairs of populations that should be stacked. If A->B
    and A->C, but we want it to look like 
            |
            A
            |\
            B C
    we would set stacked=[('A','B')] or stacked=[('B','A')]
    """
    assert leaf_order is not None, "specify leaf_order=[...]"
    assert [l in dg.leaves for l in leaf_order], "if leaf_order given, must include all leaves"
    assert not ax == None or show == False, "cannot show plot if passing axis"
    
    # We'll draw populations in reverse order of their extinction
    pops_drawn = {}
    # pops_locations will store the box corners
    pop_locations = {} # lower left, lower right, upper left, upper right
    for node in dg.G.nodes:
        pops_drawn[node] = False

    intervals = get_relative_intervals(dg)

    if ax == None:
        fig = plt.figure(fignum)
        fig.clf()
        ax = plt.subplot(1,1,1)
    
    # reorder the leaves if not given
    if leaf_order is None:
        leaf_order = copy.copy(dg.leaves)
    
    # order that we draw the population blocks, based on intervals[n][1]
    sorted_pops = sorted(intervals.items(), key=operator.itemgetter(1))
    
    # draw populations starting from the most recent
    # draw all the leaves first, then go through other nodes
    # start with left leaf,
    x0 = 0
    for node in leaf_order:
        #draw_pop(ax, node, x0, dg, pop_locations, intervals,
        #         align_left=x0)
        x0 += padding
        pops_drawn[node] = True 
    
    # plot the interior nodes
    # annotate pulse events with solid arrows
#    for pop in sorted_pops[::-1]:
#        if pops_drawn[pop] == False:
#            draw_pop(ax, pop, dg, pop_locations, intervals)
#            pops_drawn[pop] = True
    
    # draw continuous migration rates as dashed arrows
    
    
    #ax.set_xticks([])
    #ax.set_yticks([])
    
    #ax.set_xlim([min_block-padding, max_block+padding])
    #ax.set_ylim([0, 1.2])
    
    if show == False:
        return ax
    else:
        fig.tight_layout()
        plt.show()


""" this whole function needs rethinking """
#
#def draw_pop(ax, node, lower_left, dg, pop_locations, intervals, align_left=None, stacked=None):
#    """
#    if a split, draw halfway in between
#    if a merger, draw with at least distance padding between
#    """
#    print(dg.G.nodes[node])
#    if 'nu' in dg.G.nodes[node]:
#        # constant size
#        x = dg.G.nodes[node]['nu']
#        if align_left is not None:
#            left_edge = lower_left
#            right_edge = lower_left + x
#            y = [1-intervals[node][0], 1-intervals[node][1]]
#            x1 = [left_edge, left_edge]
#            x2 = [right_edge, right_edge]
#            print(y,x1,x2)
#            ax.fill_betweenx(y, x1, x2, lw=2, hatch='/', ec='darkblue')
#            pop_locations[node] = (lower_left, lower_left+x, intervals[node][1]-intervals[node][0])
#        else:
#            # get center of population
#            if len(dg.successors[node]) == 1:
#                child = dg.successors[node][0]
#                center = np.mean(pop_locations[child][:2])
#            else:
#                child1 = dg.successors[node][0]
#                child2 = dg.successors[node][1]
#                center1 = np.mean(pop_locations[child1][:2])
#                center2 = np.mean(pop_locations[child2][:2])
#                if stacked == None and node in [e[1] for e in stacked]:
#                    for e in stacked:
#                        if node in e and (child1 in e or child2 in e):
#                            break
#                    if e[1] == child1 or e[0] == child1:
#                        center = center1
#                    else:
#                        center = center2
#                else:
#                    center = np.mean([center1, center2])
#
#            left_edge = center-x/2
#            right_edge = center+x/2
#            y = [1-intervals[node][0], 1-intervals[node][1]]
#            x1 = [left_edge, left_edge]
#            x2 = [right_edge, right_edge]
#            print(y,x1,x2)
#            ax.fill_betweenx(y, x1, x2, lw=2, hatch='/', ec='darkblue')
#            pop_locations[node] = (lower_left, lower_left+x, intervals[node][1]-intervals[node][0])
#
#    elif 'nu0' in dg.G.nodes[node] and 'nuF' in dg.G.nodes[node]:
#        # exponential - fill between
#        # need to know if comes from a split, will face exponential in
#        # juxtaposition, or could have it exponential on both sides and
#        # centered?
#        # check if left or right
#        pass
#        if len(dg.predecessors) == 2:
#            # merger, can face right
#        elif len(dg.predecessors) == 1:
#            if len(dg.successors[dg.predecessors[0]]) == 2:
#                # split, get if left or right
#                sister = set(dg.successors[dg.predecessors[0]])-set(node)[0]
#                if 
#            else:
#                # pass, can face right
##        if align_left is not None:
##            y = [1-intervals[node][0], 1-intervals[node][1]]
##            x1 = [intervals[node][0], intervals[node][1]]
##            x2 = [intervals[node][0], intervals[node][1]]
##            ax.fill_betweenx(y, x1, x2, lw=2, hatch='/', ec='darkblue')
#        pass
#    
#        
