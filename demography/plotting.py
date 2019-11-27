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
               show_pulses=True, show=False, offset=0.01, buffer=0.025,
               padding=0.1):
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
    
    padding gives the distance between split populations
    offset and buffer are used to adjust how close the arrows get to the boxes
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
        draw_edge(ax, edge, dg, pop_locations, offset=offset, buffer=buffer)
    
    if show_pulses:
        draw_pulses(ax, dg, pop_locations, buffer)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    if show == False:
        return ax
    else:
        fig.tight_layout()
        plt.show()


"""
Functions to plot a visualization with sizes, migration events, and admixtures.
"""

def plot_demography(dg, fignum=1, leaf_order=None, ax=None,
                    show=False, padding=0.5, stacked=None, flipped=[],
                    root_length=0.2, color='darkblue', hatch=None,
                    boundaries=True, migrations=True):

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
    we would set stacked=[('A','B')]. Note that ('A','B') would work, but
    ('B','A') would not, since the entries need to match the directed edge in
    the DemoGraph object.
    """
    assert leaf_order is not None, "specify leaf_order=[...]"
    assert [l in dg.leaves for l in leaf_order], "if leaf_order given, must include all leaves"
    assert not ax == None or show == False, "cannot show plot if passing axis"
    
    # We'll draw populations in reverse order of their extinction
    pops_drawn = {}
    # pops_locations will store the box corners
    pop_locations = {} # lower left x, lower right x, upper left , upper right x
    for node in dg.G.nodes:
        pops_drawn[node] = False

    intervals = get_relative_intervals(dg)

    intervals[dg.root] = (-root_length,0) 

    if ax == None:
        fig = plt.figure(fignum)
        fig.clf()
        ax = plt.subplot(1,1,1)
    
    # reorder the leaves if not given
    if leaf_order is None:
        leaf_order = copy.copy(dg.leaves)
    
    # order that we draw the population blocks, based on intervals[n][1]
    all_pops = list(intervals.keys())
    bottom_times = [intervals[p][1] for p in all_pops]
    sorted_pops = [x for _,x in sorted(zip(bottom_times,all_pops))]
    
    # draw populations starting from the most recent
    # draw all the leaves first, then go through other nodes
    # start with left leaf,
    x0 = 0
    for node in leaf_order:
        draw_pop(ax, node, dg, pop_locations, intervals,
                 align_left=x0, stacked=stacked, flipped=flipped,
                 c=color, h=hatch, padding=padding)
        x0 += (pop_locations[node][1] - pop_locations[node][0])
        x0 += padding
        pops_drawn[node] = True
    
    # plot the interior nodes
    for node in sorted_pops[::-1]:
        if pops_drawn[node] == False:
            draw_pop(ax, node, dg, pop_locations, intervals,
                     stacked=stacked, flipped=flipped, c=color, h=hatch,
                     padding=padding)
            pops_drawn[node] = True
    
    # draw connection edges from bottom centers to top centers
    for edge in dg.G.edges():
        draw_pop_connections(ax, edge, pop_locations, intervals, color)
        

    # annotate pulse events with solid arrows
    for pop in dg.G.nodes:
        if 'pulse' in dg.G.nodes[pop]:
            for pulse_event in dg.G.nodes[pop]['pulse']:
                draw_pulse_event(ax, pop, pulse_event, pop_locations, intervals)
    
    # draw continuous migration rates as dashed arrows
    if migrations == True:
        draw_migrations(ax, dg, pop_locations, intervals)
    
    # draw edges around all populations
    if boundaries == True:
        draw_boundaries(ax, pop_locations, intervals, dg, color)
    
    # label leaf populations
    for leaf in leaf_order:
        center = np.mean(pop_locations[leaf][:2])
        bottom = 1-intervals[leaf][1]
        ax.text(center, bottom-0.025, leaf, ha='center', va='center')
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    #ax.set_xlim([min_block-padding, max_block+padding])
    #ax.set_ylim([0, 1.2])
    
    if show == False:
        return ax
    else:
        fig.tight_layout()
        plt.show()


def draw_migrations(ax, dg, pop_locations, intervals):
    drawn_heights = []
    buffer = 0.02
    for pop in dg.G.nodes:
        if 'm' in dg.G.nodes[pop]:
            for pop_to in dg.G.nodes[pop]['m']:
                rate = dg.G.nodes[pop]['m'][pop_to]
                if rate > 0:
                    ys_from = intervals[pop]
                    ys_to = intervals[pop_to]
                    ys = (max(ys_from[0], ys_to[0]), min(ys_from[1], ys_to[1]))
                    if ys[1] < ys[0]:
                        continue
                    # plot one arrow in each direction
                    # want to space these - maybe record heights of arrows I've
                    # drawn, so that we can pick a new location if it's too close
                    # to another
                    y = ys[0] + (ys[1]-ys[0])*0.125 + (ys[1]-ys[0])*np.random.rand()*.75
                    count_tries = 0
                    while np.any([abs(y-h)<buffer for h in drawn_heights]):
                        y = ys[0] + (ys[1]-ys[0])*0.125 + (ys[1]-ys[0])*np.random.rand()*.75
                        count_tries += 1
                        if count_tries > 10:
                            buffer /= 2
                            count_tries = 0
                    
                    if pop_locations[pop][1] < pop_locations[pop_to][0]:
                        # use right edge of pop from, left of pop_to
                        y_from, x_l_from, x_r_from = get_xs(pop, dg, pop_locations, intervals)
                        y_to, x_l_to, x_r_to = get_xs(pop_to, dg, pop_locations, intervals)
                        x_from = x_r_from[np.argmin(abs(y_from-y))]
                        x_to = x_l_to[np.argmin(abs(y_to-y))]
                        ax.annotate(
                            '', xy=(x_to, 1-y), xytext=(x_from, 1-y),
                            xycoords='data', textcoords='data',
                            arrowprops={'arrowstyle': '->', 'ls': 'dashed', 'lw': np.log(1+rate)})
                        drawn_heights.append(y)
                    if pop_locations[pop][0] > pop_locations[pop_to][1]:
                        # use left edge of pop from, right of pop_to
                        y_from, x_l_from, x_r_from = get_xs(pop, dg, pop_locations, intervals)
                        y_to, x_l_to, x_r_to = get_xs(pop_to, dg, pop_locations, intervals)
                        x_from = x_l_from[np.argmin(abs(y_from-y))]
                        x_to = x_r_to[np.argmin(abs(y_to-y))]
                        ax.annotate(
                            '', xy=(x_to, 1-y), xytext=(x_from, 1-y),
                            xycoords='data', textcoords='data',
                            arrowprops={'arrowstyle': '->', 'ls': 'dashed', 'lw': np.log(1+rate)})
                        drawn_heights.append(y)


def draw_pulse_event(ax, pop_from, pulse_event, pop_locations, intervals):
    pop_to, time_frac, weight = pulse_event
    xs_from = pop_locations[pop_from][:2]
    xs_to = pop_locations[pop_to][:2]
    if xs_from[0] > xs_to[1]:
        x_from = xs_from[0]
        x_to = xs_to[1]
    else:
        x_from = xs_from[1]
        x_to = xs_to[0]
    
    y = intervals[pop_from][0] + time_frac*(intervals[pop_from][1]-intervals[pop_from][0])
    y = 1-y
    
    ax.annotate(
        '', xy=(x_to, y), xycoords='data',
        xytext=(x_from, y), textcoords='data',
        arrowprops={'arrowstyle': '->'})
    ax.annotate(
        f'{weight}', xy=(np.mean([x_from, x_to]), y), xycoords='data',
        xytext=(0, 3), textcoords='offset points', fontsize=8, ha='center')



def draw_boundaries(ax, pop_locations, intervals, dg, color):
    # draw edges around populations, except where two touch on top/bottom
    for pop in dg.G.nodes:
        # draw left and right edges
        y, x_l, x_r = get_xs(pop, dg, pop_locations, intervals)
        ax.plot(x_l, 1-y, color=color, lw=1)
        ax.plot(x_r, 1-y, color=color, lw=1)
        
        # draw bottoms
        x0, x1 = pop_locations[pop][:2]
        y = 1-intervals[pop][1]
        if pop in dg.leaves:
            ax.plot((x0, x1), (y, y), color=color, lw=1)
        else:
            # draw bottom, minus overlap with tops of children
            x0, x1 = pop_locations[pop][:2]
            y = 1-intervals[pop][1]
            xs = [(x0, x1)]
            if pop in dg.successors:
                for succ in dg.successors[pop]:
                    s0, s1 = pop_locations[succ][2:]
                    for ii,(x0, x1) in reversed(list(enumerate(xs))):
                        if s0 >= x1: # to the right
                            continue
                        elif s1 <= x0: # to the left
                            continue
                        else:
                            if s0 <= x0:
                                if s1 >= x1:
                                    # don't plot
                                    xs.pop(ii)
                                    continue
                                else:
                                    x0 = s1
                                    xs[ii] = (x0, x1)
                                    continue
                            else:
                                if s1 >= x1:
                                    # don't plot
                                    x1 = s0
                                    xs[ii] = (x0, x1)
                                else:
                                    xs[ii] = (x0, s0)
                                    xs.append((s1, x1))
            for (x0, x1) in xs:
                ax.plot((x0, x1), (y, y), color=color, lw=1)
                        
        # draw tops
        x0, x1 = pop_locations[pop][2:]
        y = 1-intervals[pop][0]
        if pop == dg.root:
            ax.plot((x0, x1), (y, y), color=color, lw=1)
        else:
            # draw tops, minus overlap with bottoms of children
            xs = [(x0, x1)]
            for pred in dg.predecessors[pop]:
                s0, s1 = pop_locations[pred][2:]
                for ii,(x0, x1) in enumerate(xs[::-1]):
                    if s0 >= x1: # to the right
                        continue
                    elif s1 <= x0: # to the left
                        continue
                    else:
                        if s0 <= x0:
                            if s1 >= x1:
                                # don't plot
                                xs.pop(ii)
                                continue
                            else:
                                x0 = s1
                                xs[ii] = (x0, x1)
                                continue
                        else:
                            if s1 >= x1:
                                # don't plot
                                x1 = s0
                                xs[ii] = (x0, x1)
                            else:
                                xs[ii] = (x0, s0)
                                xs.append((s1, x1))
            for (x0, x1) in xs:
                ax.plot((x0, x1), (y, y), color=color, lw=1)

def draw_pop_connections(ax, edge, pop_locations, intervals, color):
    pop_from, pop_to = edge
    from_left, from_right = pop_locations[pop_from][:2]
    to_left, to_right = pop_locations[pop_to][2:]
    # if they overlap at all, don't draw the line
    if from_right < to_left:
        y_from = intervals[pop_from][1]
        y_to = intervals[pop_to][0]
        assert y_to == y_from, "y_to is not y_from in pop_connections"
        # draw line/arrow
        ax.plot((from_right, to_left), (1-y_to,1-y_to), color=color, lw=1)
    elif from_left > to_right:
        y_from = intervals[pop_from][1]
        y_to = intervals[pop_to][0]
        assert y_to == y_from, "y_to is not y_from in pop_connections"
        # draw line/arrow
        ax.plot((to_right, from_left), (1-y_to,1-y_to), color=color, lw=1)


def draw_pop(ax, node, dg, pop_locations, intervals, align_left=None,
             stacked=None, flipped=[], c='k', h='/', padding=0.5):
    """
    if a split, draw halfway in between
    if a merger, draw with at least distance padding between
    """
    # if the population is stacked, we align the center of its bottom with the
    # top of its child block
    # align_left is only given for leaf populations
    
    # here, find bottom_center
    if stacked is not None:
        stacked_pops = [e[0] for e in stacked]
    if align_left is None:
        # check if stacked
        if stacked is not None and node in stacked_pops:
            stack_ind = stacked_pops.index(node)
            pop_below = stacked[stack_ind][1]
            assert pop_below in dg.successors[node], f"pop_below is not in successors, {pop_below}, {node}"
            bottom_center = np.mean(pop_locations[pop_below][2:])
        else:
            # is not stacked
            # if it has two descendents, place it between them
            if len(dg.successors[node]) == 2:
                child1, child2 = dg.successors[node]
                # get top centers of left and right children
                top_center1 = np.mean(pop_locations[child1][2:])
                top_center2 = np.mean(pop_locations[child2][2:])
                bottom_center = np.mean([top_center1, top_center2])
            # if only one descendent, check if it is involved in a merger
            else:
                if len(dg.predecessors[dg.successors[node][0]]) == 2:
                    # merger
                    # if so, place on left or right with padding
                    order = get_merger_order(dg.successors[node][0], dg,
                                             pop_locations)
                    succ_center = np.mean(pop_locations[dg.successors[node][0]][2:])
                    if order.index(node) == 0:
                        # on left
                        if 'nu' in dg.G.nodes[node]:
                            bottom_center = succ_center - padding/2 - dg.G.nodes[node]['nu']/2
                        else:
                            bottom_center = succ_center - padding/2 - dg.G.nodes[node]['nuF']/2
                    else:
                        #on right
                        if 'nu' in dg.G.nodes[node]:
                            bottom_center = succ_center + padding/2 + dg.G.nodes[node]['nu']/2
                        else:
                            bottom_center = succ_center + padding/2 + dg.G.nodes[node]['nuF']/2
                else:
                    # if not, place on top
                    bottom_center = np.mean(pop_locations[dg.successors[node][0]][2:])
                
    else:
        bottom_left = align_left
        if 'nu' in dg.G.nodes[node]:
            # constant size
            bottom_right = bottom_left + dg.G.nodes[node]['nu']
        elif 'nuF' in dg.G.nodes[node]:
            bottom_right = bottom_left + dg.G.nodes[node]['nuF']
        bottom_center = np.mean([bottom_left, bottom_right])
    
    # asign pop corners
    get_pop_corners(node, bottom_center, dg, pop_locations, intervals, flipped)
    
    # get the limits of the pop
    y, x_l, x_r = get_xs(node, dg, pop_locations, intervals)

    ax.fill_betweenx(1-y, x_l, x_r, 
                     facecolor=c, hatch=h, alpha=0.5)

def get_pop_corners(node, bottom_center, dg, pop_locations, intervals, flipped):
    # if the population has exponential change, it faces right, unless we have
    # listed it in flipped, in which case it faces left (easier than automating
    # the decision of which way to face it
    if 'nu' in dg.G.nodes[node]:
        # constant size, rectangle
        bottom_left = top_left = bottom_center - dg.G.nodes[node]['nu']/2
        bottom_right = top_right = bottom_center + dg.G.nodes[node]['nu']/2
    elif 'nuF' in dg.G.nodes[node]:
        bottom_left = bottom_center - dg.G.nodes[node]['nuF']/2
        bottom_right = bottom_center + dg.G.nodes[node]['nuF']/2
        # if a split, check if it faces left
        # otherwise it faces right
        if node in flipped:
            top_right = bottom_right
            top_left = top_right - dg.G.nodes[node]['nu0']
        else:
            top_left = bottom_left
            top_right = top_left + dg.G.nodes[node]['nu0']
    pop_locations[node] = [bottom_left, bottom_right, top_left, top_right]


def get_xs(node, dg, pop_locations, intervals):
    [bottom_left, bottom_right, top_left, top_right] = pop_locations[node]
    y0, yF = intervals[node]
    y = np.linspace(y0, yF, 21)
    if bottom_left == top_left:
        x_l = np.ones(len(y)) * bottom_left
    else:
        nu0 = dg.G.nodes[node]['nu0']
        nuF = dg.G.nodes[node]['nuF']
        x_l = bottom_right - nu0 * np.exp(np.log(nuF/nu0)*(y-y0)/(yF-y0))
    if bottom_right == top_right:
        x_r = np.ones(len(y)) * bottom_right
    else:
        nu0 = dg.G.nodes[node]['nu0']
        nuF = dg.G.nodes[node]['nuF']
        x_r = nu0 * np.exp(np.log(nuF/nu0)*(y-y0)/(yF-y0)) + bottom_left
    return y, x_l, x_r
