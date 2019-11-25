"""
Module to handle plotting demographies. These can be used to visually debug a
demography object that has been defined, or to create quality visualizations of
the demography, either as a stand-alone plot or as a subplot axes object. 

Requires matplotlib version >= ??
"""

import numpy as np
import matplotlib.pylab as plt


# Font styles for plots
pop_label_style = dict(size=10, color='darkblue')
frac_style = dict(size=8, color='black')
text_box_style = dict(facecolor='none', edgecolor='darkblue', 
                      boxstyle='round,pad=1', linewidth=1)

def plot_graph(dg, fignum=1,
               ax=None, show=False)
    """
    Ignores population sizes and continuous migration rates, just plots the
    relationships between populations in the DemoGraph. Arrows indicate splits,
    mergers, and pulse migration events, with fraction of contributions shown
    if needed.

    Time is more-or-less ordered from bottom to top as present to past, but
    because populations can persist for different lengths of time, this plot
    isn't meant to be to scale or necessarily indicate the ordering of pulse
    migration events or splits along different lineages.
    """
    assert not ax != None and show == True, "cannot show plot if passing axis"
    
    if ax == None:
        fig = plt.figure(fignum)
        fig.clf()
        ax = plt.subplot(1,1,1)

    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    
    

    if show == False:
        return fig
    else:
        fig.tight_layout()
        plt.show()

def plot_demography():
    pass
