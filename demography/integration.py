"""
Functions to integrate moments from the Demography.
Note that moments can handle up to five populations at any given time, so we have a
check that ensures that there are no more than five populations at all times along
the graph.
moments also requires samples to be defined.
Samples can only be defined for leave nodes, and the sampling happens at the end of
the node.

Some of these functions will be also used for integrating moments.LD ans dadi, so we'll
break those out into Integration.py.
"""

import numpy as np

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
    import dadi
    dadi_installed = 1
except:
    dadi_installed = 0


"""
Functions shared between multiple simulation engines.
"""



"""
Functions to evolve moments.LD to get LD statistics (no maximum number of pops)
"""


"""
Functions to evolve using moments to get the frequency spectrum (max 5 pops)
"""


"""
Functions to evolve using dadi to get the frequency spectrum (max 3 pops)
"""

