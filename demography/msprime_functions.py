"""
Functions to create inputs for msprime simulation and to run simulation from
the DemoGraph object.

Most basic feature returns population_configurations, migration_matrix, and 
demographic_events. Also can run simulation if the DemoGraph object has
samples, sequence length, recombination rate or genetic map, or if those items
are passed to the function.
"""

try:
    import msprime
    msprime_installed = 1
except ImportError:
    msprime_installed = 0

import numpy as np




