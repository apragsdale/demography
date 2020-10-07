import numpy as np
import copy
from . import integration
from itertools import combinations_with_replacement

from . import tmrcas
from . import util
_choose = util._choose


def tmrca_vector(order_x, order_y, num_pops):
    pass


def compute_twolocus_tmrcas(dg, pop_ids, Ne, order_x, order_y, r):
    """
    Random mating, multiple populations. Compute moments of two-locus Tmrcas.
    """

    # get events
    (present_pops, integration_times, nus, migration_matrices, frozen_pops,
        selfing_rates, events) = integration.get_moments_arguments(dg)

    
