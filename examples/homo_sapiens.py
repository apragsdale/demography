"""
Homo sapiens models.
For each model, parameters may be passed. If None are passed, it uses the
default parameters given in each model's documentation, which match the
published parameters from the original cited study.

Note that all sizes are given relative to the ancestral or reference size, so
nu=0.5 would mean that the population is 1/2 the size of the ancestral
population. Time is given in units of 2N_ref generations.

Each function returns a DemoGraph object. Ne is set to the effective population
size inferred or assumed in the original model.
"""

import networkx as nx
import numpy as np
import demography


def initialize_graph():
    G = nx.DiGraph()
    G.add_node('root', nu=1, T=0)
    return G


"""
The Gutenkunst et al (2009) Out of Africa model.
Default parameters (with Ne=7300):
    nuA = 1.685
    TA = 0.219
    nuB = 0.288
    TB = 0.325
    nuEu0 = 0.137
    nuEuF = 4.07
    nuAs0 = 0.0699
    nuAsF = 7.41
    TF = 0.0581
    mAfB = 3.65
    mAfEu = 0.438
    mAfAs = 0.277
    mEuAs = 1.40
"""


def ooa_gutenkunst(params=None, Ne=7300):
    if params is None:
        (nuA, TA, nuB, TB, nuEu0, nuEuF, nuAs0, nuAsF, TF, mAfB, mAfEu, mAfAs,
            mEuAs) = (1.685, 0.219, 0.288, 0.325, 0.137, 4.07, 0.0699, 7.41,
                      0.0581, 3.65, 0.438, 0.277, 1.40)
    else:
        (nuA, TA, nuB, TB, nuEu0, nuEuF, nuAs0, nuAsF, TF, mAfB, mAfEu, mAfAs,
            mEuAs) = params

    G = initialize_graph()
    G.add_node('A', nu=nuA, T=TA)
    G.add_node('B', nu=nuB, T=TB,
               m={'YRI': mAfB})
    G.add_node('YRI', nu=nuA, T=TB+TF,
               m={'B': mAfB, 'CEU': mAfEu, 'CHB': mAfAs})
    G.add_node('CEU', nu0=nuEu0, nuF=nuEuF, T=TF,
               m={'YRI': mAfEu, 'CHB': mEuAs})
    G.add_node('CHB', nu0=nuAs0, nuF=nuAsF, T=TF,
               m={'YRI': mAfAs, 'CEU': mEuAs})
    edges = [('root', 'A'), ('A', 'B'), ('A', 'YRI'), ('B', 'CEU'),
             ('B', 'CHB')]
    G.add_edges_from(edges)
    dg = demography.DemoGraph(G, Ne=Ne)
    return dg


"""
The Tennessen et al (2013) 2 population model.
Default parameters (with Ne=7310):
    nuAf0 = 1.98
    nuAfF = 59.1
    nuB = 0.255
    nuEu0 = 0.141
    nuEu1 = 0.678
    nuEu2 = 36.7
    mAfB = 2.19
    mAfEu = 0.366
    TAf = 0.265
    TOOA = 0.0766
    TEu1 = 0.0490
    TEu2 = 0.0140
"""


def ooa_tennessen(params=None, Ne=7310):
    if params is None:
        (nuAf0, nuAfF, nuB, nuEu0, nuEu1, nuEu2, mAfB, mAfEu, TAf, TOOA, TEu1,
            TEu2) = (1.98, 59.1, 0.255, 0.141, 0.678, 36.7, 2.19, 0.366, 0.265,
                     0.0766, 0.0490, 0.0140)
    else:
        (nuA, TA, nuB, TB, nuEu0, nuEuF, nuAs0, nuAsF, TF, mAfB, mAfEu, mAfAs,
            mEuAs) = params

    G = initialize_graph()
    G.add_node('Af0', nu=nuAf0, T=TAf)
    G.add_node('Af1', nu=nuAf0, T=TOOA+TEu1,
               m={'B': mAfB, 'Eu1': mAfB})
    G.add_node('B', nu=nuB, T=TOOA,
               m={'Af1': mAfB})
    G.add_node('Eu1', nu0=nuEu0, nuF=nuEu1, T=TEu1,
               m={'Af1': mAfB})
    G.add_node('YRI', nu0=nuAf0, nuF=nuAfF, T=TEu2,
               m={'CEU': mAfB})
    G.add_node('CEU', nu0=nuEu1, nuF=nuEu2, T=TEu2,
               m={'YRI': mAfB})
    edges = [('root', 'Af0'), ('Af0', 'B'), ('Af0', 'Af1'), ('Af1', 'YRI'),
             ('B', 'Eu1'), ('Eu1', 'CEU')]
    G.add_edges_from(edges)
    dg = demography.DemoGraph(G, Ne=Ne)
    return dg


"""
Browning American model.
Admixed population has 1/6 from AFR, 1/3 from EUR, and 1/2 from ASIA.
"""


def browning_america(params=None, Ne=7310):
    if params is None:
        (nuAf, TAf, nuB, TB, nuEu0, nuEuF, nuAs0, nuAsF, nuAdm0, nuAdmF, TEuAs,
            TAdm, mAfB, mAfEu, mAfAs, mEuAs) = (1.98, 0.265, 0.255, 0.0766,
                                                0.141, 4.66, 0.0758, 6.27,
                                                4.10, 7.48, 0.0621, 0.0008,
                                                2.19, 0.366, 0.114, 0.455)
    else:
        (nuAf, TAf, nuB, TB, nuEu0, nuEuF, nuAs0, nuAsF, nuAdm0, nuAdmF, TEuAs,
            TAdm, mAfB, mAfEu, mAfAs, mEuAs) = params

    tol = 1e-10  # offset admixture pulses so they occur in the correct order
    frac_Eu_pulse = (TEuAs+tol)/(TEuAs+TAdm)
    frac_As_pulse = (TEuAs+2*tol)/(TEuAs+TAdm)

    G = initialize_graph()
    G.add_node('A', nu=nuAf, T=TAf)
    G.add_node('Af0', nu=nuAf, T=TB+TEuAs,
               m={'B': mAfB, 'EUR': mAfEu, 'ASIA': mAfAs})
    G.add_node('B', nu=nuB, T=TB,
               m={'Af0': mAfB})
    G.add_node('AFR', nu=nuAf, T=TAdm,
               m={'EUR': mAfEu, 'ASIA': mAfAs})
    G.add_node('EUR', nu0=nuEu0, nuF=nuEuF, T=TEuAs+TAdm,
               m={'Af0': mAfEu, 'AFR': mAfEu, 'ASIA': mEuAs},
               pulse={('ADMIX', frac_Eu_pulse, 2./3)})
    G.add_node('ASIA', nu0=nuAs0, nuF=nuAsF, T=TEuAs+TAdm,
               m={'Af0': mAfAs, 'AFR': mAfAs, 'EUR': mEuAs},
               pulse={('ADMIX', frac_As_pulse, 1./2)})
    G.add_node('ADMIX', nu0=nuAdm0, nuF=nuAdmF, T=TAdm)
    edges = [('root', 'A'), ('A', 'Af0'), ('A', 'B'), ('B', 'EUR'),
             ('B', 'ASIA'), ('Af0', 'ADMIX'), ('Af0', 'AFR')]
    G.add_edges_from(edges)
    dg = demography.DemoGraph(G, Ne=Ne)
    return dg


"""
Ragsdale Archaic admixture model.
"""


def ragsdale_archaic(Ne=3600):
    (nuA, TA, nuB, TB, nuEu0, nuEuF, nuAs0, nuAsF, TF,
        mAfB, mAfEu, mAfAs, mEuAs,
        TMH, TN, f_mAA_begin, mAA, mNeand, T_arch_end) = (
        3.86, 1.146, 0.244, 0.118, 0.639, 3.01, 0.181, 18.3, 0.172,
        3.76, 0, 0.179, 0.814,
        0.953, 0.287, 0.566, 0.143, 0.0594, 0.0896)

    Ttot = TF+TB+TA+TMH+TN
    TNeand = Ttot-T_arch_end
    TAA = TF+TB+TA+TMH-T_arch_end
    TAA_nomig = f_mAA_begin*TAA
    TAA_mig = (1-f_mAA_begin)*TAA

    G = nx.DiGraph()
    G.add_node('root', nu=1, T=0)
    G.add_node('Neand', nu=1, T=TNeand,
               m={'B': mNeand, 'CEU': mNeand, 'CHB': mNeand})
    G.add_node('MH_AA', nu=1, T=TN)
    G.add_node('AA_nomig', nu=1, T=TAA_nomig)
    G.add_node('AA', nu=1, T=TAA_mig,
               m={'MH': mAA, 'A': mAA, 'YRI': mAA})
    G.add_node('MH', nu=1, T=TMH,
               m={'AA': mAA})
    G.add_node('A', nu=nuA, T=TA,
               m={'AA': mAA})
    G.add_node('B', nu=nuB, T=TB,
               m={'Neand': mNeand, 'YRI': mAfB})
    G.add_node('YRI', nu=nuA, T=TB+TF,
               m={'AA': mAA, 'B': mAfB, 'CEU': mAfEu, 'CHB': mAfAs})
    G.add_node('CEU', nu0=nuEu0, nuF=nuEuF, T=TF,
               m={'Neand': mNeand, 'YRI': mAfEu, 'CHB': mEuAs})
    G.add_node('CHB', nu0=nuAs0, nuF=nuAsF, T=TF,
               m={'Neand': mNeand, 'YRI': mAfAs, 'CEU': mEuAs})
    edges = [('root', 'Neand'), ('root', 'MH_AA'), ('MH_AA', 'AA_nomig'),
             ('MH_AA', 'MH'), ('AA_nomig', 'AA'), ('MH', 'A'), ('A', 'YRI'),
             ('A', 'B'), ('B', 'CEU'), ('B', 'CHB')]
    G.add_edges_from(edges)
    dg = demography.DemoGraph(G, Ne=Ne)
    return dg


"""
Kamm Basal European model.
"""


def kamm_model(Ne=18200):
    """
    8-11 population model (depending on how you count)
    There is some confusion (to me) about the LBK/Sard branches.. need to
    figure that out.
    """
    # population sizes
    generation_time = 25
    N_Losch = 1.92e3
    N_Mbu = 1.73e4
    N_Mbu_Losch = 2.91e4
    N_Han = 6.3e3
    N_Han_Losch = 2.34e3
    N_Nean_Losch = 1.82e4
    N_Nean = 86.9
    N_LBK = 75.7
    N_Sard = 1.5e4
    N_Sard_LBK = 1.2e4
    N_Basal = N_Losch
    N_Ust = N_Basal
    N_MA1 = N_Basal
    # population merge times in years, divided by generation time
    t_Mbu_Losch = 9.58e4 / generation_time
    t_Han_Losch = 5.04e4 / generation_time
    t_Ust_Losch = 5.15e4 / generation_time
    t_Nean_Losch = 6.96e5 / generation_time
    t_MA1_Losch = 4.49e4 / generation_time
    t_LBK_Losch = 3.77e4 / generation_time
    t_Basal_Losch = 7.98e4 / generation_time
    t_Sard_LBK = 7.68e3 / generation_time
    t_GhostWHG_Losch = 1.56e3 / generation_time
    # pulse admixture times and fractions
    p_Nean_to_Eur = 0.0296
    t_Nean_to_Eur = 5.68e4 / generation_time
    p_Basal_to_EEF = 0.0936
    t_Basal_to_EEF = 3.37e4 / generation_time
    p_GhostWHG_to_Sard = 0.0317
    t_GhostWHG_to_Sard = 1.23e3 / generation_time
    # sample_times (in years), divided by estimated generation time
    t_Mbuti = 0
    t_Han = 0
    t_Sardinian = 0
    t_Loschbour = 7.5e3 / generation_time
    t_LBK = 8e3 / generation_time
    t_MA1 = 24e3 / generation_time
    t_UstIshim = 45e3 / generation_time
    t_Altai = 50e3 / generation_time

    # figure out timing of the pulses as proportions along the source branches
    frac_nean_pulse = 1-(t_Nean_to_Eur-t_Altai)/(t_Mbu_Losch-t_Altai)
    frac_losch_pulse = 1

    G = nx.DiGraph()
    G.add_node('root', nu=1, T=0)
    # neanderthals
    G.add_node('Neand_const', nu=N_Nean_Losch/Ne,
               T=(t_Nean_Losch-t_Mbu_Losch)/2/Ne)
    G.add_edge('root', 'Neand_const')
    G.add_node('Neanderthal', nu0=N_Nean_Losch/Ne, nuF=N_Nean/Ne,
               T=(t_Mbu_Losch-t_Altai)/2/Ne,
               pulse={('Ust_Losch', frac_nean_pulse, p_Nean_to_Eur)})
    G.add_edge('Neand_const', 'Neanderthal')

    G.add_node('Mbu_Losch', nu=N_Mbu_Losch/Ne,
               T=(t_Nean_Losch-t_Mbu_Losch)/2/Ne)
    G.add_edge('root', 'Mbu_Losch')

    # mbuti split
    G.add_node('Mbuti', nu=N_Mbu/Ne,
               T=t_Mbu_Losch/2/Ne)
    G.add_node('Basal_Losch', nu=N_Han_Losch/Ne,
               T=(t_Mbu_Losch-t_Basal_Losch)/2/Ne)
    G.add_edges_from([('Mbu_Losch', 'Mbuti'), ('Mbu_Losch', 'Basal_Losch')])

    # basal european split
    G.add_node('Basal Eur', nu=N_Basal/Ne,
               T=(t_Basal_Losch-t_Basal_to_EEF)/2/Ne,
               pulse={('LBK_Sard', 1, p_Basal_to_EEF)})
    G.add_node('Ust_Losch', nu=N_Han_Losch/Ne,
               T=(t_Basal_Losch-t_Ust_Losch)/2/Ne)
    G.add_edges_from([('Basal_Losch', 'Basal Eur'),
                      ('Basal_Losch', 'Ust_Losch')])

    # UstIshim split
    G.add_node('UstIshim', nu=N_Ust/Ne,
               T=(t_Ust_Losch-t_UstIshim)/2/Ne)
    G.add_node('Han_Losch', nu=N_Han_Losch/Ne,
               T=(t_Ust_Losch-t_Han_Losch)/2/Ne)
    G.add_edges_from([('Ust_Losch', 'UstIshim'), ('Ust_Losch', 'Han_Losch')])

    # han split
    G.add_node('Han', nu=N_Han/Ne,
               T=(t_Han_Losch)/2/Ne)
    G.add_node('MA1_Losch', nu=N_Losch/Ne,
               T=(t_Han_Losch-t_MA1_Losch)/2/Ne)
    G.add_edges_from([('Han_Losch', 'Han'), ('Han_Losch', 'MA1_Losch')])

    # MA1 split
    G.add_node('MA1', nu=N_MA1/Ne,
               T=(t_MA1_Losch-t_MA1)/2/Ne)
    G.add_node('LBK_Losch', nu=N_Losch/Ne,
               T=(t_MA1_Losch-t_LBK_Losch)/2/Ne)
    G.add_edges_from([('MA1_Losch', 'MA1'), ('MA1_Losch', 'LBK_Losch')])

    # LBK split
    G.add_node('LBK_Sard', nu=N_Sard_LBK/Ne,
               T=(t_LBK_Losch-t_Sard_LBK)/2/Ne)
    G.add_node('Loschbour', nu=N_Losch/Ne,
               T=(t_LBK_Losch-t_GhostWHG_to_Sard)/2/Ne,
               pulse={('Sardinian', 1, p_GhostWHG_to_Sard)})
    G.add_edges_from([('LBK_Losch', 'LBK_Sard'), ('LBK_Losch', 'Loschbour')])

    # Sardinian-LBK split
    G.add_node('LBK', nu=N_LBK/Ne,
               T=t_Sard_LBK/2/Ne)
    G.add_node('Sardinian', nu=N_Sard/Ne,
               T=t_Sard_LBK/2/Ne)
    G.add_edges_from([('LBK_Sard', 'LBK'), ('LBK_Sard', 'Sardinian')])

    dg = demography.DemoGraph(G, Ne=Ne)
    return dg



"""
Five population model in Jouganous et al (2017), that includes YRI, CEU, CHB,
KHV, and JPT from the Thousand Genomes Project.
"""


def jouganous_five_pop(Ne = 11293):
    NAf = 23721
    NB = 2831
    NEu0 = 2512
    rEu = 0.0016
    NAs0 = 1019
    rAs = 0.0026
    NKhv0 = 2356
    rKhv = 0.01 ## error in Table? This is not the value give
    NJpt0 = 4384
    rJpt = 0.01 ## error?
    mAfB = 16.8e-5
    mAfEu = 1.14e-5
    mAfAs = 0.56e-5
    mEuAs = 4.75e-5
    mChKh = 21.3e-5
    mChJp = 3.3e-5
    TA = 357e3
    TB = 119e3
    TEuAs = 46e3
    TChKh = 9.8e3
    TChJp = 9e3
    
    gens = 29
    
    nuAf = NAf/Ne
    nuB = NB/Ne
    nuEu0 = NEu0/Ne
    nuAs0 = NAs0/Ne
    nuEu0 = NEu0/Ne
    nuKhv0 = NKhv0/Ne
    nuJpt0 = NJpt0/Ne

    nuEuF = nuEu0 * np.exp(rEu * (TEuAs)/gens)
    nuAsF = nuAs0 * np.exp(rAs * (TEuAs)/gens)
    nuAs1 = nuAs0 * np.exp(rAs * (TEuAs-TChKh)/gens)
    nuAs2 = nuAs0 * np.exp(rAs * (TEuAs-TChJp)/gens)
    nuKhvF = nuKhv0 * np.exp(rKhv * (TChKh)/gens)
    nuJptF = nuJpt0 * np.exp(rJpt * (TChJp)/gens)
    
    
    G = nx.DiGraph()
    
    G.add_node('root', nu=1, T=0)
    G.add_node('A', nu=nuAf, T=(TA-TB)/2/Ne/gens)
    G.add_node('B', nu=nuB, T=(TB-TEuAs)/2/Ne/gens, m={'YRI':mAfB})
    G.add_node('YRI', nu=nuAf, T=TB/2/Ne/gens,
        m={'B':mAfB, 'CEU':mAfEu, 'CHB0':mAfAs, 'CHB1':mAfAs, 'CHB':mAfAs})
    G.add_node('CEU', nu0=nuEu0, nuF=nuEuF, T=TEuAs/2/Ne/gens,
        m={'YRI':mAfAs, 'CHB0':mEuAs, 'CHB1':mEuAs, 'CHB':mEuAs})
    G.add_node('CHB0', nu0=nuAs0, nuF=nuAs1, T=(TEuAs-TChKh)/2/Ne/gens,
        m={'YRI':mAfAs, 'CEU':mEuAs})
    G.add_node('CHB1', nu0=nuAs1, nuF=nuAs2, T=(TChKh-TChJp)/2/Ne/gens,
        m={'YRI':mAfAs, 'CEU':mEuAs, 'KHV':mChKh})
    G.add_node('CHB', nu0=nuAs2, nuF=nuAsF, T=TChJp/2/Ne/gens,
        m={'YRI':mAfAs, 'CEU':mEuAs, 'KHV':mChKh, 'JPT':mChJp})
    G.add_node('KHV', nu0=nuKhv0, nuF=nuKhvF, T=TChKh/2/Ne/gens,
        m={'CHB1':mChKh, 'CHB':mChKh})
    G.add_node('JPT', nu0=nuJpt0, nuF=nuJptF, T=TChJp/2/Ne/gens,
        m={'CHB':mChJp})

    G.add_edges_from([
        ('root','A'),
        ('A','YRI'),
        ('A','B'),
        ('B','CEU'),
        ('B','CHB0'),
        ('CHB0','CHB1'),
        ('CHB1','CHB'),
        ('CHB0','KHV'),
        ('CHB1','JPT')
    ])
    
    dg = demography.DemoGraph(G, Ne=Ne)
    return dg
