import demography
import matplotlib.pylab as plt


"""
Generic models
"""
import generic_models

# plot generic IM model
dg = generic_models.split_IM()

fig1 = demography.plotting.plot_demography(dg, fignum=1,
            leaf_order=['pop1', 'pop2'], show=False, 
            padding=3, root_length=0.5)

ax = fig1.get_axes()[0]
ax.set_title('Generic IM model')
fig1.tight_layout()


"""
Human models
"""
import homo_sapiens


# plot the Gutenkunst out of Africa model
dg = homo_sapiens.ooa_gutenkunst()

fig2 = demography.plotting.plot_demography(dg, fignum=2,
            leaf_order=['YRI', 'CEU', 'CHB'], show=False, flipped=['CEU'],
            stacked=[('A','YRI')], padding=1, gen=25)

ax = fig2.get_axes()[0]
ax.set_title('Gutenkunst Out-of-Africa model')
fig2.tight_layout()


# the Tennessen two-population out of Africa model
dg = homo_sapiens.ooa_tennessen()

fig3 = demography.plotting.plot_demography(dg, fignum=3,
            leaf_order=['YRI', 'CEU'], show=False, flipped=['Af1', 'YRI'],
            stacked=[('Af0', 'Af1')], padding=5, gen=25)

ax = fig3.get_axes()[0]
ax.set_title('Tennessen two-population model')
fig3.tight_layout()


# the Browning America model
dg = homo_sapiens.browning_america()

fig4 = demography.plotting.plot_demography(dg, fignum=4,
            leaf_order=['AFR', 'EUR', 'ASIA', 'ADMIX'], show=False,
            flipped=['EUR'],
            stacked=[('Af0','AFR'), ('A','Af0')], padding=1, gen=25)

ax = fig4.get_axes()[0]
ax.set_title('Browning American admixture model')
fig4.tight_layout()


# Ragsdale archaic model
dg = homo_sapiens.ragsdale_archaic()

fig5 = demography.plotting.plot_demography(dg, fignum=5,
            leaf_order=['AA', 'YRI', 'CEU', 'CHB', 'Neand'], show=False,
            flipped=['CEU'],
            stacked=[('A','YRI'), ('MH','A'), ('MH_AA','MH'), ('root','MH_AA')],
            padding=1, gen=29)

ax = fig5.get_axes()[0]
ax.set_title('Ragsdale archaic admixture model')
fig5.tight_layout()


# Kamm European model
dg = homo_sapiens.kamm_model()

fig6 = plt.figure(6, figsize=(10, fig5.get_figheight()))
ax = plt.subplot(111)
demography.plotting.plot_demography(dg, ax=ax,
            leaf_order=['Mbuti','Basal Eur','LBK','Sardinian',
                        'Loschbour','MA1','Han','UstIshim','Neanderthal'],
            flipped=['Neanderthal'], padding=2,
            stacked=[('root','Mbu_Losch'),('Mbu_Losch','Basal_Losch'),
                     ('Basal_Losch','Ust_Losch'),('Ust_Losch','Han_Losch'),
                     ('Han_Losch','MA1_Losch'),('MA1_Losch','LBK_Losch'),
                     ('LBK_Losch','Loschbour'),('Neand_const','Neanderthal')],
            gen=25)

ax.set_title('Kamm ancient Eurasia model')
fig6.tight_layout()


"""
Drosophila models
"""
import drosophila_melanogaster

dg = drosophila_melanogaster.li_stephan()

fig7 = demography.plotting.plot_demography(dg, fignum=7,
            leaf_order=['AFR', 'EUR'], show=False,
            stacked=[('A0','AFR')],
            padding=1, gen=1./10)

ax = fig7.get_axes()[0]
ax.set_title('Li and Stephan Drosophila melanogaster model')
fig7.tight_layout()


"""
Orangutan models
"""
import pongo

dg = pongo.locke_IM()

fig8 = demography.plotting.plot_demography(dg, fignum=8,
            leaf_order=['Bornean', 'Sumatran'], show=False,
            padding=2, gen=20)

ax = fig8.get_axes()[0]
ax.set_title('Locke orangutan model')
fig8.tight_layout()


fig1.savefig('figures/generic_IM.pdf')
fig2.savefig('figures/gutenkunst_ooa.pdf')
fig3.savefig('figures/tennessen_two_pop.pdf')
fig4.savefig('figures/browning_america.pdf')
fig5.savefig('figures/ragsdale_archaic.pdf')
fig6.savefig('figures/kamm_eurasian.pdf')
fig7.savefig('figures/li_stephan_dmel.pdf')
fig8.savefig('figures/locke_pongo.pdf')
plt.show()

