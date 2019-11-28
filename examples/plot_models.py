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



plt.show()
