# Demography

Repository for Demography Building as graphical objects, a Python package.

Demographies are encoded as directed acyclic graphs using `networkx`,
which are then parsed as a demography object. From that demography
object, we can call various simulation engines to either return 
simulated sequences or expectations for common summary statistics
used in inference.

Right now, we have support for running simulations over the demography
using `msprime` (https://msprime.readthedocs.io/en/stable/), which
simulates sequences under either the Hudson or Discrete Time Wright Fisher
models. We can also get the expected site frequency spectrum (SFS) using
either `moments` (https://bitbucket.org/simongravel/moments/) or `dadi` 
(https://bitbucket.org/gutenkunstlab/dadi/), and expected multi-population
linkage disequlibrium statistics (LD) using `moments.LD` (which is packaged
with `moments1).

## Getting Started

This will get you up and running, after cloning this repository.

We first need to install the dependencies.
If this is not your first Python excursion, you will already have most
installed, including `numpy`, `collections`, and `matplotlib`. These basic
requirements can be installed from the requirements.txt file, using either
conda:
```
conda install --file requirements.txt
```
or pip:
```
pip install -r requirements.txt
```

As is, `demography` can be used to plot and visualize demographies. If we
want to run simulations or compute statistics, we'll need to install the
simulation engines that we want.

To run simulations using `msprime`, we'll need to install it. Instructions
can be found in the `msprime` docs. If you're using conda:
```
conda config --add channels conda-forge
conda install msprime
```
This will also get you `tskit` for working with tree sequence outputs.

`dadi`, which is used to compute the expected site frequency spectrum (SFS),
can also be installed via bioconda:
```conda install -c bioconda dadi```

If you're familiar using `dadi`, `moments` has a very similar usage and
feel, but can handle up to 5 populations. Additionally, `moments` is packaged
with `moments.LD`, which rapidly computes multi-population LD statistics over
an arbitrary number of populations. To install `moments` and `moments.LD`, 
clone the moments repository,
```
git clone https://aragsdale@bitbucket.org/simongravel/moments.git
```
install the dependencies from the moments directory,
```
conda install --file requirements.txt
```
and then install moments by running
```
sudo python setup.py install
```


## Running the tests

Once all of these are installed and ready to go, let's run the tests to make
sure everything went smoothly. In the tests directory (`cd tests`):
```
python run_tests.py
```
Everything should come out ok.

> **_NOTE:_**  If tests are failing, or if you run into any other bugs, or
if you have any feature requests or find anything unintuitive, please open
an Issue. I am actively maintaining and using this software, and would
greatly appreciate the feedback or questions.

## Building a Demography

Demographic history can be represented as a directed acyclic graph (DAG),
where nodes represent populations, and edges represent the relationship
between populations. The `networkx` package in Python gives us a flexible
and convenient way of specifying DAG demographies.

Note that time and sizes are population-size scaled, so sizes `nu` are
given relative to "Ne", and times `T` are given in units of 2Ne generations.
Migration rates are also population-size scaled, in units of 2 Ne m_{i,j}, 
where m_{i,j}.

This is all probably best seen through some example:

### Examples

All examples are also coded in `models.py` in the examples directory.

#### Multi-epoch model
A single population with multiple epochs. We represent each epoch as its
own node, with edges connecting the different epoch nodes. Each node has
a population size (given by `nu`, the relative size N_e/N_{ref}, or by
`nu0` and `nuF` if it undergoes expontial growth/decay over that epoch).
Each node persists for a given time (the time of the epoch). We also give
each node/epoch a label, which needs to be unique to that node (so multiple
epochs of the same population need to each have their own unique label).

We first initialize the DAG:
```
import networkx as nx
G = nx.DiGraph()
```

Conventionally, we label the root of the demography `root`, though you
can name it anything you like. Also conventionally, it has size `nu=1`
and time `T=0`. That is because this node is assumed to be at demographic
equilibrium before applying any demograhic events.
```
G.add_node('root', nu=1, T=0)
```

Lets suppose the population goes through a period of smaller size (1 half),
followed by a sharp bottleneck and exponential growth:
```
G.add_node('A', nu=1./2, T=0.2)
G.add_node('B', nu0=0.1, nuF=3.0, T=0.1)
G.add_edge('root','A')
G.add_edge('A','B')
```
The `G.add_edge` tells us that `root` is the parent population of `A`, and
`A` is the parent of `B`. This could also be done en masse using
```G.add_edges_from([('root','A'),('A','B')])```.

Now we can create the DemoGraph object:
```
import demography
dg = demography.DemoGraph(G)
```

#### Computing summary statistics

We can compute the SFS using either `moments` or `dadi`. We just need to
specify the population to sample (must be a leaf population) and the number
of haploid samples to take. Let's sample 10 individuals:
```
fs = dg.SFS(['B'], [10])
```
The default engine for computing the SFS is `moments`, but we can also use
`dadi`. This requires specifying the number of grid points to use (either
a single grid point, or a set of three grid points to extrapolate the result -
this is explained over in the dadi docs). Rule of thumb is that the number
of grid points needs to be larger than the sample size:

```
fs_dadi = dg.SFS(['B'], [10], engine='dadi', pts=[30,40,50])
```

You can check that these results align - they should be close!

We can also compute LD statistics. We don't need to specify sample size, but we
do have to set the `pop_ids`, and the `rho` values for the population-size scale
recombination distances to compute LD statistics over. `theta` can also be set:

```
ld = dg.LD(pop_ids=['B'], theta=4*1e-4*1e-8, rho=[0,0.1,1.0,10.0])
```

#### Running a simulation in msprime

If we want to simulate under this demography in `msprime`, it's easy to get
the simulation inputs:
```
pop_configs, mig_mat, demo_events = dg.msprime_inputs(self, Ne=1e4)
```
the samples list:
```
samples = dg.msprime_samples(['B'], [10])
```
or run the simulation, getting the output tree sequence:
```
ts = dg.simulate_msprime(model='hudson', Ne=1e4,
                         pop_ids=['B'], sample_sizes=[10],
                         sequence_length=1e5, recombination_rate=2e-8,
                         recombination_map=None, mutation_rate=None,
                         replicates=None) # int value of replicates gives list of tree seqs
```

### Additional examples

#### Two-population isolation-with-migration

A population splits into two, with subsequent migration between populations:
```
(nu1, nu2, nuA, T, m12, m21) = params
G = nx.DiGraph()
G.add_node('root', nu=nuA, T=0)
G.add_node('pop1', nu=nu1, T=T, m={'pop2':m12})
G.add_node('pop2', nu=nu1, T=T, m={'pop2':m12})
G.add_edges_from([('root','pop1'),('root','pop2')])
dg = demography.DemoGraph(G)

dg.SFS(['pop1','pop2'], [20,20])
```

#### Out-of-Africa human expansion

The well-known OOA model (Gutenkunst et al, 2009) of continental-scale
human expansion in Africa and Eurasia.

```
(nuA, TA, nuB, TB, nuEu0, nuEuF, nuAs0,
    nuAsF, TF, mAfB, mAfEu, mAfAs, mEuAs) = params

G  = nx.DiGraph()
G.add_node('root', nu=1, T=0)
G.add_node('A', nu=nuA, T=TA)
G.add_node('B', nu=nuB, T=TB, m={'YRI':mAfB})
G.add_node('YRI', nu=nuA, T=TB+TF, m={'B':mAfB, 'CEU':mAfEu, 'CHB': mAfAs})
G.add_node('CEU', nu0=nuEu0, nuF=nuEuF, T=TF, m={'YRI':mAfEu, 'CHB':mEuAs})
G.add_node('CHB', nu0=nuAs0, nuF=nuAsF, T=TF, m={'YRI':mAfAs, 'CEU':mEuAs})

G.add_edges_from([('root','A'), ('A','B'), ('A','YRI'), ('B','CEU'),
                  ('B','CHB')])
```

## Plotting Demography objects

We can also plot demographic models using the `demography.plotting` functinos.
There are two main plotting features: `plot_graph` and `plot_demography`. The
first is mainly used to plot the overall topology of the demography, with
splits, mergers, and pulse migration events, and I mainly use it as a visual
debugger. The second fully plots size changes and all migration, representing
population sizes as the size of the block, and drawing continuous migration
rate with dashed arrows.

These can be called by:
```
demography.plotting.plot_graph(dg, leaf_order=leaf_order, leaf_locs=leaf_locs)
```
where `leaf_order` tells us the order to draw the leaf populations horizontally,
and `leaf_locs` tells us the x-coordinates to draw them (so we can control their
spacing). The values in `leaf_locs` have to be between 0 and 1.

We can either plot within a `matpotlib.axes` object, in which we pass the axes
to `ax=ax1`, e.g. Or we can plot it as a stand-alone figure, and it takes `fignum`
as an argument, and will print the plot to screen using `show=True`. `offset`,
`buffer`, and `padding` control the spacing of arrows and populations in the plot.

> **_NOTE:_**  To do: give list of all inputs and usage

The other plotting option is `plot_demography`, which takes slightly different
inputs. We pass `leaf_order` and `padding` for the spacing between populations,
we can pass a list of edges to `stacked` for those that are meant to be
placed vertically instead of staggered. For exponentially growing populations,
we can flip the direction that growth extends, specified in `flipped`.

### Examples

In the examples directory, you can run the `test_plotting.py` script to get a
feel for how these functions work. There are also some ...... finish this up later..
