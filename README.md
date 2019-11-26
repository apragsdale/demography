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

`dadi` can also be installed via bioconda:
```conda install -c bioconda dadi```

To install `moments` and `moments.LD`, clone the moments repository,
```git clone https://aragsdale@bitbucket.org/simongravel/moments.git```
install the dependencies from the moments directory,
```conda install --file requirements.txt```
and then run
```sudo python setup.py install```


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


### Examples


## Computing summary statistics


## Running simulations


## Plotting Demography objects

