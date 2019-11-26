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
either `moments` () or `dadi` (), and expected multi-population linkage
disequlibrium statistics (LD) using `moments.LD`.

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

To run simulations using `msprime`, we'll need to 


## Running the tests

Once all of these are installed and ready to go, let's run the tests to make
sure everything went smoothly. In the tests directory (`cd tests`):
```
python run_tests.py
```
Everything should come out ok.

## Building a Demography


### Examples


## Computing summary statistics


## Running simulations


## Plotting Demography objects

