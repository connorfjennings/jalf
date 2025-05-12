# jalf
__jalf__ (jax-ed alf) is (yet another) Python translation of the Absorption Line Fitter [__alf__](https://github.com/cconroy20/alf/tree/master/src). There is already another pythonified version of __alf__, [__alfpy__](https://github.com/menggu-astro/alfpy), made by Meng Gu. __alfpy__ does sampling via Dynasty and emcee, __jalf__ runs on Numpyro using NUTS (no u-turn sampler). This makes it very fast, and there is lots of Numpyro/jax related stuff out there that you can use it with.


## Overview
__jalf__ uses the same interpolation and smoothing algorithms as __alf__, but re-written to be jit compiled and work with jax's automatic differentiation. This was necissary to make it work with Numpyro, which uses an HMC (hamiltonian monte-carlo) sampler. The Numpyro/jax environment makes setting non-uniform prior distributions easy, and it also samples much more efficiantly than the classic MCMC sampler implemented by emcee (which is also what __alf__ runs on).

Right now, __jalf__ does not have all the bells and whistles that __alf__ or __alfpy__ have, but you can manually re-create some functionallity by directly editing the priors in __jalf.py__. i.e. if you want to keep a parameter fixed you can replace the numpyro distribution with a float.

I thank Charlie Conroy, who wrote the original __alf__, as well as Meng Gu and Aliza Beverage who wrote the python version. Some parts of this code is a direct copy of stuff from __alfpy__.

## Key Features and Differences from the Original Fortran Version
- Samplers: __alfpy__ uses Numpyro's implementation of NUTS
- Performance: __jalf__ is super duper fast. I haven't run proper performance tests yet, but where the original __alf__ took ~100cpu hours to converge, __jalf__ takes about 10 minutes on my laptop. I'll get around to implementing multiprocessing at some point and then do a proper test.
- Dependencies: __alfpy__ requires all the models from the original __alf__ project, located under `alf/infiles/`.

## Installation and Requirements
- download __alf__ and set the ALF_HOME environment variable (in the __alf__ documentation)
- download this repository
- create the python environment:
``` bash
conda env create -f jalf_env.yml
```
- create an environment variable JALF_HOME to path/jalf/


## Usage Instructions
1. Edit `jalf.py` to specify the parameters you want to fit and the default values for those not being fitted
2. With `<filename>.dat` placed in `jalf/indata/`, run the following command:
 `python jalf.py <filename> <tag>`

## Citation
If __jalf__ is helpful in your work, please kindly cite this GitHub repository, as well as all the relevant citations for the original [__alf__](https://github.com/cconroy20/alf/tree/master/src) and [__alfpy__](https://github.com/menggu-astro/alfpy) which are mentioned in their documentations.
