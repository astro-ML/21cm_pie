<h2 align="center">Optimal, fast, and robust inference of reionization-era cosmology with the 21cmPIE-INN</h2>

<p align="center">
<a href="https://arxiv.org/abs/2401.04174"><img alt="Arxiv" src="https://img.shields.io/badge/arXiv-2401.04174-b31b1b.svg"></a>


21cm_pie is a machine learning based tool for fast simulations-based inference from simulated 3D 21cm light cone data.  It contains modules to simulate and infer the posterior for a 6d parameter set. 

<img src="animation/animation.gif" width="600" height="600" alt="Animation">

## Installation

```sh
# clone the repository
git clone https://github.com/cosmostatistics/21cm_pie
# then install in dev mode
cd 21cm_pie
pip install --editable .
```

## Usage

Simulating data with [21cmFAST][21cmFAST] and adding noise with [21cmSense][21cmSense] :

[21cmFAST]: https://github.com/21cmfast/21cmFAST
[21cmSense]: https://github.com/jpober/21cmSense

```
twentyone_cm_pie data params/data.yaml --verbose
```
Training the model, typically done in three stages, first the 3D CNN, then the INN and finally both:
```
twentyone_cm_pie train params/train.yaml --verbose
```
Analysing the performance and creating inference plots
```
twentyone_cm_pie plot params/plot.yaml --verbose
```
Trained networks for the inference of simulated (with and without noise) data are stored in ```output/```.

## Acknowledgements

If you use any part of this repository please cite the following paper:

```
@article{Schosser:2024aic,
    author = "Schosser, Benedikt and Heneka, Caroline and Plehn, Tilman",
    title = "{Optimal, fast, and robust inference of reionization-era cosmology with the 21cmPIE-INN}",
    eprint = "2401.04174",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    month = "1",
    year = "2024"
}
```

When using the 3D CNN please cite:

```
@ARTICLE{2022arXiv220107587N,
       author = {{Neutsch}, S. and {Heneka}, C. and {Br{\"u}ggen}, M.},
        title = "{Inferring Astrophysics and Dark Matter Properties from 21cm Tomography using Deep Learning}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Cosmology and Nongalactic Astrophysics, Astrophysics - Astrophysics of Galaxies, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2022,
        month = jan,
          eid = {arXiv:2201.07587},
        pages = {arXiv:2201.07587},
archivePrefix = {arXiv},
       eprint = {2201.07587},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022arXiv220107587N},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

