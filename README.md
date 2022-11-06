<br>
<div align="center">
    <img align="center" src="docs/source/assets/img/logo/logo-itergp-txt-right.svg" alt="logo" width="600" style="padding-right: 10px; padding left: 10px;" title="Iterative GP Approximation"/>
</div>
<br>


<div align="center">

<h4 align="center">
  <a href="https://itergp.readthedocs.io">Home</a> |
  <a href="https://itergp.readthedocs.io/en/latest/tutorials.html">Tutorials</a> |  
  <a href="https://itergp.readthedocs.io/en/latest/api.html">API Reference</a> |
  <a href="https://arxiv.org/abs/2205.15449">Research Paper</a>
</h4>

<!-- [![CI build](https://img.shields.io/github/workflow/status/probabilistic-numerics/probnum/Linting?style=flat-square&logo=github&logoColor=white&label=CI-build)](https://github.com/probabilistic-numerics/probnum/actions?query=workflow%3ACI-build)
[![Coverage Status](https://img.shields.io/codecov/c/gh/probabilistic-numerics/probnum/main?style=flat-square&label=Coverage&logo=codecov&logoColor=white)](https://codecov.io/gh/probabilistic-numerics/probnum/branch/main)
[![PyPI](https://img.shields.io/pypi/v/probnum?style=flat-square&label=PyPI&logo=pypi&logoColor=white)](https://pypi.org/project/probnum/) -->

</div>

# IterGP: Computation-Aware Gaussian Process Inference

This repository contains an implementation of the framework described in the paper [Posterior and Computational Uncertainty in Gaussian Processes](https://arxiv.org/abs/2205.15449).


## Installation

You can install the Python package via `pip`:

```bash
pip install git+https://github.com/JonathanWenger/itergp.git
```

## Documentation and Tutorials

To understand how to use the functionality of IterGP, take a look at the [API reference](https://itergp.readthedocs.io/en/latest/api.html) and the [tutorials](https://itergp.readthedocs.io/en/latest/tutorials.html).


## Datasets

Any datasets used in the experiments can be accessed via the API:

```python
from itergp import datasets

data = datasets.uci.BikeSharing(dir="data/uci")
data.train.y
# array([ 0.20011634, -2.74432264,  0.14604912, ...,  0.40556032,
#         0.57590568, -0.54709806])
```

If the dataset is not already cached, it will be downloaded and cached locally.

## Citation

```bibtex
@inproceedings{wenger2022itergp,
  author    = {Jonathan Wenger and Geoff Pleiss and Marvin Pf{\"o}rtner and Philipp Hennig and John P. Cunningham},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  keywords  = {gaussian processes, probabilistic numerics, numerical analysis},
  title     = {Posterior and Computational Uncertainty in {G}aussian processes},
  url       = {https://arxiv.org/abs/2205.15449},
  year      = {2022}
}
```
