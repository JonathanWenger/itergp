<br>
<div align="center">
    <img align="center" src="docs/source/assets/img/logo/logo-itergp-txt-right.svg" alt="logo" width="600" style="padding-right: 10px; padding left: 10px;" title="Iterative GP Approximation"/>
</div>
<br>


# IterGP: Computation-Aware Gaussian Process Inference

This repository contains an implementation of the framework described in the paper _Posterior and Computational Uncertainty in Gaussian Processes_.

You can install the Python package via `pip`:

```bash
pip install itergp
```

## Documentation

You can build the documentation locally by running the following command in a terminal:

```bash
tox -e docs
```
You can then view the documentation locally by opening `docs/_build/html/index.html` in a browser.

## Tutorials

To get a feel for how to use IterGP, take a look at the tutorials in the documentation.


## Datasets

Any datasets used in the experiments can be accessed via the API:

```python
from itergp import datasets

data = datasets.uci.BikeSharing(dir="experiments/data/uci")
data.train.y
# array([ 0.20011634, -2.74432264,  0.14604912, ...,  0.40556032,
#         0.57590568, -0.54709806])
```

If the dataset is not already cached, it will be downloaded and cached locally.
