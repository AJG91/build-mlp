# Multi-layer perceptron

[my-website]: https://AJG91.github.io "my-website"
[ca-housing-docs]: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html "ca-housing-docs"

This repository contains code that demonstrates how to create and train a multi-layer perceptron (MLP) on the [California housing][ca-housing-docs] dataset from scikit-learn.

## Getting Started

* This project relies on `python=3.12`. It was not tested with different versions
* Clone the repository to your local machine
* Once you have, `cd` into this repo and create the virtual environment (assuming you have `conda` installed) via
```bash
conda env create -f environment.yml
```
* Enter the virtual environment with `conda activate build-mlp-env`
* Install the packages in the repo root directory using `pip install -e .` (you only need the `-e` option if you intend to edit the source code in `build_mlp/`)


## Example

See [my website][my-website] for examples on how to use this code.

## Citation

If you use this project, please use the citation information provided by GitHub via the **“Cite this repository”** button or cite it as follows:

```bibtex
@software{build_mlp_2025,
  author = {Alberto Garcia},
  title = {Build MLP},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/AJG91/build-mlp},
  license = {MIT}
}
```