# neural-dsd
Reduced order modeling of cloud droplet size distributions for atmospheric modeling

## Overview
This repo contains a PyTorch implementation of the code for the paper "Unsupervised Learning of Predictors for Microphysical Process Rates".

## Citation

The preprint for this paper can be found at:

```
@article{Lamb2023,
  title={Unsupervised Learning of Predictors for Microphysical Process Rates},
  author={Lamb, K.D. and van Lier Walqui, M. and Santos, S. and Morrison, H.},
  journal={ESS Open Archive},
  doi = {DOI: 10.22541/essoar.168995384.44033471/v1},
  year={2023}
}
```
## Content
- [Data Preparation](#data-preparation)
- [Training the Model](#training)
- [Intrinsic Dimension Calculation](#id-calculation)
- [Regime Dependence](#regime-dependence)
- [Latent Visualization](#latent-visualization)

## Data Preparation

Data sets from the 1D driver model (which uses the Tel Aviv University bin microphysics scheme) are saved as netcdf files and can be found at 10.5281/zenodo.7487288. These files contain the bin distributions, moments, and process rates for 16 different cases (varying N$_{CCN}$, vertical updraft speeds, and sinusoidal driving frequencies).

Bin distributions are prepared as inputs to the machine learning models in the Preprocessing.ipynb notebook.

## Training the Model

The autoencoder is trained on the prepared data sets using the Training.ipynb notebook.

## Intrinsic Dimension Calculation
The IntrinsicDimension.ipynb notebook determines the intrinsic dimension of the latent representation of collision-coalescence using the method described in Chen et al. 2022.

## Regime Dependence
The RegimeDependence.ipynb contains code to visualize cases from the 1D driver model and investigate how different regimes correspond with learned latent variables.

## Latent Visualization
The LatentVisualization.ipynb contains code to visualize the latent space and compare it against the one and two category moments of the DSD's.

