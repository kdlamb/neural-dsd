# Neural-DSD
Reduced order modeling of cloud droplet size distributions for atmospheric modeling.

|![](https://github.com/kdlamb/neural-dsd/blob/main/Figures/1dCloudEvol_bindists.gif)|![](https://github.com/kdlamb/neural-dsd/blob/main/Figures/1dCloudEvol.gif)|

## Overview
This repo contains a PyTorch implementation of the code for the paper "Reduced Order Modeling for Linearized Representations of Microphysical Process Rates".


## Content
- [Data Preparation](#data-preparation)
- [Intrinsic Dimension Calculation](#id-calculation)
- [Latent Dynamics](#latent-dynamics)
- [Latent Visualization](#latent-visualization)
- [DSD Reconstructions](#dsd-reconstruction)
- [Regime Dependence](#regime-dependence)
  
## Data Preparation

Data sets from the 1D driver model (which uses the Tel Aviv University bin microphysics scheme) are saved as netcdf files and can be found at 10.5281/zenodo.7487288. These files contain the bin distributions, moments, and process rates for 16 different cases (varying N$_{CCN}$, vertical updraft speeds, and sinusoidal driving frequencies).

Bin distributions are prepared as inputs to the machine learning models in the Preprocessing.ipynb notebook.

## Intrinsic Dimension Calculation
The Training-IDCalculation.ipynb notebook determines the intrinsic dimension of the latent representation of collision-coalescence using the method described in Chen et al. 2022.

## Latent Dynamics
The latent dynamics are learned using the Training-LatentDynamics.ipynb notebook.

## Latent Visualization
The LatentVisualization.ipynb contains code to visualize the latent space and compare it against the one and two category moments of the DSD's. This notebook also includes code to plot the latent manifold. 

## DSD Reconstructions
The DSDReconstructions.ipynb contains code to reconstruct the DSD from the latent variables and the moments and to calculate metrics to compare the different DSD representations. Code to train an ensemble of models to reconstruct the DSD for the single category case and the two category case are in Training-BinsFromMoments.ipynb and Training-BinsFromTwoCatMoments.ipynb.

## Regime Dependence
The RegimeDependence.ipynb contains code to visualize cases from the 1D driver model and investigate how different regimes correspond with learned latent variables.

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
