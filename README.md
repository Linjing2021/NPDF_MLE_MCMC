# NPDFfit_MLE_MCMC

A Python package for fitting molecular cloud column density probability distribution functions (N-PDFs) using an **unbinned maximum-likelihood estimator (MLE)** and **Markov Chain Monte Carlo (MCMC)** sampling.

This code accompanies:

> Feng et al., *Trace the Self-Gravitating Gas Using CO Isotopologues*, ApJ (2026).

---

## Features

- Unbinned MLE fitting of N-PDFs
- Support for Lognormal (LN), Power-law (PL), and Lognormal + Power-law (LNPL) models
- Bayesian parameter estimation with MCMC
- Automatic uncertainty estimation
- Publication-quality figures

---

## Repository Structure

```
NPDF_MLE_func.py      Core fitting functions
NPDF_fit_plot.ipynb   Example notebook
NPDF_MLE_para.xlsx    Input parameter table
NH2/                  Input column-density maps
NPDF/                 Output figures
MCMCpara/             Best-fit parameters
MCMCsamples/          MCMC chains
```

---

## Requirements

Python 3.9 or later.

Required packages:

```
numpy
scipy
matplotlib
pandas
astropy
emcee
corner
openpyxl
```

---

## Usage

1. Place the input NH$_2$ FITS files in the `NH2/` directory.
2. Edit the fitting parameters in `NPDF_MLE_para.xlsx`.
3. Open and run `NPDF_fit_plot.ipynb`.

The notebook performs the MLE fitting, runs the MCMC sampling, and automatically saves the best-fit parameters, posterior samples, and figures.

---

## License

This project is released under the **MIT License**.
See the `LICENSE` file for details.

---

## Citation

If you use this software, please cite:
> Feng et al., *Trace the Self-Gravitating Gas Using CO Isotopologues*, ApJ (2026).
Zenodo DOI TBD.

---

## Contact

Linjing Feng,
National Astronomical Observatories, Chinese Academy of Sciences
Email: ljfeng@nao.cas.cn
