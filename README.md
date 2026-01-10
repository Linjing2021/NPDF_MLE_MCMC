README

This repository implements an unbinned maximum-likelihood framework for fitting column-density probability distribution functions (N-PDFs). The method operates directly on pixel-level column-density values without binning, and uses MCMC sampling to infer model parameters and uncertainties.


> step 1: collect all NH2 file in a directory ("./NH2/" in this example), assign this path to the variable “NH2path”. each NH2 file name should start with "sourcename_", e.g. "SgrB2m_NH2.fits".
>
> step 2: run cell 4 with fit == False, have a look at N-PDFs of each cloud firstly.
> 
> step 3: set up fitting parameters in "NPDF_MLE_para.xlsx", the "source" column should match the sourcename in the NH2 file. N0_fit, N1_fit are the choosed fitting range in log10(NH2). Ncutoff gives the lowest threshold to show the N-PDF. xlim, ylim are used to control the show of the N-PDF if specified. If you want to limit the fitting process, tp0 gives the center point of the transition point limitation, tp1 gives the half width of the limitation interval.
> 
> step 4: run ! (sampling, run !)

notes:
1. If you meet the error "The chain is shorter than 50 times the integrated autocorrelation time ...", run more step in mcmc.
2. If you only want to fit a powerlaw tail, change to model = "PL", and write the Nthres in "Nthres_fit_MLE_para.xlsx" manually.
3. Set model = "LN" for only lognormal fit.
4. If you want to re-run any source, delete the corresponding ".npy" file in ./MCMCpara/ and run the fitting cell again.


N-PDF fitting code (MLE + MCMC) by Linjing Feng. Last updated: 2025.12.22.
Thanks to Yaojun Xiao for help with code testing and improvements.