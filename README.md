## GP-pyCHARMM 

This repository contains sample codes and scripts to deminstrate how to combine Gaussian process regression (GPR) with QM/MM in CHARMM through pyCHARMM. 

Currently, a minimal but complete example - based on GPR with derivative observations (GPRwDO) using the Behler-Parrinello (BP) descriptors -
is included, enabling GPRwDO-corrected QM/MM potential energy calculations for the Menshutkin reaction. 

Specifically, the example constains Python scripts for conducting the following steps: 
1. `men_resd_am1.py` (pyCHARMM) - potential energy scan at the AM1/MM level along a reaction coordinate (RC); 
2. `men_sp_am1.py` & `men_sp_b3lyp.py` (pyCHARMM + Gaussian) - single-point energy-and-force calculations at the AM/MM and B3LYP/6-31+G(d,p)/MM 
     levels based on the trajectory produced in step (1);
3. `men_train_gprwdo.py` (GPflow) - trains a BP-GPRwDO model to correct AM1/MM using the data collected in step (2);
4. `men_deploy_gprwdo.py` (GPflow + pyCHARMM) - deploys the optimized BP-GPRwDO model in pyCHARMM to correct AM1/MM for an update RC scan. 

Descriptons of other miscellaneous files/folders:
- `gen1.psf` & `gen1.crd` - pre-generated CHARMM .psf & .crd files for the solvated Menshutkin reaction system; 
- `toppar/` - contains the necessary CHARMM topology and parameter files to run the example;
- `data/` - contains Gaussian input template files to run single-point DFT/MM calculations through Gaussian's interface with CHARMM;
- `scratch/` - a scratch directory needed for Gaussian calculations;
- `dims.py` - contains system-specific dimension information;
- `feature.py` - class/function definitions for the BP descriptors;
- `kext.py` - codes for constructing extended kernels in GPRwDO.
