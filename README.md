# Control Variates Selection for Monte Carlo Integration

This is the code for the article "Control Variates Selection for Monte Carlo Integration", Rémi LELUC, François PORTIER, Johan SEGERS.

## Description

Folders
- datasets/          : contains the capture-recapture and sonar dataset in .csv format for real data experiments
- results_synthetic/ : contains the results of the numerical experiments for synthetic data.
- source/            : contains the source code for the implementation of the Monte Carlo estimates.

Dependencies in Python 3
- requirements.txt : dependencies

Python scripts
- controlVariates.py: tool functions to build control variates using families of polynomials.
- functions.py      : synthetic functions to integrate over [0,1]^d
- modelMC.py        : implements main class CVMC() with all Monte Carlo estimates

