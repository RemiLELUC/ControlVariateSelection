# Control Variate Selection for Monte Carlo Integration

This is the code associated to the research article "Control Variate Selection for Monte Carlo Integration", Rémi LELUC, François PORTIER, Johan SEGERS. See [PDF](https://rdcu.be/cnesX)

This implementation is made by [Rémi LELUC](https://remileluc.github.io/)

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

## Citation

> @article{leluc2021control,
  title={Control variate selection for Monte Carlo integration},
  author={Leluc, R{\'e}mi and Portier, Fran{\c{c}}ois and Segers, Johan},
  journal={Statistics and Computing},
  volume={31},
  number={4},
  pages={50},
  year={2021},
  publisher={Springer}
}
>
