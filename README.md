# Control Variate Selection for Monte Carlo Integration

This is the code associated to the research article "Control Variate Selection for Monte Carlo Integration", Rémi LELUC, François PORTIER, Johan SEGERS. Accepted in *Statistics and Computing 31 (2021)*, see [PDF](https://rdcu.be/cnesX).

This implementation is made by [Rémi LELUC](https://remileluc.github.io/).

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

## Abstract

Monte Carlo integration with variance reduction by means of control variates can be implemented by the ordinary least squares estimator for the intercept in a multiple linear regression model with the integrand as response and the control variates as covariates. Even without special knowledge on the integrand, significant efficiency gains can be obtained if the control variate space is sufficiently large. Incorporating a large number of control variates in the ordinary least squares procedure may however result in:

(i) a certain instability of the ordinary least squares estimator

(ii) a possibly prohibitive computation time

Regularizing the ordinary least squares estimator by preselecting appropriate control variates via the Lasso turns out to increase the accuracy without additional computational cost. The findings in the numerical experiment are confirmed by concentration inequalities for the integration error.

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

### Examples

**Dimension d=1**

```python
>>> from modelMC import CVMC
>>> # Integrand f on [0,1] (integral is pi)
>>> def f(x): return 1/np.sqrt(x*(1-x))
>>> # Loop over replications to check variance reduction
>>> N = 100 # number of replications
>>> # Initialize results
>>> I_mc_list = np.zeros(N) 
>>> I_ols_list = np.zeros(N)
>>> I_lasso_list = np.zeros(N)
>>> I_lslx_list = np.zeros(N)
>>> for i in range(N):
... # Instance of Control Variate Monte Carlo estimator
... cvmc = CVMC(seed=i,func=f,n=500,d=1,k=20,law='uniform',basis='legendre')
... # Compute different MC estimates
... I_mc = cvmc.get_van()
... I_olsmc = cvmc.get_ols(deg=20)
... I_lassomc = cvmc.get_lasso(deg=20,c1=200,c2=300)
... I_lslx = cvmc.get_lslassox(deg=20,n_sample=200,c1=50,c2=500)
... # Store the results
... I_mc_list[i]=I_mc
... I_ols_list[i]=I_olsmc
... I_lasso_list[i]=I_lassomc
... I_lslx_list[i]=I_lslx
>>> # Check Mean Squared Errors
>>> print('MSE MC     :',np.mean((I_mc_list-np.pi)**2))
>>> print('MSE OLSMC  :',np.mean((I_ols_list-np.pi)**2))
>>> print('MSE LASSOMC:',np.mean((I_lasso_list-np.pi)**2))
>>> print('MSE LSLXMC :',np.mean((I_lslx_list-np.pi)**2))
MSE MC     : 3.25e-02
MSE OLSMC  : 9.50e-03
MSE LASSOMC: 9.89e-03
MSE LSLXMC : 9.50e-03
```

**General dimension**

```python
>>> from modelMC import CVMC
>>> # Integrand phi on [0,1] (integral is 0)
>>> def phi(x):
... return np.sin(np.pi*(2*np.mean(x) - 1)) 
>>> # Loop over replications to check variance reduction
>>> N = 100 # number of replications
>>> # Initialize results
>>> I_mc_list = np.zeros(N) 
>>> I_ols_list = np.zeros(N)
>>> I_lasso_list = np.zeros(N)
>>> I_lslx_list = np.zeros(N)
>>> for i in range(N):
... # Instance of Control Variate Monte Carlo estimator
... cvmc = CVMC(seed=i,func=f,n=500,d=dim,k=5,law='uniform',basis='legendre')
... # Compute different MC estimates
... I_mc = cvmc.get_van()
... I_olsmc = cvmc.get_ols(deg=3)
... I_lassomc = cvmc.get_lasso(deg=3,c1=200,c2=500)
... I_lslx = cvmc.get_lslassox(deg=3,n_sample=200,c1=50,c2=500)
... # Store the results
... I_mc_list[i]=I_mc
... I_ols_list[i]=I_olsmc
... I_lasso_list[i]=I_lassomc
... I_lslx_list[i]=I_lslx
>>> # Check Mean Squared Errors
>>> print('MSE MC     :',np.mean((I_mc_list)**2))
>>> print('MSE OLSMC  :',np.mean((I_ols_list)**2))
>>> print('MSE LASSOMC:',np.mean((I_lasso_list)**2))
>>> print('MSE LSLXMC :',np.mean((I_lslx_list)**2))
MSE MC     : 8.21e-04
MSE OLSMC  : 3.34e-07
MSE LASSOMC: 3.29e-07
MSE LSLXMC : 3.34e-07
```


