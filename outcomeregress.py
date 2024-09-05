import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Gaussian, Binomial
from statsmodels.genmod.families.links import identity, logit

def dr(data, outcome, treatvar, ovars=None, pvars=None, family='gaussian', link='identity', vce='robust', genvars=None, debug=False):
    # Setting defaults
    if ovars is None:
        ovars = []
    if pvars is None:
        pvars = []
    if family == 'gaussian':
        family = Gaussian()
    if link == 'identity':
        link = identity()

    print("\nDoubly Robust Estimate of the effect of", treatvar, "on", outcome)

    # Filter data based on treatment variable
    treatment_values = data[treatvar].unique()
    if len(treatment_values) != 2:
        raise ValueError("The treatment variable must only take 2 values in the sample.")
    
    exp = treatment_values[1]  # Assumes the second value is the treated group
    data['treat'] = (data[treatvar] == exp).astype(int)

    # Fit propensity model
    model_propensity = sm.Logit(data['treat'], sm.add_constant(data[pvars]))
    result_propensity = model_propensity.fit(disp=debug)
    data['ptreat'] = result_propensity.predict(sm.add_constant(data[pvars]))
    data['iptwt'] = data['treat'] / data['ptreat'] + (1 - data['treat']) / (1 - data['ptreat'])
    data['ipt_est'] = (2 * data['treat'] - 1) * data[outcome] * data['iptwt']

    # Fit outcome model for treated
    model_mu1 = GLM(data[outcome][data['treat'] == 1], sm.add_constant(data[ovars]), family=family, link=link)
    result_mu1 = model_mu1.fit()
    data['mu1'] = result_mu1.predict(sm.add_constant(data[ovars]))

    # Fit outcome model for control
    model_mu0 = GLM(data[outcome][data['treat'] == 0], sm.add_constant(data[ovars]), family=family, link=link)
    result_mu0 = model_mu0.fit()
    data['mu0'] = result_mu0.predict(sm.add_constant(data[ovars]))

    data['mudiff'] = data['mu1'] - data['mu0']

    # Combine into robust estimate
    data['mdiff'] = (-1 * (data['treat'] - data['ptreat']) * data['mu1'] / data['ptreat']) - \
                    ((data['treat'] - data['ptreat']) * data['mu0'] / (1 - data['ptreat']))
    data['drdiff1'] = data['ipt_est'] + data['mdiff']
    data['dr1'] = data['treat'] * data[outcome] / data['ptreat'] - (data['treat'] - data['ptreat']) * data['mu1'] / data['ptreat']
    data['dr0'] = (1 - data['treat']) * data[outcome] / (1 - data['ptreat']) + (data['treat'] - data['ptreat']) * data['mu0'] / (1 - data['ptreat'])
    data['drdiff2'] = data['dr1'] - data['dr0']

    dr1_mean = data['dr1'].mean()
    dr0_mean = data['dr0'].mean()
    dr_est = data['drdiff2'].mean()

    n = len(data)

    # Calculate Standard Error
    data['I'] = data['dr1'] - data['dr0'] - dr_est
    data['I2'] = data['I'] ** 2
    dr_var = data['I2'].mean() / n

    # Display results
    print("\nResults:")
    print(f"DR Estimate: {dr_est}")
    print(f"DR Variance: {dr_var}")
    print(f"DR1 Mean: {dr1_mean}")
    print(f"DR0 Mean: {dr0_mean}")

    return {
        'dr_est': dr_est,
        'dr_var': dr_var,
        'dr0': dr0_mean,
        'dr1': dr1_mean
    }
