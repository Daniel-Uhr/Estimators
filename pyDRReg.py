import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.utils import resample
from scipy import stats
import statsmodels.formula.api as smf

# Function to calculate robust standard errors using the sandwich estimator
def robust_se(model, X, Y):
    predictions = model.predict(X)
    residuals = Y - predictions
    X_design = np.hstack([np.ones((X.shape[0], 1)), X])  # Add constant term
    bread = np.linalg.inv(X_design.T @ X_design)
    meat = np.sum((residuals ** 2)[:, None, None] * (X_design[:, :, None] @ X_design[:, None, :]), axis=0)
    robust_cov = bread @ meat @ bread
    robust_se = np.sqrt(np.diag(robust_cov))
    return robust_se[1]  # Return the standard error of the treatment effect

# Function to estimate ATE using Outcome Regression
def OR_ate(df, X_cols, T_col, Y_col):
    X = df[X_cols].values
    T = df[T_col].values
    Y = df[Y_col].values
    
    # Model for untreated (D=0)
    model_0 = LinearRegression().fit(X[T == 0], Y[T == 0])
    mu0 = model_0.predict(X)
    
    # Model for treated (D=1)
    model_1 = LinearRegression().fit(X[T == 1], Y[T == 1])
    mu1 = model_1.predict(X)
    
    # Estimate ATE
    OR_ate_estimate = np.mean(mu1 - mu0)
    
    # Calculate robust standard error for ATE
    se_ate = robust_se(model_1, X[T == 1], Y[T == 1]) + robust_se(model_0, X[T == 0], Y[T == 0])
    
    # Calculate confidence interval and p-value
    z_value = OR_ate_estimate / se_ate
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_value)))
    ci_lower = OR_ate_estimate - 1.96 * se_ate
    ci_upper = OR_ate_estimate + 1.96 * se_ate
    
    return {
        'Estimate': OR_ate_estimate,
        'SE': se_ate,
        't-stat': z_value,
        'p-value': p_value,
        'CI': (ci_lower, ci_upper)
    }

# Function to estimate ATT using Outcome Regression
def OR_att(df, X_cols, T_col, Y_col):
    X_treated = df[df[T_col] == 1][X_cols].values
    Y_treated = df[df[T_col] == 1][Y_col].values
    X_control = df[df[T_col] == 0][X_cols].values
    Y_control = df[df[T_col] == 0][Y_col].values
    
    # Models
    model_treated = LinearRegression().fit(X_treated, Y_treated)
    model_control = LinearRegression().fit(X_control, Y_control)
    
    # Predictions
    mu1_X = model_treated.predict(X_treated)
    mu0_X = model_control.predict(X_treated)  # Use treated X for counterfactual
    
    # Estimate ATT
    OR_att_estimate = np.mean(mu1_X - mu0_X)
    
    # Calculate robust standard error for ATT
    se_att = robust_se(model_treated, X_treated, Y_treated) + robust_se(model_control, X_control, Y_control)
    
    # Calculate confidence interval and p-value
    z_value = OR_att_estimate / se_att
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_value)))
    ci_lower = OR_att_estimate - 1.96 * se_att
    ci_upper = OR_att_estimate + 1.96 * se_att
    
    return {
        'Estimate': OR_att_estimate,
        'SE': se_att,
        't-stat': z_value,
        'p-value': p_value,
        'CI': (ci_lower, ci_upper)
    }

# Function to estimate ATE using IPW (Inverse Probability Weighting)
def IPW_ate(df, X_cols, T_col, Y_col):
    # Estimate propensity scores using logistic regression
    formula_pscore = f"{T_col} ~ " + " + ".join(X_cols)
    df['pscore'] = smf.logit(formula_pscore, data=df).fit().predict()
    
    # Calculate weights for ATE
    df['W1'] = 1 / df['pscore']
    df.loc[df[T_col] == 0, 'W1'] = 0
    df['W2'] = 1 / (1 - df['pscore'])
    df.loc[df[T_col] == 1, 'W2'] = 0
    df['W_ATE'] = df['W1'] + df['W2']
    
    # Weighted regression for ATE
    model_ate = smf.wls(f"{Y_col} ~ {T_col}", data=df, weights=df['W_ATE']).fit()
    
    return {
        'Estimate': model_ate.params[T_col],
        'SE': model_ate.bse[T_col],
        't-stat': model_ate.tvalues[T_col],
        'p-value': model_ate.pvalues[T_col],
        'CI': (model_ate.conf_int().loc[T_col, 0], model_ate.conf_int().loc[T_col, 1])
    }

# Function to estimate ATT using IPW (Inverse Probability Weighting)
def IPW_att(df, X_cols, T_col, Y_col):
    # Estimate propensity scores using logistic regression
    formula_pscore = f"{T_col} ~ " + " + ".join(X_cols)
    df['pscore'] = smf.logit(formula_pscore, data=df).fit().predict()
    
    # Calculate weights for ATT
    df['W_ATT'] = df['pscore'] / (1 - df['pscore'])
    df.loc[df[T_col] == 1, 'W_ATT'] = 1
    
    # Weighted regression for ATT
    model_att = smf.wls(f"{Y_col} ~ {T_col}", data=df, weights=df['W_ATT']).fit()
    
    return {
        'Estimate': model_att.params[T_col],
        'SE': model_att.bse[T_col],
        't-stat': model_att.tvalues[T_col],
        'p-value': model_att.pvalues[T_col],
        'CI': (model_att.conf_int().loc[T_col, 0], model_att.conf_int().loc[T_col, 1])
    }

# Function to estimate ATE and ATT using Doubly Robust Estimator
def DR_ate_att(df, X_cols, T_col, Y_col):
    X_np = df[X_cols].values  # Convert X to numpy array
    T_np = df[T_col].values  # Convert T to numpy array
    Y_np = df[Y_col].values  # Convert Y to numpy array

    # Estimate propensity scores
    ps = LogisticRegression(C=1e6, max_iter=1000).fit(X_np, T_np).predict_proba(X_np)[:, 1]
    df["ps"] = ps  # Add ps to DataFrame for consistency

    # Estimate mu0 and mu1 using a combined model
    mu_model = LinearRegression().fit(df[X_cols + [T_col]], df[Y_col])
    mu0 = mu_model.predict(df[X_cols].assign(Treated=0))
    mu1 = mu_model.predict(df[X_cols].assign(Treated=1))

    # Calculate ATE using DR formula
    dr_ate = mu1 - mu0 + (T_np / ps) * (Y_np - mu1) - ((1 - T_np) / (1 - ps)) * (Y_np - mu0)
    
    # Calculate ATT using DR formula
    dr_att = mu1 - mu0 + df[T_col] * (Y_np - mu1) - (1 - df[T_col]) * ps / (1 - ps) * (Y_np - mu0)

    return {
        'ATE_Estimate': np.mean(dr_ate),
        'ATT_Estimate': np.mean(dr_att)
    }

# Main class to perform estimation with various estimators
class pyDRReg:
    def __init__(self, df, X_cols, T_col, Y_col, method='ate', estimator='OR', n_bootstrap=50):
        self.df = df
        self.X_cols = X_cols
        self.T_col = T_col
        self.Y_col = Y_col
        self.method = method
        self.estimator = estimator.upper()  # Convert to uppercase to standardize
        self.n_bootstrap = n_bootstrap
        self.results = None
        self._run_estimation()  # Run estimation automatically upon initialization
    
    def _select_estimator(self):
        # Select the appropriate estimator function based on the method and estimator type
        if self.estimator == 'OR':
            return OR_ate if self.method == 'ate' else OR_att
        elif self.estimator == 'IPW':
            return IPW_ate if self.method == 'ate' else IPW_att
        elif self.estimator == 'DR':
            return DR_ate_att
        else:
            raise ValueError(f"Estimator '{self.estimator}' not recognized. Available estimators: 'OR', 'IPW', 'DR'.")

    def _run_estimation(self):
        estimates = []
        estimator_func = self._select_estimator()
        
        # Bootstrap process
        for _ in range(self.n_bootstrap):
            # Resample the data with replacement
            df_resampled = resample(self.df, replace=True, n_samples=len(self.df))
            
            # Calculate the estimate using the selected estimator function
            if self.estimator == 'DR':
                # DR estimator returns both ATE and ATT, select based on method
                dr_results = estimator_func(df_resampled, self.X_cols, self.T_col, self.Y_col)
                estimate = dr_results['ATE_Estimate'] if self.method == 'ate' else dr_results['ATT_Estimate']
            else:
                estimate = estimator_func(df_resampled, self.X_cols, self.T_col, self.Y_col)['Estimate']
            
            estimates.append(estimate)
        
        # Calculate standard error, confidence intervals, and p-value
        se = np.std(estimates, ddof=1)
        mean_estimate = np.mean(estimates)
        ci_lower = mean_estimate - 1.96 * se
        ci_upper = mean_estimate + 1.96 * se
        z_value = mean_estimate / se
        p_value = 2 * (1 - stats.norm.cdf(np.abs(z_value)))
        
        # Store results
        self.results = {
            'Estimator': self.estimator,
            'Method': self.method.upper(),
            'Estimate': mean_estimate,
            'SE': se,
            't-stat': z_value,
            'p-value': p_value,
            'CI': (ci_lower, ci_upper)
        }
    
    def summary(self):
        if self.results is None:
            raise ValueError("Estimation has not been completed.")
        return self.results
