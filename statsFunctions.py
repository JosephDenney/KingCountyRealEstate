import statsmodels.api as sm
import pandas as pd
import scipy.stats as scs

# Used function definitions

def welch_t(a, b): # t-stat calculationg for 2 samples with different means 
    numerator = a.mean() - b.mean()
    denom = np.sqrt(a.var(ddof=1)/a.size + b.var(ddof=1)/b.size)
    """ Calculate Welch's t-statistic for two samples. """
    return np.abs(numerator/denom)

def welch_df(a, b): # calculating this allows you to work with other variables to get a p-value!
    """ Calculate the effective degrees of freedom for two samples. """
    s1 = a.var(ddof=1) 
    s2 = b.var(ddof=1)
    n1 = a.size
    n2 = b.size
    
    numerator = (s1/n1 + s2/n2)**2
    denominator = (s1/ n1)**2/(n1 - 1) + (s2/ n2)**2/(n2 - 1)
    
    return numerator/denominator

def p_value(a, b, two_sided=False):
    t = welch_t(a, b)
    df = welch_df(a, b)
    
    p = 1-stats.t.cdf(np.abs(t), round(df))
    
    if two_sided:
        return 2*p
    else:
        return p
    
def Cohen_d(group1, group2):
    # Compute Cohen's d
    # group1: Series or NumPy array
    # group2: Series or NumPy array
    # returns a floating point number 

    diff = group1.mean() - group2.mean()

    n1, n2 = len(group1), len(group2)
    var1 = group1.var()
    var2 = group2.var()

    # Calculate the pooled threshold as shown earlier
    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    
    # Calculate Cohen's d statistic
    d = diff / np.sqrt(pooled_var)
    return abs(d)

# Create a function to build a statsmodels ols model
def build_sm_ols(df, features_to_use, target, add_constant=False, show_summary=True):
    X = df[features_to_use]
    if add_constant:
        X = sm.add_constant(X)
    y = df[target]
    ols = sm.OLS(y, X).fit()
    if show_summary:
        print(ols.summary())
    return ols

# create a function to check the validity of your model
# it should measure multicollinearity using vif of features
# it should test the normality of your residuals 
# it should plot residuals against an xaxis to check for homoskedacity
# it should implement the Breusch Pagan Test for Heteroskedasticity
##  Ho: the variance is constant            
##  Ha: the variance is not constant

# assumptions of ols
# residuals are normally distributed
def check_residuals_normal(ols):
    residuals = ols.resid
    t, p = scs.shapiro(residuals)
    if p <= 0.05:
        return False
    return True


# residuals are homoskedasticitous
def check_residuals_homoskedasticity(ols):
    import statsmodels.stats.api as sms
    resid = ols.resid
    exog = ols.model.exog
    lg, p, f, fp = sms.het_breuschpagan(resid=resid, exog_het=exog)
    if p >= 0.05:
        return True
    return False


def check_vif(df, features_to_use, target_feature):
    ols = build_sm_ols(df=df, features_to_use=features_to_use, target=target_feature, show_summary=False)
    r2 = ols.rsquared
    return 1 / (1 - r2)
    
    
# no multicollinearity in our feature space
def check_vif_feature_space(df, features_to_use, vif_threshold=3.0):
    all_good_vif = True
    for feature in features_to_use:
        target_feature = feature
        _features_to_use = [f for f in features_to_use if f!=target_feature]
        vif = check_vif(df=df, features_to_use=_features_to_use, target_feature=target_feature)
        if vif >= vif_threshold:
            print(f"{target_feature} surpassed threshold with vif={vif}")
            all_good_vif = False
    return all_good_vif
        
        
def check_model(df, 
                features_to_use, 
                target_col, 
                add_constant=False, 
                show_summary=False, 
                vif_threshold=3.0):
    has_multicollinearity = check_vif_feature_space(df=df, 
                                                    features_to_use=features_to_use, 
                                                    vif_threshold=vif_threshold)
    if not has_multicollinearity:
        print("Model contains multicollinear features")
    
    # build model 
    ols = build_sm_ols(df=df, features_to_use=features_to_use, 
                       target=target_col, add_constant=add_constant, 
                       show_summary=show_summary)
    
    # check residuals
    resids_are_norm = check_residuals_normal(ols)
    resids_are_homo = check_residuals_homoskedasticity(ols)
    
    if not resids_are_norm or not resids_are_homo:
        print("Residuals failed test/tests")
    return ols

 
# calc and return AdjR^2 and VIF scores
    
def adjusted_r_squared(rsquared, num_obs, p):

    ''' calc_adjr_and_VIF calculates adjusted r squared and VIF for given r squared, but must
    be supplied with number of data points in sample as well as number of independent regressors

    Parameters: 
    rquared (float): should be an r squared value from an ols regression to be adjusted
    num_obs (int): number of data points in your sample
    p (int): number of independent regressors excluding the constant
    
    Returns:
    adjusted r squared and a VIF score for the input parameters
    '''
    adjusted_r = (1 - (1-rsquared) * ((num_obs-1)/(num_obs-p-1)))
    vif = 1/(1-adjusted_r)
    print('Adjusted R^2 is: ',adjusted_r)
    print('VIF score is: ',vif)
    return vif, adjusted_r
 

    