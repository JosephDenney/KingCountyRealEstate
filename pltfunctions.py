import warnings
from importlib import reload

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np 

import scipy.stats as scs
import statsmodels.api as sm
import statsmodels.formula.api as sms
from statsFunctions import check_model as sf

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression

import matplotlib.pyplot as plt
import seaborn as sns

# print feature lm plot
def feat_plots(feature,target,df):
    print(feature)
    
    plt.title("{} histogram".format(feature))
    sns.distplot(df[feature])
    plt.show()
    
    sns.lmplot(x=feature, y=target, data=df, line_kws={'color': 'red'})
    plt.title("{} vs {}".format(target, feature))
    plt.show()
    
    pass
    
# Function that will create kde and scatter plots
def hist_kde_plots(feature,target,df):

    ''' hist_kde_plots creates a kde and scatter plot with a line of best fit for a given set of data
    
    this function is designed to be used with an iterable so that multiple kde and scatter plots can
    be created at once

    Parameters: 
    feature (float/int): specific feature being graphed
    target (float/int): the target (y) feature of the regression
    df (pandas df): dataframe being referenced
    
    Returns:
    graphical representations of the individual feature with the target
    '''
    
    print(feature)
    #kde plot
    df[feature].plot.kde(label=feature)
    plt.title("{} Kde plot".format(feature))
    plt.legend()
    plt.show()
    
    #Scatter Plot using sns lmplot
    sns.lmplot(x=feature, y=target, data=df, line_kws={'color': 'red'})
    plt.title("{} vs {}".format(target, feature))
    plt.show()
    
    pass

