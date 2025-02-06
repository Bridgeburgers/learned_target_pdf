'''
utility functions
'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

class Other_Imputer:
    
    def __init__(self, cols, percentile=0.98, other_val='OTHER', impute_na=True):
        '''
        An encoder that converts the bottom (1-percentile) percentile of values
        to 'OTHER' in a categorical variable. The fit and transform methods
        take a pandas dataframe with the columns present
        Parameters
        ----------
        cols : list[str]
            List of columns to be 'other-encoded'.
        percentile : float, optional
            Top percentile of values to be preserved. The default is 0.98.
        other_val : str, optional
            the value to replace bottom (1-percentile) percent of values. 
            The default is 'OTHER'.
        impute_na : bool, optional
            If True, replaces all null values with other_val. 
            The default is True.

        Returns
        -------
        None.

        '''
        
        self.cols = cols
        self.percentile = percentile
        self.other_val = other_val
        self.known_vals = {}
        self.impute_na = impute_na
        
    def fit(self, df):
        assert all([x in df for x in self.cols])
        self.known_vals = {}
        for col in self.cols:
            cum_percent = df[col].value_counts(dropna=False, normalize=True).cumsum()
            self.known_vals[col] = cum_percent.index[cum_percent <= self.percentile]
        
    def transform(self, df):
        assert all([x in df for x in self.cols])
        df = df.copy()
        for col, vals in self.known_vals.items():
            df.loc[~df[col].isin(vals), col] = self.other_val  
            if self.impute_na:
                df.loc[df[col].isnull(), col] = self.other_val
        return df
    
    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
    
    
    
class Other_Ordinal_Encoder:
    
    def __init__(self, cols, other_val='OTHER'):
        '''
        this is a label encoder that can handle unseen values by converting 
        them to other_val. However, it assumes that other_val is already
        present in each column. Therefore, Other_Imputer class should be
        used on a df before it's passed to this. This is generally used for
        an embedding layer

        Parameters
        ----------
        cols : list[str]
            columns to be label encoded.
        other_val : str, optional
            the assumed value of "OTHER" values. The default is 'OTHER'.

        Returns
        -------
        None.

        '''
                        
        self.cols = cols
        self.other_val = other_val
        self.enc = OrdinalEncoder()
        
    def fit(self, df):
        assert all([x in df for x in self.cols])
        for col in self.cols:
            assert df[col].isnull().sum() == 0
        self.enc.fit(df[self.cols])
        
    def transform(self, df):
        assert all([x in df for x in self.cols])
        for col in self.cols:
            assert df[col].isnull().sum() == 0
        df = df.copy()
        for col, vals in zip(self.enc.feature_names_in_, self.enc.categories_):
            df.loc[~df[col].isin(vals), col] = self.other_val
        df.loc[:, self.cols] = self.enc.transform(df[self.cols])
        return df
    
    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
 
    
def gamma_unbiased_estimator(x):
    '''
    fits data x to a gamma distribution and outputs unbiased estimators
    alpha_bar and theta_bar
    '''
    if not isinstance(x, np.ndarray):
        x = np.array(x)
        
    N = len(x)
    s_x = x.sum()
    s_lnx = np.log(x).sum()
    s_xlnx = (x * np.log(x)).sum()
    
    alpha_bar = N * s_x / (N * s_xlnx - s_x * s_lnx)
    theta_bar = (N * s_xlnx - s_x * s_lnx) / N**2
    
    alpha_bar_unbiased = alpha_bar - (
        3 * alpha_bar -\
        2/3 * alpha_bar / (1 + alpha_bar) -\
        4/5 * alpha_bar / (1 + alpha_bar)**2
    ) / N
        
    theta_bar_unbiased = N * theta_bar / (N-1)
    
    return (alpha_bar_unbiased, theta_bar_unbiased)


class Gamma_Bin_Creator:
    
    def __init__(self, num_bins=10, fit_all_higher_bins=True, quantile=True):
        '''
        class that bins data and fits gamma parameters to each bin. The 
        transform method simply assigns bins to each sample, but does not
        modify the saved gamma parameters for each bin. This is useful for
        importance sampling on data with very long tailed distributions, to
        be used in conjunction with Binned_Gamma_Learned_Pdf_MLP layer and
        Learned_PDF_Importance_Sample loss function

        Parameters
        ----------
        num_bins : int, optional
            Number of quantile bins to fit separate gamma distributions. The
            default is 10.
        fit_all_higher_bins : bool, optional
            If True, each bin's gamma parameter will be fit on all the data
            whose bin id is >= the current bin, not just the current bin. 
            The default is True.
        quantile: bool, optional
            If True, bins are divided based on quantiles. If False, they
            are evenly divided

        Returns
        -------
        None.

        '''
        
        self.num_bins = num_bins
        self.fit_all_higher_bins = fit_all_higher_bins
        self.quantiles = np.linspace(0,1, self.num_bins + 1)
        self.quantile = quantile
        
    def assign_bins(self, x):
        assert hasattr(self, 'bin_ranges')
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        res = np.zeros(len(x))
        for i,bin_range in enumerate(self.bin_ranges[1:]):
            res[(bin_range[0] <= x) & (x < bin_range[1])] = i+1
        return res
        
    def fit_transform(self, x):
        if self.quantile:
            bin_boundaries = np.quantile(x, self.quantiles)
        else:
            bin_boundaries = np.linspace(min(x), max(x), self.num_bins + 1)
        bin_boundaries[-1] = np.inf
        bin_ranges = [(lower, upper) for lower, upper in zip(
            bin_boundaries[0:-1], bin_boundaries[1:])]
        
        self.bin_ranges = bin_ranges
        bins = self.assign_bins(x)
        
        self.binned_gamma_params = []
        for bin in range(self.num_bins):
            if self.fit_all_higher_bins:
                x_sub = x[bins >= bin]
            else:
                x_sub = x[bins == bin]
            self.binned_gamma_params.append(gamma_unbiased_estimator(x_sub))
            
        return bins
            
    def transform(self, x):
        return self.assign_bins(x)
    
    def fit(self, x):
        _ = self.fit_transform(x)
        
    def get_binned_gamma_params(self):
        try:
            return self.binned_gamma_params
        except AttributeError as e:
            raise AttributeError(e + 'You must run \'fit\' method first.')