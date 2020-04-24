"""
This modules contains utility functions for data manipulation and plotting of
results and data
"""

import pandas as pd


def df_normalizer(dataframe):
    
    df = dataframe
    
    normed_df = pd.DataFrame()

    df_norm_key = {}

    for colname, coldata in df.iteritems():
        max_val = coldata.max()
        min_val = coldata.min()

        df_norm_key[colname] = [min_val, max_val]

        normed_col = (coldata - min_val) / (max_val - min_val)
        normed_df[colname] = normed_col
        
    return normed_df, df_norm_key 


def df_denormalize(normed_df, norm_key):
    
    denormed_df = pd.DataFrame()
    
    for colname, coldata in normed_df.iteritems():
        mn = norm_key[colname][0]
        mx = norm_key[colname][1]
        
        denormed_col = (coldata * (mx - mn)) + mn
        
        denormed_df[colname] = denormed_col
        
    return denormed_df