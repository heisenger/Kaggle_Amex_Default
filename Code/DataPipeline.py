import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

import random

def null_filtering(df, null_threshold=0.8):
    ignored_columns = df.columns[df.isnull().sum() / len(df) > null_threshold]
    return ignored_columns

def importance_filtering(df, unimportant_factors=None, rank_factor=None, exclude_bottom=100):
    unimportant_factors = unimportant_factors.sort_values(by=[rank_factor], ascending=False)
    return df.drop(list(unimportant_factors.values[-exclude_bottom:]), axis=1)

def pre_process_df(df):
    for bcol in [f'B_{i}' for i in [11,14,17]]+['D_39','D_131']+[f'S_{i}' for i in [16,23]]:
        for pcol in ['P_2','P_3']:
            if bcol in df.columns:
                df[f'{bcol}-{pcol}'] = df[bcol] - df[pcol]
    return df
                   

def agglomeration_function (df, group_id='customer_ID', num_features=[], cat_features=[], ignore=['target'], apply_pca=False, train=False):
    # new_df = df.groupby("customer_ID").tail(1).reset_index(drop=True)
    # new_df = df.groupby("customer_ID").mean().reset_index(drop=True)
    
    print(f'input shape: {df.shape}')
    # df = df.fillna(0)
    # Add numerical features
    
    # df['day'] = df.S_2.day
    # df['month'] = df.S_2.month
    
    if len(num_features)>0:
        
        new_df_num = df.groupby("customer_ID")[[i for i in num_features]].agg(['first', 'mean', 'median', 'std', 'min', 'max', 'last']).reset_index(drop=True)
        new_df_num.columns = ['_'.join(x) for x in new_df_num.columns]
        # new_df_num = new_df_num.fillna(0)
        # print(new_df_num.isnull
        
        for col in new_df_num:
            for lag_col in ['mean', 'median', 'std', 'min', 'max', 'last']:
                if 'last' in col and col.replace('last', lag_col) in new_df_num:
                    new_df_num[col + '_lag_sub'] = new_df_num[col] - new_df_num[col.replace('last', lag_col)]
                    new_df_num[col + '_lag_div'] = new_df_num[col] / new_df_num[col.replace('last', lag_col)]
        
        # Generate some random combination columns
        for col in new_df_num:
            col_2 = random.choice(list(new_df_num.columns))
            for lag_col in ['mean', 'median', 'std', 'min', 'max', 'last']:
                if 'last' in col and col.replace('last', lag_col) in new_df_num:
                    new_df_num[col + 'rand_lag_sub'] = new_df_num[col_2] - new_df_num[col.replace('last', lag_col)]
                    new_df_num[col + 'rand_lag_div'] = new_df_num[col_2] / new_df_num[col.replace('last', lag_col)]

        if apply_pca:
            pcr = make_pipeline(StandardScaler(), PCA(n_components=100))
            pcr.fit(new_df_num)
            pca = pcr.named_steps['pca']
            new_df_num = pca.transform(new_df_num)
            new_df_num = pd.DataFrame(new_df_num)

    # Add categorical features
    if len(cat_features)>0:
        new_df_cat = df.groupby("customer_ID")[[i for i in cat_features]].agg(['count', 'last', 'nunique']).reset_index(drop=True)
        new_df_cat.columns = ['_'.join(x) for x in new_df_cat.columns]
    
    if ignore != None:
        new_df = pd.concat([new_df_num, new_df_cat, df.groupby("customer_ID")[ignore].tail(1).reset_index(drop=True)], axis=1)
    else:
        new_df = pd.concat([new_df_num, new_df_cat], axis=1)
    print(f'output shape: {new_df.shape}')
    
    return new_df


# def miss_classifications(y





    