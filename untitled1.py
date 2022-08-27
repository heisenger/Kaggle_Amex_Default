import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer

def null_imputation(df: pd.DataFrame, nn=2) -> pd.DataFrame:
    imputer = KNNImputer(n_neighbors=2)
    imputed_df = imputer.fit_transform(df)
    return imputed_df

# Pad each customer to have 13 rows

def padding_fn(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df[['customer_ID']].groupby('customer_ID').customer_ID.agg('count')
    more = np.array([], dtype='int64')
    
    for j in range(1, 13):
        i = tmp.loc[tmp==j].index.values
        more = np.concatenate([more, np.repeat(i, 13-j)])