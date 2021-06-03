import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
import numpy as np

def reduce_sparsity_below_thr(df: pd.DataFrame, SPARSITY_THRESHOLD=0.85, _ax=0):
    '''

    :param df: dataframe to transform
    :param SPARSITY_THRESHOLD: samples/features (depends on _ax) with zero percentage higher than
        SPARSITY_THRESHOLD will be removed from df.
    :param _ax: along which axis we should determine sparsity and remove samples according to axis.
    if _ax is 0 (defualt) we determine sparsity by row and remove rows that are highly sparse.
    :return: transformed df
    '''

    THR = SPARSITY_THRESHOLD * df.shape[_ax]

    # Create a mask and filter out species if fraction of zeros per specie > SPARSITY_THRESHOLD
    mask = df.apply(lambda x: (x == 0).sum() < THR, axis=_ax)

    if _ax == 1:
        before = df.shape[0]
        res = df.loc[mask,:]
        after = res.shape[0]
        print(f'Reduced rows sparsity below {SPARSITY_THRESHOLD}, by removing {before-after} rows.')

    else:
        before = df.shape[1]
        res = df.loc[:, mask]
        after = res.shape[1]
        print(f'Reduced columns sparsity below {SPARSITY_THRESHOLD}, by removing {before-after} columns.')


    return res


def impute_df_zeros_with_half_min(df: pd.DataFrame, _ax = 0):
    '''
    Fill zeros with half of them minimum either column wise or rows wise.
    :param df: df to impute
    :param _ax: along which axis we impute.
    0 means we impute across index/row, this means each zero is replaced with half the minimum value of it's *column*.
    1 means we impute across columns. this means each zero is replaced with half the minimum value of it's *row*.
    :return: transformed df
    '''
    res = df.apply(lambda s: impute_series_zeros_with_half_min(s), axis=_ax)
    return res

def impute_series_zeros_with_half_min(s: pd.Series):
    # Find minimal non-zero abundance per sample
    min_abundance = s.replace(0, np.nan).dropna().min()

    #  replace all zeros with half of the min abundance
    sample = s.replace(0, min_abundance / 2)

    return sample


def log_transform(sample: pd.Series, MINUS_FLAG=True):
    coef = (-1) ** (MINUS_FLAG)
    sample = sample.apply(lambda val: coef * np.log10(val))
    return sample


def center_on_median(feature: pd.Series):
    median = feature.median()
    feature = feature.apply(lambda val: val - median)
    return feature


# def transform_metab_data(df: pd.DataFrame):
#     df = df.apply(lambda sample: remove_zeros(sample), axis=1)
#
#     # In the spirit of 'Statistical Workflow for Feature Selection in Human Metabolomics Data' we log-transform
#     # only the data the benefits from the transformation in terms of skewness
#     prior = df.copy()
#     post = df.apply(lambda sample: log_transform(sample, False), axis=1)
#
#     # If post transfromation skew decreased we replace columns with the transformed version of the values
#     mask = abs(post) < abs(prior)
#     df[mask] = df[mask].apply(lambda sample: log_transform(sample, False), axis=1)
#     #     df = df.apply(lambda feature: center_on_median(feature), axis = 0)
#
#     return df
#
#
# def transform_species_data(df: pd.DataFrame):
#     df = df.apply(lambda sample: remove_zeros(sample), axis=1)
#     df = df.apply(lambda sample: log_transform(sample, True), axis=1)
#     return df


if __name__ == "__main__":
    data_path = '/Users/d_private/PycharmProjects/mat_imputation_demo/resources/demo_data_for_funcs.csv'
    test = pd.read_csv(data_path, sep='\t', index_col=0)

    print("Sparsity filtering test")
    print(test)
    print(reduce_sparsity_below_thr(test))
    print(reduce_sparsity_below_thr(test, _ax=1))

    print("Imputation test")
    print(test)
    print(impute_df_zeros_with_half_min(test))
    print(impute_df_zeros_with_half_min(test, _ax=1))

