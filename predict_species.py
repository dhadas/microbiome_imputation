import glob

import pandas as pd
from sklearn.ensemble import RandomForestRegressor as RF
import preprocess_tools
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as RMSE

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.metrics import mean_squared_error as RMSE
import json
import os, sys
import glob


def build_fold_dfs(X, Y, folds_dict, fold_num):
    curr_fold = folds_dict[f'fold_{fold_num}']
    tr_idx, ts_idx = curr_fold['tr_idx'], curr_fold['ts_idx']

    X_tr = X.iloc[tr_idx, :]
    X_ts = X.iloc[ts_idx, :]

    print(Y.shape)
    Y_tr = Y.iloc[tr_idx, :]
    Y_ts = Y.iloc[ts_idx, :]

    return X_tr, X_ts, Y_tr, Y_ts

#
# def get_fold_data(path, fold_num, prefix='fold_'):
#     p = fr'*{prefix}{fold_num}*'
#     files = glob.glob(f'{path}/{p}')
#     return files

def generate_folds(X, out_path, NUM_FOLDS=5, SHOULD_SHUFFLE=False):

    folds_indexes = {}
    kf = KFold(n_splits=NUM_FOLDS, shuffle=SHOULD_SHUFFLE)

    for j, (tr_idx, ts_idx) in enumerate(kf.split(X), 1):
        folds_indexes[f'fold_{j}'] = {'tr_idx': [int(i) for i in list(tr_idx)],
                                      'ts_idx': [int(j) for j in list(ts_idx)]}


    os.makedirs(out_path, exist_ok=True)
    path_to_dict = f'{out_path}/folds_dict.json'
    with open(path_to_dict, mode='w+') as file:
        json.dump(folds_indexes, file)  # use `json.loads` to do the reverse

    return (path_to_dict, folds_indexes)


def handle_single_fold(X, Y, folds_dict, fold_num, mod):

    VERBOSE = True
    preds_dict = {}

    print(f"Starting fold {fold_num}.")
    X_tr, X_ts, Y_tr, Y_ts = build_fold_dfs(X, Y, folds_dict, fold_num)


    X_tr, X_ts = transform_X(X_tr, X_ts)
    X_ts = X_ts.dropna(how='all', axis='columns')

    # Removal of sparse features that were nullified in the transformation proccess from test and train data
    common_cols = X_tr.columns.intersection(X_ts.columns)
    X_tr, X_ts = X_tr[common_cols], X_ts[common_cols]

    mask = Y_ts != 0
    defult_y_min = Y_ts[mask].min().min() / 2

    # Iterate through all species (Columns of Y)
    for i, (otu, abundance) in enumerate(list(Y.iteritems())[:5], 1):
        # y = abundance
        y_tr, y_ts = Y_tr[otu], Y_ts[otu]
        y_tr, y_ts = transform_y(y_tr, y_ts)
        # common_y_idx = y_tr.index.intersection(y_ts.index)
        # y_tr, y_ts = y_tr[common_y_idx], y_ts[common_y_idx]

        # Removal of sparse features that were nullified in the transformation proccess from test and train data

        # The case in which y_ts is all zeros and therefore it cannot be replaced by
        # half minimal value and then log transformed
        # In this case we replace y_ts with half the minimum of y_tr
        if (y_ts.isna().sum() == y_ts.shape[0]):
            print(f"Replacing test set zeros for otu: {otu} with default minimal value from full test set")
            y_ts.fillna(defult_y_min)

        # RF MODEL
        mod.fit(X_tr, y_tr)

        preds = pd.Series(data=mod.predict(X_ts), index=y_ts.index, name=f'Preds_RF')
        #         train_preds = pd.Series(data = mod.predict(X_tr), index = y_tr.index, name=f'Preds_RF')

        preds_dict[otu] = preds

        #         compare_df = pd.DataFrame(data = [_y_test, preds], dtype=np.float64).T
        #         compare_df.rename({f'{otu}':'observed'}, inplace = True, axis = 1)
        if i % (Y.shape[1] // 5) == 0 and VERBOSE:
            print(f"Finished specie {i} / {Y.shape[1]}")

    print(f"Finished fold {fold_num}.")
    preds_df = pd.DataFrame.from_dict(preds_dict)

    rmse = preds_df.apply(lambda col: RMSE(col, y_ts, squared=False))
    return (preds_df, rmse)


def transform_X(X_tr, X_ts):
    X_tr = X_tr.copy()
    X_tr = preprocess_tools.impute_df_zeros_with_half_min(X_tr)
    X_tr = X_tr.apply(lambda x: preprocess_tools.log_transform(x, True))

    X_ts = X_ts.copy()
    X_ts = preprocess_tools.impute_df_zeros_with_half_min(X_ts)
    X_ts = X_ts.apply(lambda x: preprocess_tools.log_transform(x, True))

    return (X_tr, X_ts)


def transform_y(y_tr, y_ts):
    y_tr = y_tr.copy()
    y_tr = preprocess_tools.impute_series_zeros_with_half_min(y_tr)
    y_tr = preprocess_tools.log_transform(y_tr, True)

    y_ts = y_ts.copy()
    y_ts = preprocess_tools.impute_series_zeros_with_half_min(y_ts)
    y_ts = preprocess_tools.log_transform(y_ts, True)
    return (y_tr, y_ts)



def init_fold_job(path_to_X, path_to_Y, path_to_folds_dict, curr_fold_num):
    # Read folds dict

    X = pd.read_csv(path_to_X, index_col='# Sample / Feature')
    Y = pd.read_csv(path_to_Y, index_col='# Sample / Feature')
    mod_RF = RF(n_jobs = -1)

    with open(path_to_folds_dict, 'r', encoding="utf-8", ) as file:
        folds_dict = json.load(file)

    print(folds_dict)

    preds_for_fold, rmse_for_fold = handle_single_fold(X, Y, folds_dict, curr_fold_num, mod_RF)


    print('Finished')
    return (preds_for_fold, rmse_for_fold)



def local_pipe(X, Y, path_to_folds_dict):
    rmse_dict = {}
    preds_dict = {}
    mod_RF = RF(n_jobs=-1)
    with open(path_to_folds_dict, 'r', encoding="utf-8", ) as file:
        folds_dict = json.load(file)

    NUM_FOLDS = len(folds_dict)

    for fold_num in range(1, NUM_FOLDS+1):
            preds_for_fold, rmse_for_fold = handle_single_fold(X, Y, folds_dict, fold_num, mod_RF)
            rmse_dict[f'fold_{fold_num}'] = rmse_for_fold
            preds_dict[f'fold_{fold_num}'] = preds_for_fold

    rmse_df = pd.DataFrame.from_dict(rmse_dict)
    preds_df = pd.concat(preds_dict.values())


    return rmse_df, preds_df

if __name__ == '__main__':
    print(sys.argv)
    if len(sys.argv) > 3:
        path_to_X, path_to_Y, path_to_folds_dict, curr_fold_num = sys.argv[1:]
        preds_fold, rmse_fold = init_fold_job(path_to_X, path_to_Y, path_to_folds_dict, curr_fold_num)
        preds_fold.to_csv(f'preds_fold_{curr_fold_num}.csv')
        rmse_fold.to_csv(f'rmse_fold_{curr_fold_num}.csv')

    else:
        pass





# data_path = '/Users/d_private/OneDrive - mail.tau.ac.il/Lab/data/FRANZOSA_IBD_2019/PARSED_DATA/basic_process_reduce_sparse'
# path_to_Y = f'{data_path}/stgndf_PRISM_reduced_sparse.csv'
# path_to_X = f'{data_path}/mbdf_PRISM_reduced_sparse.csv'
# path_to_dict = '/Users/d_private/OneDrive - mail.tau.ac.il/Lab/data/FRANZOSA_IBD_2019/PARSED_DATA/folds_180521/folds_dict.json'
# preds_for_fold, rmse_for_fold = init_fold_job(path_to_X, path_to_Y, path_to_dict, 1)

# out_ =  f'{path}/PARSED_DATA/folds_180521'
# print(out_)
# data_path = '/Users/d_private/PycharmProjects/mat_imputation_demo/resources/demo_data_for_funcs.csv'
# X = pd.read_csv(data_path, sep='\t', index_col=0)
# generate_folds(X, out_, 3, False)
#
# Y = pd.read_csv(f'{data_path}/stgndf_PRISM_reduced_sparse.csv', index_col = '# Sample / Feature')
# X = pd.read_csv(f'{data_path}/mbdf_PRISM_reduced_sparse.csv', index_col = '# Sample / Feature')
# # rmse_df, preds_df = local_pipe(X, Y, path_to_dict)
#
# X = pd.read_csv(path_to_X, index_col='# Sample / Feature')
# Y = pd.read_csv(path_to_Y, index_col='# Sample / Feature')
# print(X)