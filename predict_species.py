import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.model_selection import KFold
import preprocess_tools
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.metrics import mean_squared_error as RMSE
import json
import sys
import os
from Transformer import Transformer
import ignore_Y_zeros_transformer
import logging



# Create a custom logger

# logging.basicConfig(filename='predict_species.log', filemode='wb', level='INFO')

def build_fold_dfs(X, Y, folds_dict, fold_num):
    curr_fold = folds_dict[f'fold_{fold_num}']
    tr_idx, ts_idx = curr_fold['tr_idx'], curr_fold['ts_idx']

    X_tr = X.iloc[tr_idx, :]
    X_ts = X.iloc[ts_idx, :]

    print(Y.shape)
    Y_tr = Y.iloc[tr_idx, :]
    Y_ts = Y.iloc[ts_idx, :]

    return X_tr, X_ts, Y_tr, Y_ts

def handle_single_fold(X, Y, folds_dict, fold_num, mod, transformer = None):
    logger = logging.getLogger('predict_species')
    f_handler = logging.FileHandler(filename='predict_species.log', mode='w')
    f_handler.setLevel(logging.WARNING)
    logger.addHandler(f_handler)


    # Set default transformer
    if transformer == None:
        transformer = Transformer()

    VERBOSE = True
    preds_dict = {}
    rmse_dict = {}

    print(logger)
    print(f"Starting fold {fold_num}.")
    logger.info(f"Starting fold {fold_num}.")
    X_tr, X_ts, Y_tr, Y_ts = build_fold_dfs(X, Y, folds_dict, fold_num)
    ts_idx = Y_ts.index

    # Transform X
    X_tr, X_ts = transformer.transform_X(X_tr, X_ts)

    mask = Y_ts != 0
    defult_y_min = Y_ts[mask].min().min() / 2

    # Iterate through all species (Columns of Y)
    for i, (otu, abundance) in enumerate(list(Y.iteritems())[:3], 1):
        # y = abundance
        print(otu)
        y_tr, y_ts = Y_tr[otu], Y_ts[otu]
        y_tr, y_ts = transformer.transform_y(y_tr, y_ts)

        if y_ts is None:
            logging.info(f"{fold_num}:{otu}: y_ts nullified post transformation - skipping.")
            preds = pd.Series(data=np.nan, index=ts_idx, name=f'Preds_RF')
            preds_dict[otu] = preds
            continue


        # Remove nullified samples
        common_train_samples = X_tr.index.intersection(y_tr.index)
        curr_X_tr, y_tr = X_tr.loc[common_train_samples, :], y_tr[common_train_samples]

        # Remove nullified samples from test
        common_test_samples = X_ts.index.intersection(y_ts.index)
        curr_X_ts, y_ts = X_ts.loc[common_test_samples, :], y_ts[common_test_samples]
        curr_X_ts = curr_X_ts.dropna(how='all', axis=1)

        # Removal of sparse features that were nullified in the transformation proccess from test and train data
        common_cols = curr_X_tr.columns.intersection(curr_X_ts.columns)
        curr_X_tr, curr_X_ts = curr_X_tr[common_cols], curr_X_ts[common_cols]

        # Train model
        mod.fit(curr_X_tr, y_tr)

        preds = pd.Series(np.nan, index=ts_idx, )
        preds[y_ts.index] = mod.predict(curr_X_ts)

        logger.info(f"{fold_num}:{otu}: Done. size of train: {y_tr.shape[0]}, size of test: {y_ts.shape[0]}")


        preds_dict[otu] = preds
        rmse_dict[otu] = RMSE(preds.dropna(), y_ts, squared=False)

        if i % (Y.shape[1] // 5) == 0 and VERBOSE:
            print(f"Finished specie {i} / {Y.shape[1]}")



    preds_df = pd.DataFrame.from_dict(preds_dict)
    rmse_df = pd.Series(rmse_dict, name=f'RMSE_fold_{fold_num}')

    print(f"Finished fold {fold_num}.")

    return preds_df, rmse_df


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



def init_fold_job(path_to_X, path_to_Y, path_to_folds_dict, curr_fold_num, transformer):
    # Read folds dict
    mod_RF = RF(n_jobs = -1)

    X = pd.read_csv(path_to_X, index_col='# Sample / Feature')
    Y = pd.read_csv(path_to_Y, index_col='# Sample / Feature')

    folds_dict = read_fold_dict(path_to_folds_dict)

    preds_for_fold, rmse_for_fold = handle_single_fold(X, Y, folds_dict, curr_fold_num, mod_RF, transformer)

    print('Finished')
    return (preds_for_fold, rmse_for_fold)

def read_fold_dict(path_to_folds_dict):
    with open(path_to_folds_dict, 'r', encoding="utf-8", ) as file:
        folds_dict = json.load(file)

    return folds_dict

def local_pipe(X, Y, path_to_folds_dict, transformer: Transformer=None):
    rmse_dict = {}
    preds_dict = {}
    mod_RF = RF(n_jobs=-1)
    with open(path_to_folds_dict, 'r', encoding="utf-8", ) as file:
        folds_dict = json.load(file)

    NUM_FOLDS = len(folds_dict)

    for fold_num in range(1, NUM_FOLDS+1):
            preds_for_fold, rmse_for_fold = handle_single_fold(X, Y, folds_dict, fold_num, mod_RF, transformer)
            rmse_dict[f'fold_{fold_num}'] = rmse_for_fold
            preds_dict[f'fold_{fold_num}'] = preds_for_fold

    rmse_df = pd.DataFrame.from_dict(rmse_dict)


    return preds_dict, rmse_df

if __name__ == '__main__':

    if len(sys.argv) > 3:
        action = sys
        path_to_X, path_to_Y, path_to_folds_dict, curr_fold_num,  = sys.argv[1:]
        preds_fold, rmse_fold = init_fold_job(path_to_X, path_to_Y, path_to_folds_dict, curr_fold_num)
        preds_fold.to_csv(f'preds_fold_{curr_fold_num}.csv')
        rmse_fold.to_csv(f'rmse_fold_{curr_fold_num}.csv')

    else:
        pass


def generate_folds(X, out_path, NUM_FOLDS=5, SHOULD_SHUFFLE=False):
    print(X, out_path, NUM_FOLDS, SHOULD_SHUFFLE)
    folds_dict = {}
    kf = KFold(n_splits=NUM_FOLDS, shuffle=SHOULD_SHUFFLE)

    for j, (tr_idx, ts_idx) in enumerate(kf.split(X), 1):
        folds_dict[f'fold_{j}'] = {'tr_idx': [int(i) for i in list(tr_idx)],
                                      'ts_idx': [int(j) for j in list(ts_idx)]}


    os.makedirs(out_path, exist_ok=True)
    path_to_dict = f'{out_path}/folds_dict.json'
    with open(path_to_dict, mode='w+') as file:
        json.dump(folds_dict, file)  # use `json.loads` to do the reverse

    return (path_to_dict, folds_dict)


# if __name__ == '__main__':
#     data_path = '/Users/d_private/OneDrive - mail.tau.ac.il/Lab/data/FRANZOSA_IBD_2019/PARSED_DATA/basic_process_reduce_sparse'
#     path_to_Y = f'{data_path}/stgndf_PRISM_reduced_sparse.csv'
#     path_to_X = f'{data_path}/mbdf_PRISM_reduced_sparse.csv'
#     path_to_dict = '/Users/d_private/OneDrive - mail.tau.ac.il/Lab/data/FRANZOSA_IBD_2019/PARSED_DATA/folds_240521/folds_dict.json'
#     K = 20
#     ignore_zeros_transformer = ignore_Y_zeros_transformer.get_transformer()
#     # preds_for_fold, rmse_for_fold = init_fold_job(path_to_X, path_to_Y, path_to_dict, 1)
#
#     # out_ =  f'{path}/PARSED_DATA/folds_180521'
#     # print(out_)
#     # data_path = '/Users/d_private/PycharmProjects/mat_imputation_demo/resources/demo_data_for_funcs.csv'
#     # X = pd.read_csv(data_path, sep='\t', index_col=0)
#     # generate_folds(X, out_, 3, False)
#     #
#     # Y = pd.read_csv(f'{data_path}/stgndf_PRISM_reduced_sparse.csv', index_col = '# Sample / Feature')
#     # X = pd.read_csv(f'{data_path}/mbdf_PRISM_reduced_sparse.csv', index_col = '# Sample / Feature')
#     # # rmse_df, preds_df = local_pipe(X, Y, path_to_dict)
#     #
#     X = pd.read_csv(path_to_X, index_col='# Sample / Feature')
#     Y = pd.read_csv(path_to_Y, index_col='# Sample / Feature')
#     folds_dict = read_fold_dict(path_to_dict)
#     mod = RF(n_jobs = -1)
#     # print(X)
#
#     preds, rmse = handle_single_fold(X, Y, folds_dict, 1, mod, ignore_zeros_transformer)