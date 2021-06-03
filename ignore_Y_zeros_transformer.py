from Transformer import Transformer
import preprocess_tools

def transform_X(X_tr, X_ts):
    X_tr = X_tr.copy()
    X_tr = preprocess_tools.impute_df_zeros_with_half_min(X_tr)
    X_tr = X_tr.apply(lambda x: preprocess_tools.log_transform(x, True))

    X_ts = X_ts.copy()
    X_ts = preprocess_tools.impute_df_zeros_with_half_min(X_ts)
    X_ts = X_ts.apply(lambda x: preprocess_tools.log_transform(x, True))

    return (X_tr, X_ts)

def transform_y_ignore_zeros(y_tr, y_ts):
    '''
    :param y_tr: pd.Series representing training data of dependent variable
    :param y_ts: pd.Series representing test data of dependent variable
    :return: transformed y_tr, y_ts
    '''
    res_y_tr = y_tr[y_tr!=0] # Remove zeros from training data
    res_y_ts = y_ts[y_ts!=0]

    # Removal of sparse features that were nullified in the transformation proccess from test and train data

    # The case in which y_ts is all zeros and therefore it cannot be replaced by
    # half minimal value and then log transformed
    # In this case we replace y_ts with half the minimum of y_tr
    if (res_y_ts.shape[0] == 0):
        return (res_y_tr, None)

    return (res_y_tr, res_y_ts)


def get_transformer():
    return Transformer(transform_X, transform_y_ignore_zeros)

# print(Transformer(transform_X, transform_y_ignore_zeros))
print(get_transformer())

