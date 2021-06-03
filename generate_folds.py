from sklearn.model_selection import KFold
import os, sys
import json


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

if __name__ == '__main__':
    if len(sys.argv) > 3:
        sys.argv[3] = int(sys.argv[3])

    if len(sys.argv) > 4:
        sys.argv[4] = bool(sys.argv[4])

    generate_folds(*sys.argv[1:])