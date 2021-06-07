import predict_species, Transformer, ignore_Y_zeros_transformer
import sys

if __name__ == '__main__':

    print("In distributed_RF")
    if len(sys.argv) > 3:
        action = sys
        path_to_X, path_to_Y, path_to_folds_dict, curr_fold_num,  = sys.argv[1:]
        print(sys.argv[1:])
        _transformer = ignore_Y_zeros_transformer.get_transformer()
        preds_fold, rmse_fold = predict_species.init_fold_job(path_to_X, path_to_Y,
                                                              path_to_folds_dict, curr_fold_num, _transformer)

        preds_fold.to_csv(f'preds_fold_{curr_fold_num}.csv')
        rmse_fold.to_csv(f'rmse_fold_{curr_fold_num}.csv')

    else:
        print("In distributed_RF, num of args <= 3")
        pass