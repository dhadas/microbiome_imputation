import pandas as pd
import matplotlib.pyplot as plt



# Describe the sparsity of the 16s data, for each smaple plot the percent of zeros in it.

def count_zeros(row: pd.Series): return sum(row == 0)

def percent_zeros(row: pd.Series): return sum(row == 0)/len(row)

def get_spartsity_plot(df: pd.DataFrame, _ax = 0, should_plot=True):
    '''
    :param df: df to describe
    :param _ax: Axis to get sparsity measures for. 0 means we get sparsity per feature
    :param should_plot: weather plot locally or return a vector of sparsity fraction.
    :return:
    '''
    temp = df.copy()

    if _ax == 0:
        res = temp.apply(lambda col: (col == 0).sum()/len(col), axis=_ax)

    else:
        res = temp.apply(lambda  row: (row==0).sum()/len(row), axis=_ax)

    if should_plot:
        res.plot(kind='hist')
        plt.show()
    return res


if __name__ == "__main__":
    data_path = '/Users/d_private/PycharmProjects/mat_imputation_demo/resources/demo_data_for_funcs.csv'
    test = pd.read_csv(data_path, sep='\t', index_col=0)
    print(test)
    zeros_per_feature = get_spartsity_plot(test)
    print(zeros_per_feature)
    # print(zeros_per_feature)
    # zeros_per_sample = get_spartsity_plot(test, 1)

