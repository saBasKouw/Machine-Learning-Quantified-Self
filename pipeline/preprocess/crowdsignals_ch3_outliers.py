##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 3                                               #
#                                                            #
##############################################################

from pipeline.util.VisualizeDataset import VisualizeDataset
from pipeline.preprocess.OutlierDetection import DistributionBasedOutlierDetection
from pipeline.preprocess.OutlierDetection import DistanceBasedOutlierDetection
#from pipeline.outliers.KalmanFilters import KalmanFilters
from pipeline.preprocess.DataTransformation import PrincipalComponentAnalysis
from pipeline.preprocess.ImputationMissingValues import ImputationMissingValues
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

def reshape_data(data,labels):
    columns = ["s1","s2","s3","s4","s5","s6","s7","s8"]
    new_df = pd.DataFrame(columns=columns)
    for i in range(0,len(data.columns),8):
        new_df = new_df.append(pd.DataFrame(data[data.columns[i:i+8]].values,columns=columns),ignore_index=True)
    #new_df[["labelrock", "labelpaper", "labelscissors", "labelok"]] = 0
    new_df = new_df.assign(labelrock=0,labelpaper=0,labelscissors=0,labelok=0)
    rocks = len(labels[labels["labelrock"]==1])
    papers = len(labels[labels["labelpaper"] == 1])
    scissors = len(labels[labels["labelscissors"]==1])
    oks = len(labels[labels["labelok"]==1])
    current = 0
    new_df.iloc[current:rocks*8,new_df.columns.get_loc("labelrock")] = 1
    current = rocks*8
    new_df.iloc[current:current+papers*8,new_df.columns.get_loc("labelpaper")] = 1
    current += papers*8
    new_df.iloc[current:current+scissors*8,new_df.columns.get_loc("labelscissors")] = 1
    current += scissors*8
    new_df.iloc[current:current+oks*8,new_df.columns.get_loc("labelok")] = 1
    # for j in enumerate(labels):
    #     current_index = j*8
    #     new_df.iloc[0,new_df.columns.get_loc(labels.columns[0])] = labels[j][0]
    #     print(new_df)
    time = np.array(range(0, len(new_df)), dtype=np.float32)
    time = time * (1 / 200 / 8)
    new_df["Time (s)"] = time
    new_df["Time (s)"] = new_df["Time (s)"] + 1559822765
    new_df["Date"] = pd.to_datetime(new_df['Time (s)'],unit='s')
    new_df = new_df.set_index("Date")
    new_df = new_df.drop("Time (s)",axis=1)
    return new_df


def normality(data):
    # QQ Plot
    from numpy.random import seed
    from numpy.random import randn
    from statsmodels.graphics.gofplots import qqplot
    from matplotlib import pyplot
    # seed the random number generator
    seed(1)
    # generate univariate observations
    #data = 5 * randn(100) + 50
    # q-q plot
    qqplot(data, line='s')
    pyplot.show()

def missing_values(dataset):
    print(dataset.isna().sum())
    MisVal = ImputationMissingValues()
    for col in [c for c in dataset.columns if not 'label' in c]:
        dataset = MisVal.impute_interpolate(copy.deepcopy(dataset), col)
    print(dataset.isna().sum())
    return dataset
    #DataViz.plot_imputed_values(dataset, ['original', 'mean','median', 'interpolation'], col, imputed_mean_dataset[col],imputed_median_dataset[col], imputed_interpolation_dataset[col])
    #DataViz.plot_imputed_values(dataset, ['original', 'mean'], 'hr_watch_rate', imputed_mean_dataset[col])

def transform(dataset):
    PCA = PrincipalComponentAnalysis()
    selected_predictor_cols = [c for c in dataset.columns if (not ('label' in c)) and (not (c == 'hr_watch_rate'))]
    pc_values = PCA.determine_pc_explained_variance(dataset, selected_predictor_cols)

    # Plot the variance explained.

    plot.plot(range(1, len(selected_predictor_cols)+1), pc_values, 'b-')
    plot.xlabel('principal component number')
    plot.ylabel('explained variance')
    plot.show()

    n_pcs = 4

    dataset = PCA.apply_pca(copy.deepcopy(dataset), selected_predictor_cols, n_pcs)
    #And we visualize the result of the PC's

    DataViz.plot_dataset(dataset, ['pca_', 'label'], ['like', 'like'], ['line', 'points'])
    return dataset


# Let is create our visualization class again.
DataViz = VisualizeDataset()

# Read the result from the previous chapter, and make sture the index is of the type datetime.
dataset_path = '../'
try:
    dataset = pd.read_csv(dataset_path + 'chapter2_result.csv', index_col=0)
except IOError as e:
    print('File not found, try to run previous crowdsignals scripts first!')
    raise e

dataset.index = pd.to_datetime(dataset.index)
dataset = reshape_data(dataset[dataset.columns[:-4]],dataset[dataset.columns[-4:]])

#normality(dataset)


# Compute the number of milliseconds covered by an instance based on the first two rows
#milliseconds_per_instance = ((dataset.index[1] - dataset.index[0]).microseconds/1000)*2
# Step 1: Let us see whether we have some outliers we would prefer to remove.

# Determine the columns we want to experiment on.
outlier_columns = dataset.columns

# Create the outlier classes.
OutlierDistr = DistributionBasedOutlierDetection()
OutlierDist = DistanceBasedOutlierDetection()

#And investigate the approaches for all relevant attributes.
for col in outlier_columns:
    #normality(dataset[col])
    #DataViz.plot_dataset(dataset, [col,col], ['exact', 'exact'], ['line', 'points'])
    # And try out all different approaches. Note that we have done some optimization
    # of the parameter values for each of the approaches by visual inspection.
    # dataset_ = OutlierDistr.chauvenet(dataset, col)
    # DataViz.plot_binary_outliers(dataset_, col, col + '_outlier')
    # dataset_ = OutlierDistr.mixture_model(dataset, col)
    # DataViz.plot_dataset(dataset_, [col, col + '_mixture'], ['exact','exact'], ['line', 'points'])
    # # # This requires:
    # # # n_data_points * n_data_points * point_size =
    # # # 31839 * 31839 * 64 bits = ~8GB available memory
    # try:
    #     dataset_ = OutlierDist.simple_distance_based(dataset, [col], 'euclidean', 0.1, 0.99)
    #     DataViz.plot_binary_outliers(dataset_, col, 'simple_dist_outlier')
    # except MemoryError as e:
    #     print('Not enough memory available for simple distance-based outlier detection...')
    #     print('Skipping.')
    #
    # # KalFilter = KalmanFilters()
    # # kalman_dataset = KalFilter.apply_kalman_filter(dataset, col)
    # # DataViz.plot_imputed_values(kalman_dataset, ['original', 'kalman'], col, kalman_dataset[col])
    # # DataViz.plot_dataset(kalman_dataset, [col, col+'_kalman'], ['exact','exact'], ['line', 'line'])
    #
    try:
        dataset_ = OutlierDist.local_outlier_factor(dataset, [col], 'euclidean', 5)
        DataViz.plot_dataset(dataset_, [col, 'lof'], ['exact','exact'], ['line', 'points'])
    except MemoryError as e:
        print('Not enough memory available for lof...')
        print('Skipping.')
    #transform(dataset,col)
    #
    # # Remove all the stuff from the dataset again.
    # cols_to_remove = [col + '_outlier', col + '_mixture', 'simple_dist_outlier', 'lof']
    # for to_remove in cols_to_remove:
    #     if to_remove in dataset:
    #         del dataset[to_remove]

# We take Chauvent's criterion and apply it to all but the label data...

def remove_outliers(dataset):
    for col in [c for c in dataset.columns if not 'label' in c]:
        print('Measurement is now: ' , col)
        try:
            dataset = OutlierDist.local_outlier_factor(dataset, [col], 'euclidean', 5)
            #DataViz.plot_dataset(dataset_, [col, 'lof'], ['exact','exact'], ['line', 'points'])
        except MemoryError as e:
            print('Not enough memory available for lof...')
            print('Skipping.')
        dataset.loc[dataset['lof'] > 1.2, col] = np.nan
        del dataset["lof"]
    return dataset

def pipeline(dataset):
    dataset = remove_outliers(dataset)
    dataset = missing_values(dataset)
    dataset = transform(dataset)
    dataset.to_csv(dataset_path + 'chapter3_result_outliers.csv')
    #DataViz.plot_dataset(dataset, ["s1","s1"], ['exact', 'exact'], ['line', 'points'])

#pipeline(dataset)

