##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 3                                               #
#                                                            #
##############################################################
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC

# Simple class to impute missing values of a single columns.
class ImputationMissingValues:

    # Impute the mean values in case if missing data.
    def impute_mean(self, dataset, col):
        dataset[col] = dataset[col].fillna(dataset[col].mean())
        return dataset

    # Impute the median values in case if missing data.
    def impute_median(self, dataset, col):
        dataset[col] = dataset[col].fillna(dataset[col].median())
        return dataset

    # Interpolate the dataset based on previous/next values..
    def impute_interpolate(self, dataset, col):
        dataset[col] = dataset[col].interpolate()
        # And fill the initial data points if needed:
        dataset[col] = dataset[col].fillna(method='bfill')
        return dataset

    def lin_model(self,dataset,col):
        for col_val in [c for c in dataset.columns if not 'label' in c and not col in c]:
            dataset = self.impute_interpolate(dataset, col_val)
        print(dataset.isna().sum())
        missing = dataset[dataset[col].isna()]
        nonmissing = dataset[dataset[col].notna()]

        features = nonmissing.loc[:,nonmissing.columns!=col]
        labels = nonmissing[col]
        model = LinearRegression()
        model.fit(features,labels)

        features_miss = missing.loc[:,nonmissing.columns!=col]
        prediction = model.predict(features_miss)
        #dataset.ix[features_miss.index.values][col] = prediction
        dataset.at[features_miss.index.values,col] = prediction
        return dataset