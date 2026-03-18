### Looking at the dataset Features

### Features with no significance

### Features with no variance

### Feature Correlation
We want to do a correlation check across all features to see whether there is any linear relationship between values. This helps see if they are any potential redundant information within the dataset.

If two features are highly correlated and represent similar information, one of them could be removed to simplify the model and not double weight a feature of the dataset.

So firstly we want to create a correlation matrix and display only one half (Trianglur matrix) of values to see how each column feature relates to the other. Correlation is symmetric.

Ideally, features should not be highly correlated with eachother. Moderate correlation is acceptable.

## Relation to the target Variable
Most important one to check is if any feature highly correlates to our target (Our target being our classifcaiton if something is fraud or not). 

If there is it suggests that there is a data leakage or the problem is easily seperable.
If there is not we just continue with creating a model.

## Visualising Fraud Distribution
We select the two features with the highest absolute correlation with the target variable. These may not be the most important features, but they are the most informative for plotting on a graph.

The main thing we can see with the plot is, is it possible to actually separate the features if the distributions cluster around eachother it'll be hard for a model to distinguish between fraud and non-fraud.

