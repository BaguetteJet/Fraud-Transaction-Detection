### The First Model - XGBoost

## What is XGBoost?
XGboost is a model based on gradient boosted decision trees. Instead of building a single decision tree, it constructs multiple trees sequentially, where each new tree is trained to correct the errors made by previous ones.

The first tree gives an initial preiction that may be relatively simple or noisy, the next trees focus on the errors of the previos opnes, gradually improving the overall model performance. The final prediction is a combination of all trees. 

## What is Gradient Boosting?
Gradient Boosting builds models step by step by minimising a loss function.

For every iteration:
- The model checks how wrong its current prediction is
- It calculates the gradient 
- A new tree is trained to approximise the gradient and improve the prediction.

If we build this one first it can help if our second model still is the unsupervised neural net, as the library that XGBoost comes in tells us what features were the most important in its training whihc helps with feature selection for the next model.

### Why not other models
Normal Descision Tree:
- Just if-else statements
- Too simple for our data

Random Forest:
- Doesn't learn from mistakes
- Uses randomness which can cause less accuracy