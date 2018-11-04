# Python-Machine-Learning
This is an old side project I did in my first year of learning programming, when I was considering data science as a career.

Python implementation of common machine learning algorithms. Currently supports neural networks, logistic regression and linear regression. Datasets are expected to be Numpy arrays (Pandas dataframes count).

### NeuralNetwork.py
Offers `train_network` to build a neural network on training data. Once trained, function returns a `NeuralNetModel` object with a `get_predictions` method for generating predictions on new datasets. "Neural Network.ipynb" contains an example of a network trained on the MNIST dataset of handwritten digits. 

### LinearRegression.py
Offers `gradient_descent` to train a linear model. Returns a vector of weights that can be used with the `get_predictions` method to generate predictions on new data. 

### LogisticRegression.py
Offers two implementations of logistic regression, `BFGS` and `gradient_descent`, using those respective methods. Each method returns a vector of weights to be used with the `get_predictions` method. 

### DataSets.py
Offers two methods `get_binary_data` and `get_dense_data`. Each method creates a dataset with normally distributed features, with either a linearly separable vector of labels (for binary data) or a linear combination of the features (for dense data) with randomly chosen weights. Weights are returned alongside the data for comparison with weights trained by a model. 
