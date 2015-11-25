import numpy as np
from scipy.special import expit
from scipy.optimize import fmin_cg

__all__ = ['NeuralNetModel', 'train_network']

class NeuralNetModel():
    """ Output of a neural network training algorithm."""
    def __init__(self, neural_network):
        self._network = neural_network
        
    def get_weights(self):
        """ Return the weights used by the model to make predictions."""
        return self._network.weights
    
    def predict(self, X):
        """ Classify the observations in X.
        
        Args: 
            X (array): data to be classified. 
        Returns:
            array: each entry is the class assigned to the corresponding
            row in X.
        Raises:
            ValueError: if the features in X do not match the model. 
        """
        
        num_features = self._network.layers[0]
        if not X.shape[1] == num_features:
            raise ValueError("Number of features in data must be {0}."
                             .format(num_features))
        forward_propogate(self.get_weights(), X, self._network)
        class_predictions = self._network.get_final_activation()
        return np.argmax(class_predictions, axis = 1)
    
    def __repr__(self):
        return("Trained neural network with {0} hidden layers of size(s) {1}."
               .format(len(self._network.layers)-2, self._network.layers[1:-1]))

class NeuralNetwork():
    """ Internal network class to keep track of its properties."""
    def __init__(self, layer_sizes):
        self.layers = layer_sizes
        self._shapes = []
        self.activations = []
        self.raw_outputs = []
        self._initialize_shapes()
        self.weights = None
        
    def num_transitions(self):
        """ Return the number of transitions between layers."""
        return len(self.layers) - 1
    
    def get_final_activation(self):
        """ Return the last activation, giving the current probabilities."""
        return self.activations[self.num_transitions()]
        
    def _initialize_shapes(self):
        """ Determine the shapes for the weights in the network."""
        for index in range(self.num_transitions()):
            input_size = self.layers[index]
            output_size = self.layers[index+1]
            shape = (output_size, 1 + input_size)
            self._shapes.append(shape)
    
    def initialize_weights(self):
        """ Return random initial weights according to this network's shapes."""
        weights = []
        for index in range(self.num_transitions()):
            init_epsilon = self._get_initial_epsilon(index)
            shape_x, shape_y = self._shapes[index]
            weight = init_epsilon * (np.random.rand(shape_x, shape_y) * 2 - 1)
            weights.append(weight)
        return weights
        
    def _get_initial_epsilon(self, index):
        """ Return scaling factor for initialized weights."""
        input_size = self.layers[index]
        output_size = self.layers[index+1]
        return (6 / (input_size + output_size))**(1/2)
        
    def reshape_weights(self, flat_weights):
        """ Restore shape of weights according to this network's shapes."""
        start_index = 0
        shaped_weights = []
        for shape in self._shapes:
            matrix_size = shape[0] * shape[1]
            end_index = start_index + matrix_size
            shaped_weights.append(flat_weights[start_index:end_index].reshape(shape))
            start_index = end_index
        return shaped_weights
        
def train_network(X, Y, layers, regularization = 0, max_iters = 200):
    """ Train a neural network and return the model.
    
    Args:
        X (array): data consisting of rows of features. 
        Y (array): array of labels corresponding to each row in X.
            Must consist of integers from 0 to n for some integer n.
        layers (list): the number of features in each layer. The first
            entry must be the number of features (columns) in X, the 
            last must be the number of classes, and those inbetween
            determine the size of each hidden layer.
        regularization (int): penalty factor for having larger weights.
            (defualt: 0).
        max_iters (int): the max number of iterations used by the algorithm
            when searching for optimal weights. A higher number will produce
            a better fit but extends run time (default: 200).
    """
    check_input_validity(X, Y, layers)
    num_classes = layers[-1]
    network = NeuralNetwork(layers)
    initial_weights = flatten_weights(network.initialize_weights()) 
    Y = process_labels(Y, num_classes)
    optimal = fmin_cg(compute_cost, 
                      initial_weights, 
                      back_propogate, 
                      args = (X, Y, network, regularization), 
                      maxiter = max_iters)
    forward_propogate(network.reshape_weights(optimal), X, network)
    network.weights = network.reshape_weights(optimal)
    return NeuralNetModel(network)
    
def compute_cost(flat_weights, X, label_matrix, network, regularization):
    """ Propogate weights through network and compute cost function."""
    weights = network.reshape_weights(flat_weights)
    forward_propogate(weights, X, network)
    return cost_function(weights, label_matrix, network, regularization)
    
def forward_propogate(weights, X, network):
    """ Perform forward propogation on the given network and dataset."""
    raw_outputs = [X]
    activations = [X]
    for i in range(network.num_transitions()):
        activations[i] = insert_ones(activations[i])
        weight = weights[i]
        raw_output = activations[i].dot(weight.transpose())
        activation = sigmoid(raw_output)
        raw_outputs.append(raw_output)
        activations.append(activation)
    network.activations = activations
    network.raw_outputs = raw_outputs
        
def cost_function(weights, label_matrix, network, regularization):
    """ Compute the cost function for the network's current state."""
    a = network.get_final_activation()
    Y = label_matrix
    m = len(label_matrix)
    weight_sum = 0
    for weight in weights:
        weight_sum += (weight[:,1:]**2).sum()
    reg_term = (regularization / (2*m)) * weight_sum
    return (-Y * log(a) - (1-Y) * log(1-a)).sum() / m + reg_term
    
def back_propogate(flat_weights, X, label_matrix, network, regularization):
    """ Use back propogation to get the gradient of the cost function."""
    weights = network.reshape_weights(flat_weights)
    #Todo: clean up algorithm so this step isn't necessary
    if len(network.activations) == 0:
        forward_propogate(weights, X, network)
    deltas = get_deltas(weights, label_matrix, network)
    weight_gradients = get_weight_gradients(weights, deltas, network, regularization)
    return weight_gradients

def get_deltas(weights, label_matrix, network):
    """ Return a list of the deltas needed for the gradient computation."""
    deltas = []
    delta = network.get_final_activation() - label_matrix
    deltas.append(delta)
    for index in reversed(range(1, network.num_transitions())):
        weight = weights[index][:,1:]
        sigmoid_grad = sigmoid_gradient(network.raw_outputs[index])
        delta = delta.dot(weight) * sigmoid_grad
        deltas.insert(0, delta)
    return deltas

def get_weight_gradients(weights, deltas, network, regularization):
    """ Return a flat array of the gradients of the weights."""
    activations = network.activations
    weight_gradients = []
    m = activations[0].shape[0]
    for index, weight in enumerate(weights):
        weight[:,0] = 0
        delta, activation = deltas[index], activations[index]
        base_term = delta.transpose().dot(activation) / m
        reg_term = regularization * weight / m
        weight_gradients.append(base_term + reg_term)
    return flatten_weights(np.array(weight_gradients))

def flatten_weights(weights):
    """ Return a flat array of the weights."""
    return np.concatenate([weight.flatten() for weight in weights])

def process_labels(Y, num_labels):
    """ Given a sequence of labels 0 to n, produce a 0-1 matrix where entry
    i, j is 1 if and only if the ith label is j."""
    label_matrix = np.zeros((len(Y), num_labels))
    for i in range(num_labels):
        label_matrix[:,i] = 1 * (Y == i)
    return label_matrix

def sigmoid_gradient(z):
    """ Gradient of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))

def sigmoid(z):
    """ Vectorized sigmoid/logistic function."""
    return expit(z)
    
def insert_ones(X):
    """ Insert a column of ones in front of the dataset X and return it."""
    X = array_to_ndarray(X)
    num_rows = X.shape[0]
    return np.hstack((np.ones((num_rows, 1)), X))
    
def array_to_ndarray(X):
    """ Return a multidimensional version of X if it isn't already one."""
    if len(X.shape) == 1:
        X = X.reshape(X.shape[0], 1)
    return X
    
def log(num_array):
    """ Logarithm extended to include 0 to avoid log of 0 rounding errors."""
    offset = 1e-20
    return np.log(num_array + offset)

def check_input_validity(X, Y, layers):
    """ Raise error if invalid input is passed to network training method."""
    try:
        observations, features = X.shape
        label_size = Y.size
    except AttributeError:
        raise AttributeError("X and Y must be numpy arrays, "
                             "or pandas data frames/series.")
    if not observations == label_size:
        raise ValueError("Number of rows in X does not match "
                         "number of labels.")
    if not features == layers[0]:
        raise ValueError("Number of features in X does not match "
                         "first entry of layers.")
    unique_labels = Y.unique()
    if not set(unique_labels) <= set(range(layers[-1])):
        raise ValueError("Labels in Y must be numbers from 0 and n, "
                         "where n is the final entry of layers.")