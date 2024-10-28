import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

# Returns the ReLU value of the input x
def relu(x):
    return max(0, x)

# Returns the derivative of the ReLU value of the input x
def relu_derivative(x):
    return (x>0).astype(int)

## TODO 1a: Return the sigmoid value of the input x
def sigmoid(x):
    return 1/(1+ np.exp(-x))

## TODO 1b: Return the derivative of the sigmoid value of the input x
def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig*(1-sig)
## TODO 1c: Return the derivative of the tanh value of the input x
def tanh(x):
    return np.tanh(x)

## TODO 1d: Return the derivative of the tanh value of the input x
def tanh_derivative(x):
    tanh_value = tanh(x)
    return 1- tanh_value**2

# Mapping from string to function
str_to_func = {
    'sigmoid': (sigmoid, sigmoid_derivative),
    'relu': (relu, relu_derivative),
    'tanh': (tanh, tanh_derivative)
}

# Given a list of activation functions, the following function returns
# the corresponding list of activation functions and their derivatives
def get_activation_functions(activations):  
    activation_funcs, activation_derivatives = [], []
    for activation in activations:
        activation_func, activation_derivative = str_to_func[activation]
        activation_funcs.append(activation_func)
        activation_derivatives.append(activation_derivative)
    return activation_funcs, activation_derivatives

class NN:
    def __init__(self, input_dim, hidden_dims, activations=None):
        '''
        Parameters
        ----------
        input_dim : int
            size of the input layer.
        hidden_dims : LIST<int>
            List of positive integers where each integer corresponds to the number of neurons 
            in the hidden layers. The list excludes the number of neurons in the output layer.
            For this problem, we fix the output layer to have just 1 neuron.
        activations : LIST<string>, optional
            List of strings where each string corresponds to the activation function to be used 
            for all hidden layers. The list excludes the activation function for the output layer.
            For this problem, we fix the output layer to have the sigmoid activation function.
        ----------
        Returns : None
        ----------
        '''
        assert(len(hidden_dims) > 0)
        assert(activations == None or len(hidden_dims) == len(activations))
         
        # If activations is None, we use sigmoid activation for all layers
        if activations == None:
            self.activations = [sigmoid]*(len(hidden_dims)+1)
            self.activation_derivatives = [sigmoid_derivative]*(len(hidden_dims)+1)
        else:
            self.activations, self.activation_derivatives = get_activation_functions(activations + ['sigmoid'])

        ## TODO 2: Initialize weights and biases for all hidden and output layers
        ## Initialization can be done with random normal values, you are free to use
        ## any other initialization technique.
        layers = [input_dim] + hidden_dims + [1]
        self.weights = [np.random.normal(0, 1, (layers[i], layers[i + 1])) for i in range(len(layers) - 1)]
        self.biases = [np.random.normal(0, 1, layers[i + 1]) for i in range(len(layers) - 1)]

    def forward(self, X):
        '''
        Parameters
        ----------
        X : input data, numpy array of shape (N, D) where N is the number of examples and D 
            is the dimension of each example
        ----------
        Returns : output probabilities, numpy array of shape (N, 1) 
        ----------
        '''
        # Forward pass
        
        ## TODO 3a: Compute activations for all the nodes with the corresponding
        ## activation function of each layer applied to the hidden nodes
        # Initialize the list of activations with the input layer
        self.a = [X]
    # Iterate over hidden layers
        for i in range(len(self.weights) - 1):  # Exclude the output layer for this loop
            z = np.dot(self.a[-1], self.weights[i]) + self.biases[i]  # Weighted sum
            a = self.activations[i](z)  # Activation function
            self.a.append(a)  # Store the activation

        ## TODO 3b: Calculate the output probabilities of shape (N, 1) where N is number of examples
    # Output layer
        z_output = np.dot(self.a[-1], self.weights[-1]) + self.biases[-1]  # Weighted sum for output layer
        output_probs = sigmoid(z_output)  # Apply sigmoid activation for output layer
        self.a.append(output_probs)  # Store the output activation
        return output_probs

    

    def backward(self, X, y):
        '''
        Parameters
        ----------
        X : input data, numpy array of shape (N, D) where N is the number of examples and D 
            is the dimension of each example
        y : target labels, numpy array of shape (N, 1) where N is the number of examples
        ----------
        Returns : gradients of weights and biases
        ----------
        '''
        # Backpropagation
        N = X.shape[0]
        ## TODO 4a: Compute gradients for the output layer after computing derivative of 
        ## sigmoid-based binary cross-entropy loss
        ## Hint: When computing the derivative of the cross-entropy loss, don't forget to 
        ## divide the gradients by N (number of examples)  
        output_probs = self.a[-1]
        delta_output = (output_probs - y.reshape(-1, 1)) / N
        self.grad_weights = []
        self.grad_biases = []
        self.grad_weights.append(np.dot(self.a[-2].T, delta_output))
        self.grad_biases.append(np.sum(delta_output, axis=0))
        ## TODO 4b: Next, compute gradients for all weights and biases for all layers
        ## Hint: Start from the output layer and move backwards to the first hidden layer
        delta = delta_output 
        for l in range(len(self.weights) - 2, -1, -1):
            delta = np.dot(delta, self.weights[l + 1].T) * self.activation_derivatives[l](self.a[l + 1])
            self.grad_weights.append(np.dot(self.a[l].T, delta))
            self.grad_biases.append(np.sum(delta, axis=0))
            
        self.grad_weights.reverse()
        self.grad_biases.reverse()
        
        return self.grad_weights, self.grad_biases

    def step_bgd(self, weights, biases, delta_weights, delta_biases, optimizer_params, epoch):
        '''
        Parameters
        ----------
            weights: Current weights of the network.
            biases: Current biases of the network.
            delta_weights: Gradients of weights with respect to loss.
            delta_biases: Gradients of biases with respect to loss.
            optimizer_params: Dictionary containing the following keys:
                learning_rate: Learning rate for the update step.
                gd_flag: 1 for Vanilla GD, 2 for GD with Exponential Decay, 3 for Momentum
                momentum: Momentum coefficient, used when gd_flag is 3.
                decay_constant: Decay constant for exponential learning rate decay, used when gd_flag is 2.
            epoch: Current epoch number
        '''
        gd_flag = optimizer_params['gd_flag']
        learning_rate = optimizer_params['learning_rate']
        momentum = optimizer_params['momentum']
        decay_constant = optimizer_params['decay_constant']

        ### Calculate updated weights using methods as indicated by gd_flag
        updated_W = []
        updated_B = []
        ## TODO 5a: Variant 1(gd_flag = 1): Vanilla GD with Static Learning Rate
        ## Use the hyperparameter learning_rate as the static learning rate
        if gd_flag == 1:
            updated_W = [w - learning_rate * dw for w, dw in zip(weights, delta_weights)]
            updated_B = [b - learning_rate * db for b, db in zip(biases, delta_biases)]
            
        ## TODO 5b: Variant 2(gd_flag = 2): Vanilla GD with Exponential Learning Rate Decay
        ## Use the hyperparameter learning_rate as the initial learning rate
        ## Use the parameter epoch for t
        ## Use the hyperparameter decay_constant as the decay constant
        elif gd_flag == 2:
            decayed_learning_rate = learning_rate * np.exp(-decay_constant * epoch)
            updated_W = [w - decayed_learning_rate * dw for w, dw in zip(weights, delta_weights)]
            updated_B = [b - decayed_learning_rate * db for b, db in zip(biases, delta_biases)]
            
        ## TODO 5c: Variant 3(gd_flag = 3): GD with Momentum
        ## Use the hyperparameters learning_rate and momentum
        elif gd_flag == 3:
            if not hasattr(self, 'velocity'):
                self.velocity = [np.zeros_like(w) for w in weights]
            self.velocity = [momentum * v + (1 - momentum) * dw for v, dw in zip(self.velocity, delta_weights)]
            updated_W = [w - learning_rate * v for w, v in zip(weights, self.velocity)]
            updated_B = [b - learning_rate * db for b, db in zip(biases, delta_biases)]
            
        return updated_W, updated_B

    def step_adam(self, weights, biases, delta_weights, delta_biases, optimizer_params):
        '''
        Parameters
        ----------
            weights: Current weights of the network.
            biases: Current biases of the network.
            delta_weights: Gradients of weights with respect to loss.
            delta_biases: Gradients of biases with respect to loss.
            optimizer_params: Dictionary containing the following keys:
                learning_rate: Learning rate for the update step.
                beta: Exponential decay rate for the first moment estimates.
                gamma: Exponential decay rate for the second moment estimates.
                eps: A small constant for numerical stability.
        '''
        
        learning_rate = optimizer_params['learning_rate']
        beta1 = optimizer_params['beta1']
        beta2 = optimizer_params['beta2']
        eps = optimizer_params['eps']       

        ## TODO 6: Return updated weights and biases for the hidden layer based on the update rules for Adam Optimizer
        if not hasattr(self, 'm'):
            self.m = [np.zeros_like(w) for w in weights]
        
        if not hasattr(self, 'v'):
            self.v = [np.zeros_like(w) for w in weights]    
        
        if not hasattr(self, 't'):
            self.t = 0 
        
        self.t += 1  
        updated_W = []
        updated_B = []  
        
        for i in range(len(weights)):
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * delta_weights[i]
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * (delta_weights[i] ** 2)
            m_hat = self.m[i] / (1 - beta1 ** self.t)
            v_hat = self.v[i] / (1 - beta2 ** self.t)
            updated_weight = weights[i] - learning_rate * m_hat / (np.sqrt(v_hat) + eps)
            updated_W.append(updated_weight)
            
        for i in range(len(biases)):
            updated_bias = biases[i] - learning_rate * delta_biases[i]
            updated_B.append(updated_bias)
            
        return updated_W, updated_B

    def train(self, X_train, y_train, X_eval, y_eval, num_epochs, batch_size, optimizer, optimizer_params):
        train_losses = []
        test_losses = []
        for epoch in range(num_epochs):
            # Divide X,y into batches
            X_batches = np.array_split(X_train, X_train.shape[0]//batch_size)
            y_batches = np.array_split(y_train, y_train.shape[0]//batch_size)
            for X, y in zip(X_batches, y_batches):
                # Forward pass
                self.forward(X)
                # Backpropagation and gradient descent weight updates
                dW, db = self.backward(X, y)
                if optimizer == "adam":
                    self.weights, self.biases = self.step_adam(
                        self.weights, self.biases, dW, db, optimizer_params)
                elif optimizer == "bgd":
                    self.weights, self.biases = self.step_bgd(
                        self.weights, self.biases, dW, db, optimizer_params, epoch)

            # Compute the training accuracy and training loss
            train_preds = self.forward(X_train)
            train_loss = np.mean(-y_train*np.log(train_preds) - (1-y_train)*np.log(1-train_preds))
            train_accuracy = np.mean((train_preds > 0.5).reshape(-1,) == y_train)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            train_losses.append(train_loss)

            # Compute the test accuracy and test loss
            test_preds = self.forward(X_eval)
            test_loss = np.mean(-y_eval*np.log(test_preds) - (1-y_eval)*np.log(1-test_preds))
            test_accuracy = np.mean((test_preds > 0.5).reshape(-1,) == y_eval)
            print(f"Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
            test_losses.append(test_loss)

        return train_losses, test_losses

    
    # Plot the loss curve
    def plot_loss(self, train_losses, test_losses, optimizer, optimizer_params):
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        if optimizer == "bgd":
            plt.savefig(f'loss_bgd_{optimizer_params["gd_flag"]}.png')
        else:
            plt.savefig(f'loss_adam.png')
 

# Example usage:
if __name__ == "__main__":
    # Read from data.csv 
    csv_file_path = "data_train.csv"
    eval_file_path = "data_eval.csv"
    
    data = np.genfromtxt(csv_file_path, delimiter=',', skip_header=0)
    data_eval = np.genfromtxt(eval_file_path, delimiter=',', skip_header=0)

    # Separate the data into X (features) and y (target) arrays
    X_train = data[:, :-1]
    y_train = data[:, -1]
    X_eval = data_eval[:, :-1]
    y_eval = data_eval[:, -1]

    # Create and train the neural network
    input_dim = X_train.shape[1]
    X_train = X_train**2
    X_eval = X_eval**2
    hidden_dims = [4,2] # the last layer has just 1 neuron for classification
    num_epochs = 30
    batch_size = 100
    activations = ['sigmoid', 'sigmoid']
    
    
    # For Adam optimizer you can use the following
    # optimizer = "adam"
    # optimizer_params = {
    #     'learning_rate': 0.01,
    #     'beta1' : 0.9,
    #     'beta2' : 0.999,
    #     'eps' : 1e-8
    # }
    optimizers = [
        ("bgd", {'learning_rate': 0.05, 'gd_flag': 1, 'momentum': 0.99, 'decay_constant': 0.2}),
        ("bgd", {'learning_rate': 0.1, 'gd_flag': 2, 'momentum': 0.99, 'decay_constant': 0.2}),
        ("bgd", {'learning_rate': 0.1, 'gd_flag': 3, 'momentum': 0.99, 'decay_constant': 0.2}),
        ("adam", {'learning_rate': 0.01, 'beta1': 0.9, 'beta2': 0.999, 'eps': 1e-8}),
    ]
    
    for optimizer, optimizer_params in optimizers: 
        model = NN(input_dim, hidden_dims)
        train_losses, test_losses = model.train(X_train, y_train, X_eval, y_eval,
                                    num_epochs, batch_size, optimizer, optimizer_params) #trained on concentric circle data 
        model.plot_loss(train_losses, test_losses, optimizer, optimizer_params)
    test_preds = model.forward(X_eval)

    test_accuracy = np.mean((test_preds > 0.5).reshape(-1,) == y_eval)
    print(f"Final Test accuracy: {test_accuracy:.4f}")

    model.plot_loss(train_losses, test_losses, optimizer, optimizer_params)
