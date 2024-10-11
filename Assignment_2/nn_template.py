import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

# Returns the ReLU value of the input x
def relu(x):
    return np.maximum(0, x)

# Returns the derivative of the ReLU value of the input x
def relu_derivative(x):
    return (x > 0).astype(int)

## TODO 1a: Return the sigmoid value of the input x
def sigmoid(x):
   return 1 / (1 + np.exp(-x))

## TODO 1b: Return the derivative of the sigmoid value of the input x
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

## TODO 1c: Return the tanh value of the input x
def tanh(x):
    return np.tanh(x)

## TODO 1d: Return the derivative of the tanh value of the input x
def tanh_derivative(x):
    return 1 - np.tanh(x)**2

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

        ## Initialize weights and biases for all hidden and output layers
        self.weights = []
        self.biases = []
        layers = [input_dim] + hidden_dims + [1]
        
        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i], layers[i + 1]))
            self.biases.append(np.random.randn(layers[i + 1]))
    
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
        self.z = []  # List to store weighted inputs
        self.a = [X]  # List to store activations, with input X as the first activation
        for i in range(len(self.weights) - 1):  # Excluding output layer
            z = np.dot(self.a[-1], self.weights[i]) + self.biases[i]
            self.z.append(z)
            a = self.activations[i](z)
            self.a.append(a)

        # Output layer
        z = np.dot(self.a[-1], self.weights[-1]) + self.biases[-1]
        self.z.append(z)
        output_probs = sigmoid(z)
        self.a.append(output_probs)

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
        N = X.shape[0]
        self.grad_weights = [np.zeros(w.shape) for w in self.weights]
        self.grad_biases = [np.zeros(b.shape) for b in self.biases]

        # Compute gradient for output layer
        delta = (self.a[-1] - y.reshape(-1, 1)) / N  # Reshape y for broadcasting
        self.grad_weights[-1] = np.dot(self.a[-2].T, delta)
        self.grad_biases[-1] = np.sum(delta, axis=0)

        # Backpropagate to hidden layers
        for l in range(len(self.weights) - 2, -1, -1):
            delta = np.dot(delta, self.weights[l + 1].T) * self.activation_derivatives[l](self.z[l])
            self.grad_weights[l] = np.dot(self.a[l].T, delta)
            self.grad_biases[l] = np.sum(delta, axis=0)

        return self.grad_weights, self.grad_biases

    def step_bgd(self, weights, biases, delta_weights, delta_biases, optimizer_params, epoch):
        gd_flag = optimizer_params['gd_flag']
        learning_rate = optimizer_params['learning_rate']
        momentum = optimizer_params['momentum']
        decay_constant = optimizer_params['decay_constant']

        if gd_flag == 1:  # Vanilla SGD with static learning rate
            updated_W = [w - learning_rate * dw for w, dw in zip(weights, delta_weights)]
            updated_B = [b - learning_rate * db for b, db in zip(biases, delta_biases)]
        elif gd_flag == 2:  # Vanilla SGD with exponential learning rate decay
            lr = learning_rate * np.exp(-decay_constant * epoch)
            updated_W = [w - lr * dw for w, dw in zip(weights, delta_weights)]
            updated_B = [b - lr * db for b, db in zip(biases, delta_biases)]
        elif gd_flag == 3:  # SGD with momentum
            if not hasattr(self, 'velocity_W'):
                self.velocity_W = [np.zeros_like(w) for w in weights]
                self.velocity_B = [np.zeros_like(b) for b in biases]
            self.velocity_W = [momentum * v + (1 - momentum) * dw for v, dw in zip(self.velocity_W, delta_weights)]
            self.velocity_B = [momentum * v + (1 - momentum) * db for v, db in zip(self.velocity_B, delta_biases)]
            updated_W = [w - learning_rate * v for w, v in zip(weights, self.velocity_W)]
            updated_B = [b - learning_rate * v for b, v in zip(biases, self.velocity_B)]
       
        return updated_W, updated_B

    def step_adam(self, weights, biases, delta_weights, delta_biases, optimizer_params):
        
        optimizer = "adam"
        optimizer_params = {
        'learning_rate': 0.01,
        'beta1' : 0.8,
        'beta2' : 0.999,
        'eps' : 1e-8
     }

        learning_rate = optimizer_params['learning_rate']
        beta1 = optimizer_params['beta1']
        beta2 = optimizer_params['beta2']
        eps = optimizer_params['eps']

        if not hasattr(self, 'm_weights'):
            self.m_weights = [np.zeros_like(w) for w in weights]
            self.v_weights = [np.zeros_like(w) for w in weights]
            self.m_biases = [np.zeros_like(b) for b in biases]
            self.v_biases = [np.zeros_like(b) for b in biases]
            self.t = 0
    
        self.t += 1

        updated_W = []
        updated_B = []

        for i in range(len(weights)):
            # Update biased first moment estimate
            self.m_weights[i] = beta1 * self.m_weights[i] + (1 - beta1) * delta_weights[i]
            self.m_biases[i] = beta1 * self.m_biases[i] + (1 - beta1) * delta_biases[i]

            # Update biased second moment estimate
            self.v_weights[i] = beta2 * self.v_weights[i] + (1 - beta2) * (delta_weights[i] ** 2)
            self.v_biases[i] = beta2 * self.v_biases[i] + (1 - beta2) * (delta_biases[i] ** 2)

        # Bias correction for first moment (m_hat)
            m_hat_W = self.m_weights[i] / (1 - beta1 ** self.t)
            m_hat_B = self.m_biases[i] / (1 - beta1 ** self.t)

        # Bias correction for second moment (v_hat)
            v_hat_W = self.v_weights[i] / (1 - beta2 ** self.t)
            v_hat_B = self.v_biases[i] / (1 - beta2 ** self.t)

        # Update weights and biases
            updated_W.append(weights[i] - learning_rate * m_hat_W / (np.sqrt(v_hat_W) + eps))
            updated_B.append(biases[i] - learning_rate * m_hat_B / (np.sqrt(v_hat_B) + eps))

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
    def plot_loss(self, train_losses, test_losses):
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('loss.png')
 

# Example usage:
if __name__ == "__main__":
    # Read from data.csv 
    csv_file_path = "C:/Users/rohit/Downloads/rohit/assgmt2/data_train.csv"
    eval_file_path = "C:/Users/rohit/Downloads/rohit/assgmt2/data_eval.csv"

    
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
    optimizer = "bgd"
    optimizer_params = {
        'learning_rate': 0.1,
        'gd_flag': 1,
        'momentum': 0.9,
        'decay_constant': 0.2
    }
    
    # For Adam optimizer you can use the following
    # optimizer = "adam"
    # optimizer_params = {
    #     'learning_rate': 0.01,
    #     'beta1' : 0.8,
    #     'beta2' : 0.999,
    #     'eps' : 1e-8
    # }


     
    model = NN(input_dim, hidden_dims)
    train_losses, test_losses = model.train(X_train, y_train, X_eval, y_eval,
                                    num_epochs, batch_size, optimizer, optimizer_params) #trained on concentric circle data 
    test_preds = model.forward(X_eval)

    test_accuracy = np.mean((test_preds > 0.5).reshape(-1,) == y_eval)
    print(f"Final Test accuracy: {test_accuracy:.4f}")

    model.plot_loss(train_losses, test_losses)
