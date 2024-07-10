import numpy as np

class Network:
    def __init__(self, input_size, hidden_size, primary_output_size, n_hidden_layers, parameters):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.primary_output_size = primary_output_size
        self.n_hidden_layers = n_hidden_layers
        self.parameters = np.array(parameters)
        
        self.weights = {}
        self.biases = {}
        
        self.build_model()
    
    def build_model(self):
        pointer = 0

        # Assign parameters without nesting
        def assign_params(parameters, pointer, shape):
            num_elements = np.prod(shape)
            param = parameters[pointer:pointer + num_elements].reshape(shape)
            pointer += num_elements
            return param, pointer

        # Input layer to first hidden layer
        self.weights['W1'], pointer = assign_params(self.parameters, pointer, (self.input_size, self.hidden_size))
        self.biases['b1'], pointer = assign_params(self.parameters, pointer, (self.hidden_size,))
        
        # Add n hidden layers
        for i in range(2, self.n_hidden_layers + 1):
            self.weights[f'W{i}'], pointer = assign_params(self.parameters, pointer, (self.hidden_size, self.hidden_size))
            self.biases[f'b{i}'], pointer = assign_params(self.parameters, pointer, (self.hidden_size,))
        
        # Last hidden layer to output layer
        output_size = self.primary_output_size
        self.weights[f'W{self.n_hidden_layers + 1}'], pointer = assign_params(self.parameters, pointer, (self.hidden_size, output_size))
        self.biases[f'b{self.n_hidden_layers + 1}'], pointer = assign_params(self.parameters, pointer, (output_size,))

    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, x):
        # Input to first hidden layer
        x = self.relu(np.dot(x, self.weights['W1']) + self.biases['b1'])
        
        # Hidden layers
        for i in range(2, self.n_hidden_layers + 1):
            x = self.relu(np.dot(x, self.weights[f'W{i}']) + self.biases[f'b{i}'])
        
        # Last hidden layer to output layer
        x = np.dot(x, self.weights[f'W{self.n_hidden_layers + 1}']) + self.biases[f'b{self.n_hidden_layers + 1}']
        
        return x