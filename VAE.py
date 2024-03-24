import numpy as np
from vae_operations import VaeOperations

# Defines a Variational Autoencoder.
class VAE:
    def __init__(self, input_dim, latent_dim=50, hidden_dims=None):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [50, 50]
        self.weights = self.initialize_weights()
        self.biases = self.initialize_biases()

    # Weights are initialized to random values based on a
    # normal distribution scaled by the square root of 2 divided by the previous layer's dimension
    def initialize_weights(self):
        weights = {}
        prev_dim = self.input_dim

        # Initialize encoder weights
        for i, dim in enumerate(self.hidden_dims):
            weights[f'W_encoder_{i + 1}'] = np.random.randn(prev_dim, dim) * np.sqrt(2. / prev_dim)
            prev_dim = dim

        # Initialize weights for mean and log variance vectors
        weights['W_mean'] = np.random.randn(prev_dim, self.latent_dim) * np.sqrt(2. / prev_dim)
        weights['W_log_var'] = np.random.randn(prev_dim, self.latent_dim) * np.sqrt(2. / prev_dim)

        # Initialize decoder weights
        decoder_dims = [self.latent_dim] + self.hidden_dims[::-1]  # Reverse hidden_dims for decoding
        prev_dim = self.latent_dim
        for i, dim in enumerate(decoder_dims):
            weights[f'W_decoder_{i + 1}'] = np.random.randn(prev_dim, dim if i < len(
                decoder_dims) - 1 else self.latent_dim) * np.sqrt(2. / prev_dim)
            prev_dim = dim

        return weights

    # Initialise biases and fill them with 0s
    def initialize_biases(self):
        biases = {}
        # Initialize encoder biases
        for i, dim in enumerate(self.hidden_dims):
            biases[f'b_encoder_{i + 1}'] = np.zeros(dim)

        # Initialize biases for mean and log variance vectors
        biases['b_mean'] = np.zeros(self.latent_dim)
        biases['b_log_var'] = np.zeros(self.latent_dim)

        # Initialize decoder biases
        decoder_dims = [self.latent_dim] + self.hidden_dims[::-1]
        for i, dim in enumerate(decoder_dims):
            biases[f'b_decoder_{i + 1}'] = np.zeros(dim if i < len(decoder_dims) - 1 else self.latent_dim)

        return biases

    # Processes the input X through the encoder network to produce mean and log variance vectors (mean, log_var).
    def encode(self, X):
        hidden_activations = []
        hidden = X
        for i in range(len(self.hidden_dims)):
            hidden = VaeOperations.relu(
                np.dot(hidden, self.weights[f'W_encoder_{i + 1}']) + self.biases[f'b_encoder_{i + 1}'])
            hidden_activations.append(hidden)

        mean = np.dot(hidden, self.weights['W_mean']) + self.biases['b_mean']
        log_var = np.dot(hidden, self.weights['W_log_var']) + self.biases['b_log_var']
        return mean, log_var, hidden_activations

    # Performs the "reparameterization trick" to sample from the latent space defined by mean and log_var,
    # enabling gradient backpropagation through random sampling.
    def reparameterize(self, mean, log_var):
        """Performs the reparameterization trick to sample from latent space."""
        eps = np.random.normal(size=mean.shape)
        return mean + np.exp(0.5 * log_var) * eps

    # Takes a latent space vector z and processes it through the decoder network to reconstruct the input data.
    def decode(self, z):
        hidden = z
        for i in range(len(self.hidden_dims) + 1):
            hidden = np.dot(hidden, self.weights[f'W_decoder_{i + 1}']) + self.biases[f'b_decoder_{i + 1}']
            if i < len(self.hidden_dims):
                hidden = VaeOperations.relu(hidden)
            else:
                hidden = VaeOperations.sigmoid(hidden)
        return hidden.reshape(-1, self.latent_dim)

    # Orchestrates the training process over a specified number of epochs, using mini-batch gradient descent with a given
    # batch_size, learning_rate, and regularization strength lambda_param.
    # It involves encoding inputs, reparameterization, decoding (reconstruction), and updating weights and biases
    # based on gradients computed from the loss function.
    def train(self, X, epochs, batch_size, learning_rate, lambda_param):
        for epoch in range(epochs):
            np.random.shuffle(X)
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i + batch_size]
                mean, log_var, hidden_activations = self.encode(X_batch)  # Now also getting hidden_activations
                z = self.reparameterize(mean, log_var)
                reconstructed_X = self.decode(z)
                self.update_weights_gradients(X_batch, reconstructed_X, z, mean, log_var, hidden_activations, learning_rate, lambda_param)

    # Computes gradients and updates weights and biases for both encoder and decoder.
    # This method utilizes the gradients of the reconstruction loss and the Kullback-Leibler (KL) divergence loss
    # to perform the updates.
    def update_weights_gradients(self, X_batch, reconstructed_X, z, mean, log_var, hidden_activations, learning_rate, lambda_param):
        """Updates weights and biases based on gradients."""
        batch_size = X_batch.shape[0]

        # Compute gradients for the reconstruction loss
        d_loss_recon = 2 * (reconstructed_X - X_batch) / batch_size  # Assuming MSE loss

        # Initialize the gradient of loss w.r.t. latent variable z
        d_loss_z = d_loss_recon

        hidden = hidden_activations[-1]
        # Backpropagate through decoder
        for i in reversed(range(len(self.hidden_dims) + 1)):
            layer_key = f'W_decoder_{i + 1}'
            bias_key = f'b_decoder_{i + 1}'

            activation = z if i == 0 else np.tanh(np.dot(z if i == 1 else hidden, self.weights[f'W_decoder_{i}']))
            grad_weights = np.dot(activation.T, d_loss_z) / batch_size
            grad_biases = np.sum(d_loss_z, axis=0) / batch_size

            # Update weights and biases
            self.weights[layer_key] -= learning_rate * grad_weights + lambda_param * self.weights[layer_key]
            self.biases[bias_key] -= learning_rate * grad_biases + lambda_param * self.biases[bias_key]

            if i > 0:
                d_loss_z = np.dot(d_loss_z, self.weights[layer_key].T) * (1 - np.tanh(activation) ** 2)  # Derivative of tanh for activation function

        # Compute gradients for KL divergence loss
        d_kl_mean = mean / batch_size
        d_kl_log_var = (np.exp(log_var) - 1) / batch_size

        # Update weights and biases for mean and log variance
        self.weights['W_mean'] -= learning_rate * np.dot(hidden.T, d_kl_mean) + lambda_param * self.weights['W_mean']
        self.biases['b_mean'] -= learning_rate * np.sum(d_kl_mean, axis=0) + lambda_param * self.biases['b_mean']
        self.weights['W_log_var'] -= learning_rate * np.dot(hidden.T, d_kl_log_var) + lambda_param * self.weights['W_log_var']
        self.biases['b_log_var'] -= learning_rate * np.sum(d_kl_log_var, axis=0) + lambda_param * self.biases['b_log_var']

    # Generate new data samples
    def generate(self, num_samples=1):
        z = np.random.randn(num_samples, self.latent_dim)
        return self.decode(z).reshape(num_samples, 50)  # Ensure output shape

    #Generate response vectors
    def generate_response(self, query_vector):
        mean, log_var, _ = self.encode(query_vector.reshape(1, -1))  # Ensure input is 2D
        z = self.reparameterize(mean, log_var)
        generated_response = self.decode(z)
        return generated_response.flatten()
