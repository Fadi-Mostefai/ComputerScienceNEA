# VAE.py

import numpy as np
from vae_operations import VaeOperations

class VAE:
    def __init__(self, input_dim, latent_dim, hidden_dims=None):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [256, 128]

        self.weights = self.initialize_weights()
        self.biases = self.initialize_biases()

    def initialize_weights(self):
        weights = {}
        prev_dim = self.input_dim
        for i, hidden_dim in enumerate(self.hidden_dims):
            weights[f"W{i+1}"] = np.random.randn(prev_dim, hidden_dim) * np.sqrt(2.0 / prev_dim)
            prev_dim = hidden_dim
        weights["mean"] = np.random.randn(self.hidden_dims[-1], self.latent_dim) * np.sqrt(2.0 / self.hidden_dims[-1])
        weights["log_var"] = np.random.randn(self.hidden_dims[-1], self.latent_dim) * np.sqrt(2.0 / self.hidden_dims[-1])
        weights["decoder"] = np.random.randn(self.latent_dim, self.input_dim) * np.sqrt(2.0 / self.latent_dim)
        return weights

    def initialize_biases(self):
        biases = {}
        for i, hidden_dim in enumerate(self.hidden_dims):
            biases[f"b{i+1}"] = np.zeros(hidden_dim)
        biases["mean"] = np.zeros(self.latent_dim)
        biases["log_var"] = np.zeros(self.latent_dim)
        biases["decoder"] = np.zeros(self.input_dim)
        return biases

    def encode(self, X):
        hidden = X
        for i, hidden_dim in enumerate(self.hidden_dims):
            hidden = np.dot(hidden, self.weights[f"W{i+1}"]) + self.biases[f"b{i+1}"]
            hidden = VaeOperations.relu(hidden)
        mean = np.dot(hidden, self.weights["mean"]) + self.biases["mean"]
        log_var = np.dot(hidden, self.weights["log_var"]) + self.biases["log_var"]
        return mean, log_var

    def reparameterize(self, mean, log_var):
        epsilon = np.random.randn(*mean.shape)
        return mean + np.exp(0.5 * log_var) * epsilon

    def decode(self, z):
        return np.dot(z, self.weights["decoder"]) + self.biases["decoder"]

    def train(self, X, epochs=100, batch_size=32, learning_rate=0.001, lambda_param=0.001):
        num_samples = X.shape[0]
        num_batches = num_samples // batch_size

        for epoch in range(epochs):
            # Shuffle the data for each epoch
            np.random.shuffle(X)

            total_loss = 0.0
            for batch_idx in range(num_batches):
                # Extract the current mini-batch
                start_idx = batch_idx * batch_size
                end_idx = (batch_idx + 1) * batch_size
                X_batch = X[start_idx:end_idx]

                mean, log_var = self.encode(X_batch)
                z = self.reparameterize(mean, log_var)
                X_pred = self.decode(z)

                reconstruction_loss = VaeOperations.mean_squared_error(X_batch, X_pred)
                kl_divergence = -0.5 * np.sum(1 + log_var - np.square(mean) - np.exp(log_var), axis=1)
                loss = np.mean(reconstruction_loss + kl_divergence) + VaeOperations.l2_regularization(self.weights,
                                                                                                      lambda_param)

                dX_pred = VaeOperations.mean_squared_error_derivative(X_batch, X_pred)
                d_mean = (mean - X_batch) / batch_size
                d_log_var = (np.exp(log_var) - 1 - log_var + np.square(mean - X_batch)) / batch_size

                d_weights_decoder = np.dot(z.T, dX_pred)
                d_biases_decoder = np.sum(dX_pred, axis=0)
                d_weights_mean = np.dot(X_batch.T, d_mean)
                d_biases_mean = np.sum(d_mean, axis=0)
                d_weights_log_var = np.dot(X_batch.T, d_log_var)
                d_biases_log_var = np.sum(d_log_var, axis=0)

                self.weights["decoder"] -= learning_rate * d_weights_decoder
                self.biases["decoder"] -= learning_rate * d_biases_decoder
                self.weights["mean"] -= learning_rate * d_weights_mean
                self.biases["mean"] -= learning_rate * d_biases_mean
                self.weights["log_var"] -= learning_rate * d_weights_log_var
                self.biases["log_var"] -= learning_rate * d_biases_log_var

                total_loss += loss

            # Compute the average loss for the epoch
            avg_loss = total_loss / num_batches

            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss}")

    def generate(self, num_samples=1):
        z = np.random.randn(num_samples, self.latent_dim)
        return self.decode(z)

    def generate_response(self, query_vector):
        # Assume query_vector is the vectorized representation of the user query
        z = self.reparameterize(*self.encode(query_vector))
        generated_response = self.decode(z)
        return generated_response

# Example usage:
# vae = VAE(input_dim=784, latent_dim=2, hidden_dims=[256, 128])
# vae.train(X_train, epochs=50, learning_rate=0.001, lambda_param=0.001)
# generated_samples = vae.generate(num_samples=10)
