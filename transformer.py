import numpy as np


class Embedding:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # Randomly initialise the embedding matrix
        self.E = np.random.randn(vocab_size, embedding_dim) * 0.01

    def forward(self, tokens):
        embedding = np.zeros((len(tokens), self.embedding_dim))
        for i, token in enumerate(tokens):
            # Set row i of the embedding matrix to "token" row of E
            embedding[i, :] = self.E[token, :]
        return embedding


class PositionalEncoding:
    def __init__(self, max_length, embedding_dim):
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.P = np.zeros((max_length, embedding_dim))

        # We use trig functions for our choice of positional matrix
        for pos in range(max_length):
            for j in range(embedding_dim):
                i = j // 2

                denominator = 10000 ** (2 * i / embedding_dim)

                if j % 2 == 0:
                    self.P[pos, j] = np.sin(pos / denominator)
                else:
                    self.P[pos, j] = np.cos(pos / denominator)

    def forward(self, X):
        n = X.shape[0]
        return X + self.P[:n, :]


class Attention:
    def __init__(self, embedding_dim, attention_dim):
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        # Randomly initialise W_Q, W_K, W_V
        self.W_Q = np.random.randn(embedding_dim, attention_dim) * 0.01
        self.W_K = np.random.randn(embedding_dim, attention_dim) * 0.01
        self.W_V = np.random.randn(embedding_dim, attention_dim) * 0.01

    def forward(self, X):
        # Calculate Q, K, V
        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V

        # Calculate scaled attention scores
        S = Q @ K.T / np.sqrt(self.attention_dim)

        # Apply softmax row-wise (subtracting the max to make it numerically stable)
        S = S - np.max(S, axis=1, keepdims=True)
        exp_S = np.exp(S)
        A = exp_S / np.sum(exp_S, axis=1, keepdims=True)

        # Weighted combination of values
        H = A @ V

        return H


class MultiHeadAttention:
    def __init__(self, embedding_dim, attention_dim, num_heads):
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads

        # Create num_heads Attention objects
        self.heads = [Attention(embedding_dim, attention_dim) for _ in range(num_heads)]

        # Final output matrix
        self.W_O = np.random.randn(self.num_heads * attention_dim, embedding_dim) * 0.01

    def forward(self, X):
        # Run X through every attention head
        outputs = [attention.forward(X) for attention in self.heads]

        # Concatenate the outputs
        concat = outputs[0]
        for i in range(1, len(outputs)):
            concat = np.hstack((concat, outputs[i]))

        # Apply W_O
        MHA = concat @ self.W_O

        return MHA


class LayerNorm:
    def __init__(self, embedding_dim, eps=1e-5):
        self.embedding_dim = embedding_dim
        self.eps = eps

        # Learnable scale and shift parameters
        self.gamma = np.ones(embedding_dim)
        self.beta = np.zeros(embedding_dim)

    def forward(self, X):
        # Calculate the mean of each token vector
        mean = np.mean(X, axis=1, keepdims=True)

        # Calculate the variance of each token vector
        variance = np.mean((X - mean) ** 2, axis=1, keepdims=True)

        # Normalize each token vector
        X_normalized = (X - mean) / np.sqrt(variance + self.eps)

        # Apply the learnable scale and shift
        output = X_normalized * self.gamma + self.beta

        return output


class FeedForward:
    def __init__(self, embedding_dim, hidden_dim):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # First linear layer
        self.W1 = np.random.randn(self.embedding_dim, self.hidden_dim) * 0.01
        self.b1 = np.zeros(self.hidden_dim) * 0.01

        # Second linear layer
        self.W2 = np.random.randn(self.hidden_dim, self.embedding_dim) * 0.01
        self.b2 = np.zeros(self.embedding_dim) * 0.01

    def forward(self, X):
        # First linear layer
        Z = X @ self.W1 + self.b1

        # ReLU
        A = np.maximum(Z, 0)

        # Second linear layer
        output = A @ self.W2 + self.b2

        return output


class TransformerBlock:
    def __init__(
        self,
        embedding_dim,
        attention_dim,
        num_heads,
        hidden_dim,
    ):
        # Multi-head attention
        self.attention = MultiHeadAttention(embedding_dim, attention_dim, num_heads)

        # First layer normalization
        self.norm1 = LayerNorm(embedding_dim)

        # Feed-forward network
        self.feed_forward = FeedForward(embedding_dim, hidden_dim)

        # Second layer normalization
        self.norm2 = LayerNorm(embedding_dim)

    def forward(self, X):
        # Multi-head attention
        M = self.attention.forward(X)

        # First residual connection and normalization
        Z = self.norm1.forward(X + M)

        # Feed-forward network
        F = self.feed_forward.forward(Z)

        # Second residual connection and normalization
        output = self.norm2.forward(Z + F)

        return output
