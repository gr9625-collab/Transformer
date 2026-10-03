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
    def __init__(self, embedding_dim, attention_dim, causal=True):
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.causal = causal

        self.W_Q = np.random.randn(embedding_dim, attention_dim) * 0.01
        self.W_K = np.random.randn(embedding_dim, attention_dim) * 0.01
        self.W_V = np.random.randn(embedding_dim, attention_dim) * 0.01

    def forward(self, X):
        self.X = X

        # Calculate Q, K, V
        self.Q = X @ self.W_Q
        self.K = X @ self.W_K
        self.V = X @ self.W_V

        # Calculate scaled attention scores
        S = self.Q @ self.K.T / np.sqrt(self.attention_dim)

        # ========================================
        # CAUSAL MASK
        # ========================================

        if self.causal:
            n = X.shape[0]

            # Create mask for future positions
            mask = np.triu(
                np.ones((n, n), dtype=bool),
                k=1,
            )

            # Set future attention scores to -infinity
            S[mask] = -np.inf

        # ========================================
        # SOFTMAX
        # ========================================

        # Numerical stability
        S = S - np.max(S, axis=1, keepdims=True)

        exp_S = np.exp(S)

        self.A = exp_S / np.sum(
            exp_S,
            axis=1,
            keepdims=True,
        )

        # Weighted combination of values
        H = self.A @ self.V

        return H

    def backward(self, grad):
        # grad = dL/dH

        # ========================================
        # WEIGHTED VALUE COMBINATION H = AV
        # ========================================

        # Calculate dL/dA
        dL_dA = grad @ self.V.T

        # Calculate dL/dV
        dL_dV = self.A.T @ grad

        # ========================================
        # SOFTMAX A = softmax(S)
        # ========================================

        # Calculate dL/dS
        dL_dS = self.A * (dL_dA - np.sum(dL_dA * self.A, axis=1, keepdims=True))

        # ========================================
        # ATTENTION SCORES S = QK^T / sqrt(d_a)
        # ========================================

        # Calculate dL/dQ
        dL_dQ = dL_dS @ self.K / np.sqrt(self.attention_dim)

        # Calculate dL/dK
        dL_dK = dL_dS.T @ self.Q / np.sqrt(self.attention_dim)

        # ========================================
        # Q = XW_Q
        # K = XW_K
        # V = XW_V
        # ========================================

        # Calculate dL/dW_Q
        self.dL_dW_Q = self.X.T @ dL_dQ

        # Calculate dL/dW_K
        self.dL_dW_K = self.X.T @ dL_dK

        # Calculate dL/dW_V
        self.dL_dW_V = self.X.T @ dL_dV

        # Combine the three X gradients
        dL_dX = dL_dQ @ self.W_Q.T + dL_dK @ self.W_K.T + dL_dV @ self.W_V.T

        return dL_dX


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
        self.concat = outputs[0]
        for i in range(1, len(outputs)):
            self.concat = np.hstack((self.concat, outputs[i]))

        # Apply W_O
        MHA = self.concat @ self.W_O

        return MHA

    def backward(self, grad):
        # grad = dL/dMHA

        # ========================================
        # OUTPUT PROJECTION: MHA = concat @ W_O
        # ========================================

        # Calculate dL/dW_O
        self.dL_dW_O = self.concat.T @ grad

        # Calculate dL/dconcat
        dL_dconcat = grad @ self.W_O.T

        # ========================================
        # SPLIT GRADIENT BETWEEN HEADS
        # ========================================

        # Split dL/dconcat into one gradient
        # of size attention_dim for each head
        head_grads = []
        for i in range(self.num_heads):
            head_grad = dL_dconcat[
                :, i * self.attention_dim : (i + 1) * self.attention_dim
            ]
            head_grads.append(head_grad)

        # ========================================
        # BACKPROPAGATE THROUGH EACH HEAD
        # ========================================

        # Pass each gradient through its corresponding
        # attention head and collect its dL/dX
        head_dXs = [
            head.backward(head_grad) for head, head_grad in zip(self.heads, head_grads)
        ]

        # ========================================
        # COMBINE X GRADIENTS
        # ========================================

        # Every head received the same X, so add
        # their contributions to dL/dX
        dL_dX = 0
        for head_dX in head_dXs:
            dL_dX += head_dX

        return dL_dX


class LayerNorm:
    def __init__(self, embedding_dim, eps=1e-5):
        self.embedding_dim = embedding_dim
        self.eps = eps

        # Learnable scale and shift parameters
        self.gamma = np.ones(embedding_dim)
        self.beta = np.zeros(embedding_dim)

    def forward(self, X):
        # Save X for backpropagation
        self.X = X

        # Calculate the mean of each token vector
        self.mean = np.mean(X, axis=1, keepdims=True)

        # Centre each token vector
        self.X_centered = X - self.mean

        # Calculate the variance of each token vector
        self.variance = np.mean(
            self.X_centered**2,
            axis=1,
            keepdims=True,
        )

        # Calculate the standard deviation
        self.std = np.sqrt(self.variance + self.eps)

        # Normalize each token vector
        self.X_normalized = self.X_centered / self.std

        # Apply the learnable scale and shift
        output = self.X_normalized * self.gamma + self.beta

        return output

    def backward(self, grad):
        # grad = dL/doutput

        # ========================================
        # SCALE AND SHIFT
        # ========================================

        # Calculate dL/dgamma
        self.dL_dgamma = np.sum(grad * self.X_normalized, axis=0)

        # Calculate dL/dbeta
        self.dL_dbeta = np.sum(grad, axis=0)

        # ========================================
        # NORMALIZATION
        # ========================================

        # Calculate dL/dX
        G_X = grad * self.gamma

        dL_dX = (
            1
            / self.std
            * (
                G_X
                - np.mean(G_X, axis=1, keepdims=True)
                - self.X_normalized
                * np.mean(
                    G_X * self.X_normalized,
                    axis=1,
                    keepdims=True,
                )
            )
        )

        return dL_dX


class FeedForward:
    def __init__(self, embedding_dim, hidden_dim):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # First linear layer
        self.W1 = np.random.randn(self.embedding_dim, self.hidden_dim) * 0.01
        self.b1 = np.zeros(self.hidden_dim)

        # Second linear layer
        self.W2 = np.random.randn(self.hidden_dim, self.embedding_dim) * 0.01
        self.b2 = np.zeros(self.embedding_dim)

    def forward(self, X):
        # Save X for backpropagation
        self.X = X

        # First linear layer
        self.Z = X @ self.W1 + self.b1

        # ReLU
        self.A = np.maximum(self.Z, 0)

        # Second linear layer
        output = self.A @ self.W2 + self.b2

        return output

    def backward(self, grad):
        # grad = dL/doutput

        # ========================================
        # SECOND LINEAR LAYER
        # ========================================

        # Calculate dL/dW2
        self.dL_dW2 = self.A.T @ grad

        # Calculate dL/db2
        self.dL_db2 = np.sum(grad, axis=0)

        # Calculate dL/dA
        dL_dA = grad @ self.W2.T

        # ========================================
        # RELU
        # ========================================

        # Calculate dL/dZ
        dL_dZ = dL_dA * (self.Z > 0)

        # ========================================
        # FIRST LINEAR LAYER
        # ========================================

        # Calculate dL/dW1
        self.dL_dW1 = self.X.T @ dL_dZ

        # Calculate dL/db1
        self.dL_db1 = np.sum(dL_dZ, axis=0)

        # Calculate dL/dX
        dL_dX = dL_dZ @ self.W1.T

        # Return the gradient with respect to the input
        return dL_dX


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

    def backward(self, grad):
        # grad = dL/doutput

        # ========================================
        # SECOND LAYER NORM
        # ========================================

        # Backpropagate through norm2
        grad = self.norm2.backward(grad)

        # ========================================
        # SECOND RESIDUAL CONNECTION: Z + F
        # ========================================

        # Backpropagate through feed-forward and combine with residual
        grad = grad + self.feed_forward.backward(grad)

        # ========================================
        # FIRST LAYER NORM
        # ========================================

        # Backpropagate through norm1
        grad = self.norm1.backward(grad)

        # ========================================
        # FIRST RESIDUAL CONNECTION: X + M
        # ========================================

        # Backpropagate through attention
        grad = grad + self.attention.backward(grad)

        return grad


class VocabularyProjection:
    def __init__(self, embedding_dim, vocab_size):
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        # Initialise parameters
        self.W = np.random.randn(embedding_dim, vocab_size) * 0.01
        self.b = np.zeros(vocab_size)

    def forward(self, X):
        # Save X for backpropagation
        self.X = X

        # Project each token vector onto vocabulary space
        logits = X @ self.W + self.b

        return logits

    def backward(self, grad):
        # grad = dL/dlogits

        # Calculate dL/dW
        self.dL_dW = self.X.T @ grad

        # Calculate dL/db
        self.dL_db = np.sum(grad, axis=0)

        # Calculate dL/dX
        dL_dX = grad @ self.W.T

        return dL_dX


class CrossEntropyLoss:
    def forward(self, logits, targets):
        # logits has shape (n, vocab_size)
        # targets has shape (n,)

        self.targets = targets
        n = logits.shape[0]

        # ========================================
        # SOFTMAX
        # ========================================

        # Shift logits for numerical stability
        shifted = logits - np.max(logits, axis=1)


        # Exponentiate
        exponential = np.exp(shifted)


        # Calculate probabilities and save for backward
        self.probs = exponential / np.sum(exponential, axis=1)


        # ========================================
        # CROSS-ENTROPY LOSS
        # ========================================

        # Extract probability assigned to the
        # correct token at each position
        correct_probs = 


        # Calculate average negative log probability
        loss = 


        return loss

    def backward(self):
        # ========================================
        # SOFTMAX + CROSS-ENTROPY BACKWARD
        # ========================================

        n = self.probs.shape[0]

        # Start with P
        grad = self.probs.copy()

        # Subtract 1 from the probability corresponding
        # to the correct token in each row


        # Account for averaging over n tokens


        # grad = dL/dlogits
        return grad
