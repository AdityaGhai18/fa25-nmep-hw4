from typing import Optional

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(
        self, num_heads: int, embedding_dim: int, qk_length: int, value_length: int
    ):
        """
        The Multi-Head Attention layer will take in Q, K, and V
        matrices and will output an attention matrix of shape (b, t, c).

        First, Q, K, and V should be projected to have
        a shape of (B, T, C) where C = num_heads * qk_length
        (OR value_length). You are then expected to split
        the C dimension into num_heads different heads, each
        with shape (B, T, vec_length).

        Next, you will compute the scaled dot-product attention
        between Q, K, and V.

        Finally, you will concatenate the heads and project the
        output to have a shape of (B, T, C).

        Check out the `masked_fill` method in PyTorch to help
        you implement the masking step!
        """
        super().__init__()

        self.num_heads = num_heads
        self.qk_length = qk_length
        self.value_length = value_length
        self.embedding_dim = embedding_dim #need embedding dim to project from

        self.Q = nn.Linear(embedding_dim, num_heads * qk_length)
        self.K = nn.Linear(embedding_dim, num_heads * qk_length)
        self.V = nn.Linear(embedding_dim, num_heads * value_length)

        # last linear layer after concat
        # name it output_projection to match usage in forward
        self.output_projection = nn.Linear(num_heads * value_length, embedding_dim)


        #raise NotImplementedError("Need to implement MHA layers")

    def split_heads(self, x: torch.Tensor, vec_length: int) -> torch.Tensor:
        """
        Split the C dimension of the input tensor into num_heads
        different heads, each with shape (B, T, vec_length).
        Hint: check out the `view` and 'permute` methods in PyTorch to help
        you reshape the tensor.

        Args:
            x: torch.Tensor of shape (B, T, C), where C = num_heads * vec_length
            vec_length: int, the length of the query/key/value vectors

        Returns:
            torch.Tensor of shape (B, num_heads, T, vec_length)
        """
        B, T, C = x.size()

        assert C // self.num_heads == vec_length, (
            "Input tensor does not have the correct shape for splitting."
        )

        x = x.view(B, T, self.num_heads, vec_length)
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, num_heads, T, vec_length)


        return x
    
        #raise NotImplementedError("Need to implement split_heads")

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine the num_heads different heads into a single tensor.
        Hint: check out the `contiguous` method in PyTorch to help
        you reshape the tensor.

        Args:
            x: torch.Tensor of shape (B, num_heads, T, vec_length)

        Returns:
            torch.Tensor of shape (B, T, num_heads * vec_length)
        """
        B, num_heads, T, vec_length = x.size()

        x = x.permute(0, 2, 1, 3).contiguous()  
        x = x.view(B, T, num_heads * vec_length)


        return x
        #raise NotImplementedError("Need to implement combine_heads")

    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the scaled dot-product attention given Q, K, and V.
        This is where the pad_mask and causal_mask are applied.

        Args:
            Q: torch.Tensor of shape (B, num_heads, T, qk_length)
            K: torch.Tensor of shape (B, num_heads, T, qk_length)
            V: torch.Tensor of shape (B, num_heads, T, value_length)
            mask: Optional boolean torch.Tensor, broadcastable to (B, num_heads, T, T).
        """

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.qk_length ** 0.5)  # this will give (B, num_heads, T, T)
        #transpose with -2, -1 to swap last two dims
        #so that K goes from (B, num_heads, T, qk_length) to (B, num_heads, qk_length, T)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = torch.softmax(scores, dim=-1)  # (B, num_heads, T, T)

        #this is our activation table looking thing
        out = torch.matmul(attention_weights, V)  # (B, num_heads, T, value_length)

        return out

        #raise NotImplementedError("Need to implement scaled_dot_product_attention")

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        The forward pass of the Multi-Head Attention layer.

        Args:
            Q: torch.Tensor of shape (B, T, C)
            K: torch.Tensor of shape (B, T, C)
            V: torch.Tensor of shape (B, T, C)
            mask: Optional torch.Tensor of shape (B, T, T) or None

        Returns:
            torch.Tensor of shape (B, T, C)
        """

        Q = self.Q(Q)
        K = self.K(K)
        V = self.V(V)

        Q_split = self.split_heads(Q, self.qk_length)
        K_split = self.split_heads(K, self.qk_length)
        V_split = self.split_heads(V, self.value_length)

        attn = self.scaled_dot_product_attention(Q_split, K_split, V_split, mask)

        combined = self.combine_heads(attn)
        out = self.output_projection(combined)
        return out
    
        #raise NotImplementedError("Need to implement forward pass of MHA")



class FeedForwardNN(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int):
        """
        The Feed-Forward Neural Network layer will take in
        an input tensor of shape (B, T, C) and will output
        a tensor of the same shape.

        The FFNN will have two linear layers, with a ReLU
        activation function in between.

        Args:
            hidden_dim: int, the size of the hidden layer
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)
        self.relu = nn.ReLU()
        # Define any layers you'll need in the forward pass
        #raise NotImplementedError("Need to implement FeedForwardNN layers")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the FeedForwardNN.
        """
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
        #raise NotImplementedError("Need to implement forward pass of FeedForwardNN")
