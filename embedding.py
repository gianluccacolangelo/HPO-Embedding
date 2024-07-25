import torch
import torch.nn as nn


class HPOEmbedder:
    def __init__(self, embedding_weights_path, device=None):
        # Determine the device
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load the embedding weights
        state_dict = torch.load(embedding_weights_path, map_location=self.device)

        # Assuming the embedding layer is named 'embedding.weight' in the state_dict
        embedding_key = "embedding.weight"
        embedding_weights = state_dict

        # Create an Embedding layer using the extracted weights
        embedding_dim = embedding_weights.shape[1]
        num_embeddings = embedding_weights.shape[0]
        self.embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding_layer.weight = nn.Parameter(embedding_weights)
        self.embedding_layer.to(self.device)

    def get_embedding(self, input_indices):
        # Ensure the input is a tensor
        input_tensor = torch.tensor(input_indices, dtype=torch.long).to(self.device)
        # Get the embedding
        embedding = self.embedding_layer(input_tensor)
        return embedding
