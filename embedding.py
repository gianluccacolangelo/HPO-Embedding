import torch
import torch.nn as nn
import pickle
from typing import List


class HPOEmbedder:
    def __init__(
        self, embedding_weights_path="pre_trained_embedding_weights.pth", device=None
    ):
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

        # Open phenotypes vocabulary
        with open("phen_vocabulary.pkl", "rb") as file:
            self.phen_vocabulary = pickle.load(file)

    def _phen_to_idx(self, phen_list: List[str]) -> torch.Tensor:
        """
        This function convert any list of phens to a tensor of indices
        ready to feed the embedding layer!
        """
        indices = [self.phen_vocabulary[phenotype] for phenotype in phen_list]
        return torch.Tensor(indices).type(torch.long)

    def get_embedding(self, phen_list: List[str]) -> torch.Tensor:
        # Convert phenotype list to tensor of indices
        input_indices = self._phen_to_idx(phen_list).to(self.device)
        # Get the embedding
        embedding = self.embedding_layer(input_indices)
        return embedding

## TODO:
[] que no trackee gradiente
[] preguntar por otras optimizaciones del estilo
[] que el output sea un tensor en numpy
