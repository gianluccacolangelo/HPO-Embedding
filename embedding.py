import torch
import torch.nn as nn
import pickle
from typing import List
import numpy as np


class PhenotypeNotFoundError(Exception):
    """Custom exception for when a phenotype is not found in the vocabulary."""

    pass


class HPOEmbedder:
    def __init__(
        self, embedding_weights_path="pre_trained_embedding_weights.pth", device=None
    ):
        # Determine the device
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load the embedding weights
        embedding_weights = torch.load(embedding_weights_path, map_location=self.device)

        # Create an Embedding layer using the extracted weights
        embedding_dim = embedding_weights.shape[1]
        num_embeddings = embedding_weights.shape[0]
        self.embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding_layer.weight = nn.Parameter(
            embedding_weights, requires_grad=False
        )
        self.embedding_layer.to(self.device)

        # Set the model to evaluation mode
        self.embedding_layer.eval()

        # Open phenotypes vocabulary
        with open("phen_vocabulary.pkl", "rb") as file:
            self.phen_vocabulary = pickle.load(file)

    @torch.no_grad()
    def _phen_to_idx(self, phen_list: List[str]) -> torch.Tensor:
        """
        This function converts any list of phens to a tensor of indices
        ready to feed the embedding layer!

        Raises:
        PhenotypeNotFoundError: If a phenotype is not found in the vocabulary.

        Returns:
        torch.Tensor: A 1D tensor of shape (len(phen_list),)
        """
        try:
            indices = [self.phen_vocabulary[phenotype] for phenotype in phen_list]
        except KeyError as e:
            raise PhenotypeNotFoundError(f"Phenotype not found in vocabulary: {str(e)}")
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def get_embedding(self, phen_list: List[str]) -> torch.Tensor:
        """
        Get the embedding for a list of phenotypes.

        Returns:
        torch.Tensor: A 2D tensor of shape (len(phen_list), embedding_dim)
        """
        # Convert phenotype list to tensor of indices
        input_indices = self._phen_to_idx(phen_list)
        # Get the embedding
        embedding = self.embedding_layer(input_indices)
        return embedding

    def get_embedding_numpy(self, phen_list: List[str]) -> np.ndarray:
        """
        Get the embedding for a list of phenotypes as a numpy array.

        Returns:
        np.ndarray: A 2D numpy array of shape (len(phen_list), embedding_dim)
        """
        embedding = self.get_embedding(phen_list)
        return embedding.cpu().numpy()
