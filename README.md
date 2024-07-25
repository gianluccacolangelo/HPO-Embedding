```markdown
# HPO-Embedding

This is a pre-trained embedding for HPO (Human Phenotype Ontology) symptoms, developed as part of my master's degree thesis. It converts HPO terms (in the format HP:XXXXX) into corresponding points in a 200-dimensional space.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/HPO-Embedding.git
cd HPO-Embedding
```

### 2. Set up the Conda environment

First, make sure you have Conda installed. If not, you can download it from [here](https://docs.conda.io/en/latest/miniconda.html).

Then, create and activate a new Conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml -n your_preferred_name
conda activate your_preferred_name
```

## Usage

Here's a step-by-step guide to using the `HPOEmbedder` class:

1. First, import the necessary modules and instantiate the `HPOEmbedder`:

```python
from embedding import  HPOEmbedder

# Initialize the embedder
embedder = HPOEmbedder()
```

2. Prepare a list of HPO terms you want to embed:

```python
hpo_terms = ["HP:0000118", "HP:0000707", "HP:0000154"]
```

3. Get the embeddings:

```python
embeddings = embedder.get_embedding(hpo_terms)
print(f"Embeddings shape: {embeddings.shape}")
print(f"Embeddings: {embeddings}")
```

4. If you need the embeddings as a numpy array (e.g., for use with scikit-learn):

```python
embeddings_numpy = embedder.get_embedding_numpy(hpo_terms)
print(f"Numpy embeddings shape: {embeddings_numpy.shape}")
print(f"Numpy embeddings: {embeddings_numpy}")
```

### Full Example

```python
from hpo_embedder import HPOEmbedder

if __name__ == "__main__":
    # Initialize the embedder
    embedder = HPOEmbedder("pre_trained_embedding_weights.pth")

    # List of HPO terms
    hpo_terms = ["HP:0000118", "HP:0000707", "HP:0000154"]

    # Get embeddings
    embeddings = embedder.get_embedding(hpo_terms)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings: {embeddings}")

    # Get numpy embeddings
    embeddings_numpy = embedder.get_embedding_numpy(hpo_terms)
    print(f"Numpy embeddings shape: {embeddings_numpy.shape}")
    print(f"Numpy embeddings: {embeddings_numpy}")

    # Error handling example
    try:
        invalid_terms = ["HP:0000118", "INVALID:TERM"]
        embedder.get_embedding(invalid_terms)
    except PhenotypeNotFoundError as e:
        print(f"Error: {e}")
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

For any questions or issues, please contact [gianlucisnt@gmail.com].

```
