# HPO-Embedding
This is a pre-trained embedding for HPO symptoms that is part of my master's degree thesis. You give it an HP:XXXXX code and as a result you have its corresponding point in a 30-dimensional space.


## Installation

Clone the repository and ensure you have PyTorch installed:

```bash
git clone <repository-url>
cd <repository-directory>
pip install torch
```

## Usage

Here's an example of how to use the `HPOEmbedder` class:

```python
if __name__ == "__main__":
    embedder = HPOEmbedder("pre_trained_embedding_weights.pth")
    input_indices = [1, 2, 3]  # Example input indices
    embedding = embedder.get_embedding(input_indices)
    print(embedding)
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

For any questions or issues, please contact [gianlucisnt@gmail.com].
