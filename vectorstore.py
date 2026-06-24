"""In-memory vector store backed by Ollama embeddings.

Chunks raw text, embeds chunks with the Ollama embedding model, and supports
cosine-similarity retrieval. Small enough to keep entirely in NumPy; optionally
persisted to a ``.npz`` file.
"""

import numpy as np

from config import (
    CHUNK_OVERLAP,
    CHUNK_WORDS,
    EMBED_MODEL,
    EMBED_TIMEOUT,
    get_client,
)


def chunk_text(text, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    """Split text into overlapping word windows.

    Args:
        text: The source text.
        size: Maximum number of words per chunk.
        overlap: Number of words shared between consecutive chunks.

    Returns:
        list[str]: Non-empty text chunks.
    """
    words = text.split()
    if not words:
        return []
    step = max(1, size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + size]).strip()
        if chunk:
            chunks.append(chunk)
        if start + size >= len(words):
            break
    return chunks


def embed_texts(texts, model=EMBED_MODEL):
    """Embed a list of texts with the Ollama embedding model.

    Args:
        texts: List of strings to embed.
        model: Ollama embedding model identifier.

    Returns:
        numpy.ndarray: Array of shape ``(len(texts), dim)`` (float32). Empty
        input yields an empty array.
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    resp = get_client(EMBED_TIMEOUT).embed(model=model, input=texts)
    vectors = resp.embeddings if hasattr(resp, "embeddings") else resp["embeddings"]
    return np.asarray(vectors, dtype=np.float32)


def _normalise(matrix):
    """L2-normalise the rows of a matrix, guarding against zero vectors.

    Args:
        matrix: 2-D array of row vectors.

    Returns:
        numpy.ndarray: Row-normalised copy.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class VectorStore:
    """A minimal cosine-similarity vector store over text chunks."""

    def __init__(self):
        """Initialise an empty store."""
        self.chunks = []
        self.vectors = None  # np.ndarray of shape (n, dim)

    def add(self, chunks):
        """Embed and add text chunks to the store.

        Args:
            chunks: List of text chunks to index.
        """
        if not chunks:
            return
        new_vecs = _normalise(embed_texts(chunks))
        self.chunks.extend(chunks)
        if self.vectors is None:
            self.vectors = new_vecs
        else:
            self.vectors = np.vstack([self.vectors, new_vecs])

    def search(self, query, k=8):
        """Return the ``k`` chunks most similar to a query string.

        Args:
            query: The natural-language query.
            k: Maximum number of chunks to return.

        Returns:
            list[tuple[str, float]]: ``(chunk, score)`` pairs, highest score first.
        """
        if self.vectors is None or len(self.chunks) == 0:
            return []
        q = _normalise(embed_texts([query]))[0]
        scores = self.vectors @ q
        top = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top]

    def save(self, path):
        """Persist the store (chunks + vectors) to a ``.npz`` file.

        Args:
            path: Destination file path.
        """
        if self.vectors is None:
            return
        np.savez_compressed(path, chunks=np.array(self.chunks, dtype=object),
                            vectors=self.vectors)

    def load(self, path):
        """Load a previously saved store from a ``.npz`` file.

        Args:
            path: Source file path.

        Returns:
            VectorStore: ``self``, for chaining.
        """
        data = np.load(path, allow_pickle=True)
        self.chunks = list(data["chunks"])
        self.vectors = data["vectors"]
        return self
