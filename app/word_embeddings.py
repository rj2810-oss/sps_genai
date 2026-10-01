import spacy

# Only the static word vectors are needed, so skip the heavy pipeline components.
_EXCLUDE = ["tagger", "parser", "ner", "lemmatizer", "attribute_ruler", "senter"]


class WordEmbeddingModel:
    """Returns spaCy word vectors, as in the Module 2 Practical 3 notebook."""

    def __init__(self, model_name="en_core_web_lg"):
        self.nlp = spacy.load(model_name, exclude=_EXCLUDE)
        self.dimension = self.nlp.vocab.vectors_length

    def get_embedding(self, word):
        """Vector for a word. Multi-word input returns the average of its word vectors.
        Returns None when spaCy has no vector for the input."""
        doc = self.nlp(word.strip())
        if len(doc) == 0 or not doc.has_vector or not any(t.has_vector for t in doc):
            return None
        return doc.vector.tolist()

    def similarity(self, word1, word2):
        d1, d2 = self.nlp(word1.strip()), self.nlp(word2.strip())
        if not (d1.has_vector and d2.has_vector) or not d1.vector_norm or not d2.vector_norm:
            return None
        return float(d1.similarity(d2))
