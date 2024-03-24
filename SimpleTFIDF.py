import math
import numpy as np


# Implements a simple TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer.
class SimpleTFIDFVectorizer:
    def __init__(self, max_features=None):
        self.max_features = max_features
        self.vocab = {}
        self.idf = {}

    # Fits the vectorizer to the documents and transforms the documents into TF-IDF vectors in a single step.
    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)

    # Computes the IDF values for all terms in the documents.
    def fit(self, documents):
        term_df = {}
        total_documents = len(documents)

        # Build document frequency (DF) for each term
        for doc in documents:
            seen_terms = set()
            for term in doc.split():
                if term not in seen_terms:
                    term_df[term] = term_df.get(term, 0) + 1
                    seen_terms.add(term)

        # Calculate IDF for each term
        self.idf = {term: math.log((1 + total_documents) / (1 + df)) + 1 for term, df in term_df.items()}

        # Handle max_features
        if self.max_features:
            # Sort terms by IDF values, prioritizing terms with higher IDF (rarer across documents)
            sorted_terms = sorted(self.idf.items(), key=lambda item: item[1], reverse=True)
            selected_terms = sorted_terms[:self.max_features]
            self.vocab = {term: i for i, (term, _) in enumerate(selected_terms)}
        else:
            self.vocab = {term: i for i, term in enumerate(self.idf.keys())}

    # Converts the list of documents into a matrix of TF-IDF features.
    # It uses the IDF values computed in the fit step and term frequencies within each document to compute the TF-IDF scores.
    def transform(self, documents):
        tfidf_matrix = []

        for doc in documents:
            doc_vector = [0] * len(self.vocab)
            term_tf = {}

            words = doc.split()
            total_terms = len(words)
            for term in words:
                if term in self.vocab:
                    term_tf[term] = term_tf.get(term, 0) + 1 / total_terms

            for term, tf in term_tf.items():
                index = self.vocab.get(term)
                if index is not None:
                    idf = self.idf[term]
                    doc_vector[index] = tf * idf

            tfidf_matrix.append(doc_vector)

        return np.array(tfidf_matrix)
    
    # Computes the cosine similarity between two vectors
    # This measure evaluates the cosine of the angle between two vectors projected in a multi-dimensional space,
    # serving as an indication of similarity between two document vectors.
    def cosine_similarity(self, vec1, vec2):
        len_diff = abs(len(vec1) - len(vec2))
        if len(vec1) < len(vec2):
            vec1 = np.pad(vec1, (0, len_diff), 'constant')
        elif len(vec2) < len(vec1):
            vec2 = np.pad(vec2, (0, len_diff), 'constant')

        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0
        else:
            return dot_product / (norm_vec1 * norm_vec2)



