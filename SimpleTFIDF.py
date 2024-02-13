import math

class SimpleTFIDFVectorizer:
    def __init__(self, max_features=None):
        self.max_features = max_features  # Optional max_features which limits the number of top features (terms) to consider based on their term frequency across the document corpus.
        self.vocab = {}  # A dictionary to store the vocabulary (unique terms) and their indices.
        self.idf = {}  # A dictionary to store the inverse document frequency values for each term.

    # A convenience method that first fits the vectorizer to the documents (building the vocabulary and computing IDF values) and then transforms the documents into their TF-IDF representation.
    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)

    # A method which processes a list of documents to build the model's vocabulary and compute the IDF for each term.
    def fit(self, documents):
        term_count = {}
        total_documents = len(documents)

        # Build term frequency (TF) and document frequency (DF) for each term
        for doc in documents:
            unique_terms = set(doc.split())
            for term in unique_terms:
                term_count[term] = term_count.get(term, 0) + 1

        # Calculate Inverse Document Frequency (IDF)
        self.idf = {term: math.log(total_documents / (1 + term_count[term])) for term in term_count}

        # Select top features based on max_features
        if self.max_features:
            sorted_terms = sorted(term_count.keys(), key=lambda x: term_count[x], reverse=True)
            selected_features = sorted_terms[:self.max_features]
            self.vocab = {term: index for index, term in enumerate(selected_features)}
        else:
            self.vocab = {term: index for index, term in enumerate(term_count)}

    # A method which transforms the documents into a matrix of TF-IDF features based on the previously built vocabulary and IDF values.
    def transform(self, documents):
        tfidf_matrix = []

        for doc in documents:
            doc_vector = [0] * len(self.vocab)

            # Calculate Term Frequency (TF) for each term in the document
            term_count = {}
            total_terms = len(doc.split())
            for term in doc.split():
                term_count[term] = term_count.get(term, 0) + 1

            # Calculate TF-IDF for each term in the document
            for term, index in self.vocab.items():
                tf = term_count.get(term, 0) / total_terms
                idf = self.idf.get(term, 0)
                doc_vector[index] = tf * idf

            tfidf_matrix.append(doc_vector)

        return tfidf_matrix

    # An additional method for calculating the cosine similarity between two vectors. This is useful for measuring the similarity between two documents.
    def cosine_similarity(self, vec1, vec2):
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm_a = math.sqrt(sum(a * a for a in vec1))
        norm_b = math.sqrt(sum(b * b for b in vec2))
        if norm_a == 0 or norm_b == 0:
            return 0
        else:
            return dot_product / (norm_a * norm_b)
        # Cosine similarity is calculated as the dot product of the vectors divided by the product of their norms (magnitudes).
