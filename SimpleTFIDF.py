import math

class SimpleTFIDFVectorizer:
    def __init__(self, max_features=None):
        self.max_features = max_features
        self.vocab = {}
        self.idf = {}

    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)

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
