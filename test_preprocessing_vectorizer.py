import unittest
from preprocessing import DataPreprocessor
from SimpleTFIDF import SimpleTFIDFVectorizer
import numpy as np

class TestDataPreprocessingAndVectorization(unittest.TestCase):

    def setUp(self):
        self.data_preprocessor = DataPreprocessor()
        self.tfidf_vectorizer = SimpleTFIDFVectorizer()

        # Example text
        self.sample_texts = ["<p>Hello World!</p>", "<div>Python programming</div>"]
        self.cleaned_texts = [" hello world  ", " python programming "]

        # Manually creating a simple vocabulary and IDF for testing
        self.vocab = {'hello': 0, 'world': 1, 'python': 2, 'programming': 3}
        self.idf = {'hello': 1.0, 'world': 1.0, 'python': 1.0, 'programming': 1.0}
        self.tfidf_vectorizer.vocab = self.vocab
        self.tfidf_vectorizer.idf = self.idf

    def test_clean_text(self):
        for input_text, expected_output in zip(self.sample_texts, self.cleaned_texts):
            cleaned_text = self.data_preprocessor.clean_text(input_text)
            self.assertEqual(cleaned_text, expected_output)

    def test_vectorization(self):
        # Assuming every word in the vocab appears once in the document for simplicity
        expected_vector = np.array([0.25, 0.25, 0.25, 0.25])

        # Manually setting the vocabulary and idf values to avoid the fit step
        self.tfidf_vectorizer.vocab = self.vocab
        self.tfidf_vectorizer.idf = self.idf

        # Combining the cleaned texts to simulate a document
        combined_cleaned_text = " ".join(self.cleaned_texts)
        vectorized_text = self.tfidf_vectorizer.transform([combined_cleaned_text])

        # Since transform returns a list of vectors, we extract the first (and only) vector
        np.testing.assert_array_almost_equal(vectorized_text[0], expected_vector)

unittest.main()
