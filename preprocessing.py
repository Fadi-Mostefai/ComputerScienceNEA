import pandas as pd
import numpy as np
import re

class DataPreprocessor:
    def __init__(self):
        self.vocab = None  # Will later hold the vocabulary (set of unique words) from the processed documents.

    # The following method cleans a given text string by removing HTML tags and non-alphabetic characters, then converts it to lowercase.
    # This standardizes the text for further processing.
    def clean_text(self, text):
        text = re.sub("<.*?>", " ", text)  # Remove HTML tags
        text = re.sub("[^a-zA-Z]", " ", text)  # Remove non-alphabetic characters
        return text.lower()

    # The following method converts a cleaned text into a vector representation based on the previously created vocabulary (self.vocab).
    # Each position in the vector corresponds to a term in the vocabulary, and the value at each position counts the occurrence of the term in the text.
    def vectorize_text(self, text):
        # Splits the text into words
        words = text.split()
        # Initializes a zero vector of the same length as the vocabulary
        vector = np.zeros(len(self.vocab))
        # Iterates over each word in the text. If the word is in the vocabulary, it increments the corresponding position in the vector.
        for word in words:
            index = self.vocab.get(word)
            if index is not None:
                vector[index] += 1
        return vector

    # The following method Loads questions and answers data from Excel files specified by questions_path and answers_path using pandas.
    def load_and_prepare_data(self, questions_path, answers_path):
        questions_df = pd.read_excel(questions_path)
        answers_df = pd.read_excel(answers_path)

        # Fill NaN values in both dataframes to ensure string operations do not fail
        questions_df.fillna('', inplace=True)
        answers_df.fillna('', inplace=True)

        # Merge questions and answers, ensuring no suffixes are required by dropping duplicate columns from answers
        answers_df.drop(columns=['Score', 'Body'], inplace=True)
        merged_df = pd.merge(questions_df, answers_df, left_on='Id', right_on='ParentId')

        # Preprocess the text
        merged_df['cleaned_text'] = merged_df['Title'] + ' ' + merged_df['Body']
        merged_df['cleaned_text'] = merged_df['cleaned_text'].apply(self.clean_text)

        # Creating a vocabulary from the cleaned text
        all_words = " ".join(merged_df['cleaned_text']).split()
        self.vocab = {word: i for i, word in enumerate(set(all_words))}

        # Vectorize the cleaned text
        features = np.array([self.vectorize_text(text) for text in merged_df['cleaned_text']])
        labels = np.array(merged_df['Score'].tolist())

        # 'features' contains the vectorized representation of each document
        # 'labels' contains the corresponding scores from the merged dataframe.
        return features, labels



