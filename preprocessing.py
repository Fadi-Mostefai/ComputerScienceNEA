import pandas as pd
import numpy as np
import re

class DataPreprocessor:
    def __init__(self):
        self.vocab = None

    def clean_text(self, text):
        text = re.sub("<.*?>", " ", text)  # Remove HTML tags
        text = re.sub("[^a-zA-Z]", " ", text)  # Remove non-alphabetic characters
        return text.lower()

    def vectorize_text(self, text):
        words = text.split()
        vector = np.zeros(len(self.vocab))
        for word in words:
            index = self.vocab.get(word)
            if index is not None:
                vector[index] += 1
        return vector

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

        return features, labels



