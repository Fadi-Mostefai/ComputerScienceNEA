import pandas as pd
import numpy as np
import re


# Manages text preprocessing tasks such as cleaning and vectorizing textual data.
class DataPreprocessor:
    def __init__(self):
        self.vocab = None

    # Removes HTML tags and non-alphabetic characters, then converts it to lowercase.
    # This standardizes the text for further processing.
    def clean_text(self, text):
        text = re.sub("<.*?>", " ", text)
        text = re.sub("[^a-zA-Z]", " ", text)
        return text.lower()

    # Converts a cleaned text into a vector representation based on the previously created vocabulary (self.vocab).
    # Each position in the vector corresponds to a term in the vocabulary, and the value at each position counts the occurrence of the term in the text.
    def vectorize_text(self, text):
        words = text.split()
        vector = np.zeros(len(self.vocab))
        for word in words:
            index = self.vocab.get(word)
            if index is not None:
                vector[index] += 1
        return vector

    # Loads question and answer data from specified paths, cleans the text, and merges them for processing.
    def load_and_prepare_data(self, questions_path, answers_path):
        questions_df = pd.read_excel(questions_path)
        answers_df = pd.read_excel(answers_path)

        merged_df = pd.merge(questions_df, answers_df, left_on='Id', right_on='ParentId', suffixes=("_q", "_a"))

        merged_df['cleaned_question'] = merged_df['Body_q'].apply(self.clean_text)
        merged_df['cleaned_answer'] = merged_df['Body_a'].apply(self.clean_text)

        merged_df['qa_pair'] = merged_df['cleaned_question'] + " " + merged_df['cleaned_answer']

        return merged_df['qa_pair'].tolist()




