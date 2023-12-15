import pandas as pd
from LRML import LinearRegression
import random
from SimpleTFIDF import SimpleTFIDFVectorizer
import numpy as np


def train_test_split(data, labels, test_size=0.2, random_state=None):
    if random_state:
        random.seed(random_state)

    data_labels = list(zip(data, labels))
    random.shuffle(data_labels)

    split_index = int(len(data_labels) * (1 - test_size))

    train_data, train_labels = zip(*data_labels[:split_index])
    test_data, test_labels = zip(*data_labels[split_index:])

    return train_data, test_data, train_labels, test_labels


questions = pd.read_excel("New Questions.xlsx")
answers = pd.read_excel("New Answers.xlsx")

merged_df = pd.merge(questions, answers, left_on='Id', right_on='ParentId', suffixes=("_questions", "_answers"))

merged_df['QuestionText'] = merged_df['Title'] + ' ' + merged_df['Body_questions']
merged_df['AnswerText'] = merged_df['Body_answers']

X = merged_df['AnswerText']
y = merged_df['Score_answers']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf_vectorizer = SimpleTFIDFVectorizer(max_features=5000)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

model = LinearRegression
model.fit(tfidf_train, y_train)
predictions = model.predict(tfidf_test)


def mse(y_test, predictions):
    return np.mean((y_test-predictions)**2)


mse = mse(y_test, predictions)

print(f'Mean Square error: {mse}')

