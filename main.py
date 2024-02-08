import tkinter as tk
from tkinter import simpledialog, messagebox, Frame, Label, Entry, Button, Listbox, Toplevel
import pandas as pd
from LogisticRegression import LogisticRegression  # Ensure this is correctly implemented
from GoogleAPI import GoogleSearch  # Ensure this is correctly implemented
from preprocessing import DataPreprocessor  # Ensure this is correctly implemented
from SimpleTFIDF import SimpleTFIDFVectorizer  # Ensure this class is updated as discussed


class MainApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("Query Application")

        # Initialize data preprocessor
        self.data_preprocessor = DataPreprocessor()

        # Load and prepare data
        self.questions_path = 'New Questions.xlsx'
        self.answers_path = 'New Answers.xlsx'
        self.X, self.y = self.data_preprocessor.load_and_prepare_data(self.questions_path, self.answers_path)

        # Initialize and fit the logistic regression model with the prepared data
        self.model = LogisticRegression()
        self.model.fit(self.X, self.y)

        # Initialize Google Search with API keys
        self.google_search = GoogleSearch("AIzaSyCbtBcWEqXQXjK4yaOkN24TIsnkgrsKxkg", "42059fbb7eb1e44ad")

        # When initializing SimpleTFIDFVectorizer in main.py, set max_features to limit vocabulary size
        self.tfidf_vectorizer = SimpleTFIDFVectorizer(max_features=1000)  # Example: limit to top 1000 terms

        # Setup UI components
        self.setup_ui()

    def setup_ui(self):
        # UI setup for Google Search
        self.google_frame = Frame(self.root)
        self.google_label = Label(self.google_frame, text="Google Search")
        self.google_label.pack(side=tk.TOP)
        self.google_search_entry = Entry(self.google_frame, width=50)
        self.google_search_entry.pack(side=tk.LEFT)
        self.google_search_button = Button(self.google_frame, text="Search", command=self.perform_google_search)
        self.google_search_button.pack(side=tk.LEFT)
        self.google_frame.pack(pady=10)

        # UI setup for Neural Network Query
        self.nn_frame = Frame(self.root)
        self.nn_label = Label(self.nn_frame, text="Neural Network Query")
        self.nn_label.pack(side=tk.TOP)
        self.nn_query_entry = Entry(self.nn_frame, width=50)
        self.nn_query_entry.pack(side=tk.LEFT)
        self.nn_query_button = Button(self.nn_frame, text="Query", command=self.neural_network_query)
        self.nn_query_button.pack(side=tk.LEFT)
        self.nn_frame.pack(pady=10)

    def perform_google_search(self):
        query = self.google_search_entry.get()
        results = self.google_search.search(query)
        self.display_results(results)

    def display_results(self, results):
        result_window = Toplevel(self.root)
        result_window.title("Search Results")
        listbox = Listbox(result_window, width=100, height=20)
        listbox.pack(fill=tk.BOTH, expand=True)
        for result in results:
            listbox.insert(tk.END, result['title'])

    def neural_network_query(self):
        query = self.nn_query_entry.get()
        selected_answer = self.select_answer_based_on_query(query)
        messagebox.showinfo("Predicted Answer", selected_answer)

        user_score = simpledialog.askinteger("Rate the Answer", "Please rate the answer out of 100:", minvalue=0,
                                             maxvalue=100)

        if user_score is not None:
            self.update_excel_files(query, selected_answer, user_score)

    def select_answer_based_on_query(self, query):
        # Fit TF-IDF Vectorizer with questions and the query
        documents = pd.read_excel(self.questions_path)['Title'].tolist() + [query]
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)

        # Calculate cosine similarity between the query and all questions
        query_vector = tfidf_matrix[-1]
        question_vectors = tfidf_matrix[:-1]
        similarities = [self.tfidf_vectorizer.cosine_similarity(query_vector, vec) for vec in question_vectors]

        # Adjust similarity scores based on a threshold or modify the selection criteria as needed

        # Find the index of the most similar question
        idx = similarities.index(max(similarities))

        # Load questions and answers DataFrames
        questions_df = pd.read_excel(self.questions_path)
        answers_df = pd.read_excel(self.answers_path)

        # Filter answers for the selected question, excluding those with a score < 0
        question_id = questions_df.iloc[idx]['Id']
        relevant_answers = answers_df[(answers_df['ParentId'] == question_id) & (answers_df['Score'] > 0)]

        # If there are relevant answers, select the one with the highest score
        if not relevant_answers.empty:
            top_answer = relevant_answers.loc[relevant_answers['Score'].idxmax()]['Body']
        else:
            top_answer = "Sorry, I couldn't find a relevant answer for your query."

        return top_answer

    def update_excel_files(self, question, answer, score):
        # Load existing data
        questions_df = pd.read_excel(self.questions_path)
        answers_df = pd.read_excel(self.answers_path)

        # Generate a new question ID
        new_question_id = questions_df['Id'].max() + 1 if not questions_df.empty else 1

        # Create new question DataFrame and append
        new_question_df = pd.DataFrame([{'Id': new_question_id, 'Score': score, 'Title': question, 'Body': question}])
        questions_df = pd.concat([questions_df, new_question_df], ignore_index=True)

        # Generate a new answer ID
        new_answer_id = answers_df['Id'].max() + 1 if not answers_df.empty else 1

        # Create new answer DataFrame and append
        new_answer_df = pd.DataFrame(
            [{'Id': new_answer_id, 'ParentId': new_question_id, 'Score': score, 'Body': answer}])
        answers_df = pd.concat([answers_df, new_answer_df], ignore_index=True)

        # Save updated DataFrames back to Excel
        questions_df.to_excel(self.questions_path, index=False)
        answers_df.to_excel(self.answers_path, index=False)


if __name__ == "__main__":
    root = tk.Tk()
    app = MainApplication(root)
    root.mainloop()


