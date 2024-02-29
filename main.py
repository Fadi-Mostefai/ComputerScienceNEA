import tkinter as tk
from tkinter import messagebox, simpledialog
import pandas as pd
import numpy as np
from LogisticRegression import LogisticRegression
from GoogleAPI import GoogleGUI
from preprocessing import DataPreprocessor
from SimpleTFIDF import SimpleTFIDFVectorizer


class MainApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("Query Application")
        self.setup_initial_menu()
        self.data_preprocessor = DataPreprocessor()
        self.model = LogisticRegression(learning_rate=0.01, n_iters=1000, lambda_param=0.1)
        self.tfidf_vectorizer = SimpleTFIDFVectorizer()
        self.model_initialised = False

    def lazy_load_model(self):
        if not self.model_initialised:
            self.model = LogisticRegression(learning_rate=0.01, n_iters=1000, lambda_param=0.1)
            self.train_model()
            self.model_initialised = True

    def train_model(self):
        questions_df = pd.read_excel("New Questions.xlsx")
        answers_df = pd.read_excel("New Answers.xlsx")

        # Preprocess questions
        questions_df['processed_text'] = questions_df['Body'].apply(self.data_preprocessor.clean_text)

        # Fit and transform questions using TFIDF Vectorizer
        self.tfidf_vectorizer.fit(questions_df['processed_text'])
        X = self.tfidf_vectorizer.transform(questions_df['processed_text'])

        # Assuming 'Score' is used as a binary label for simplicity
        y = questions_df['Score'].values

        # Train the logistic regression model
        self.model.fit(X, y)

        # Store answers for later use
        self.answers_df = answers_df

    def setup_initial_menu(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        self.root.geometry("400x200")

        button_google_search = tk.Button(self.root, text="Google Search", command=self.initiate_google_search)
        button_google_search.pack(pady=10)

        button_nn_query = tk.Button(self.root, text="Neural Network Query", command=lambda: self.setup_nn_query_ui())
        button_nn_query.pack(pady=10)

    def initiate_google_search(self):
        self.change_window(GoogleGUI, "Google Search")

    def setup_nn_query_ui(self):
        self.change_window(self.nn_query_gui_setup, "Neural Network Query")

    def change_window(self, gui_setup_func, title):
        self.root.withdraw()
        new_window = tk.Toplevel()
        new_window.title(title)
        new_window.geometry("800x600")
        if gui_setup_func == GoogleGUI:
            gui_setup_func(new_window, "AIzaSyCbtBcWEqXQXjK4yaOkN24TIsnkgrsKxkg", "42059fbb7eb1e44ad", self.back_to_main_menu)
        else:
            gui_setup_func(new_window)

    def nn_query_gui_setup(self, root):
        self.lazy_load_model()  # Load and train the model only when needed
        self.change_window(self.nn_query_gui_setup, "Neural Network Query")

        query_label = tk.Label(root, text="Enter your query:")
        query_label.pack(pady=5)

        query_entry = tk.Entry(root, width=50)
        query_entry.pack(pady=5)

        query_button = tk.Button(root, text="Submit Query",
                                 command=lambda: self.neural_network_query(query_entry.get(), root))
        query_button.pack(pady=10)

        back_button = tk.Button(root, text="Back to Main Menu", command=lambda: self.back_to_main_menu(root))
        back_button.pack(side=tk.LEFT, padx=(20, 10), pady=20)

        exit_button = tk.Button(root, text="Exit", command=lambda: self.exit_app(root))
        exit_button.pack(side=tk.RIGHT, padx=(10, 20), pady=20)

    def neural_network_query(self, query, root):

        processed_query = self.data_preprocessor.clean_text(query)
        query_vector = self.tfidf_vectorizer.transform([processed_query])[0]
        prediction = self.model.predict(np.array([query_vector]))[0]

        # Use prediction to find the best answer
        selected_answer = self.select_answer_based_on_query(query)
        messagebox.showinfo("Predicted Answer", selected_answer)

    def select_answer_based_on_query(self, query):
        documents = pd.read_excel("New Questions.xlsx")['Title'].tolist() + [query]
        tfidf_matrix = SimpleTFIDFVectorizer.fit_transform(documents)
        query_vector = tfidf_matrix[-1]
        question_vectors = tfidf_matrix[:-1]
        similarities = [SimpleTFIDFVectorizer.cosine_similarity(query_vector, vec) for vec in question_vectors]
        idx = similarities.index(max(similarities))
        questions_df = pd.read_excel("New Questions.xlsx")
        answers_df = pd.read_excel("New Answers.xlsx")
        question_id = questions_df.iloc[idx]['Id']
        relevant_answers = answers_df[(answers_df['ParentId'] == question_id) & (answers_df['Score'] > 0)]
        if not relevant_answers.empty:
            top_answer = relevant_answers.loc[relevant_answers['Score'].idxmax()]['Body']
        else:
            top_answer = "Sorry, I couldn't find a relevant answer for your query."
        return top_answer

    def update_excel_files(self, question, answer, score):
        questions_df = pd.read_excel("New Questions.xlsx")
        answers_df = pd.read_excel("New Answers.xlsx")
        new_question_id = questions_df['Id'].max() + 1 if not questions_df.empty else 1
        new_question_df = pd.DataFrame([{'Id': new_question_id, 'Score': score, 'Title': question, 'Body': question}])
        questions_df = pd.concat([questions_df, new_question_df], ignore_index=True)
        new_answer_id = answers_df['Id'].max() + 1 if not answers_df.empty else 1
        new_answer_df = pd.DataFrame(
            [{'Id': new_answer_id, 'ParentId': new_question_id, 'Score': score, 'Body': answer}])
        answers_df = pd.concat([answers_df, new_answer_df], ignore_index=True)
        questions_df.to_excel("New Questions.xlsx", index=False)
        answers_df.to_excel("New Answers.xlsx", index=False)

    def back_to_main_menu(self, root):
        root.destroy()
        main_root = tk.Tk()
        MainApplication(main_root)
        main_root.mainloop()

    def exit_app(self, root):
        root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = MainApplication(root)
    root.mainloop()



