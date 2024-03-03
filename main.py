import tkinter as tk
from tkinter import messagebox, simpledialog
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication
import sys
from LogisticRegression import LogisticRegression  # Placeholder for your existing import
from GoogleAPI import GoogleGUI  # Placeholder for your existing import
from preprocessing import DataPreprocessor  # Assuming this prepares data for VAE
from SimpleTFIDF import SimpleTFIDFVectorizer  # For text vectorization
from VAE import VAE  # Your VAE class


class MainApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Query Application")
        self.geometry("400x300")
        self.setup_initial_menu()

        # VAE and related initialization
        self.data_preprocessor = DataPreprocessor()
        self.tfidf_vectorizer = SimpleTFIDFVectorizer()  # Assume it's fit to your data

        self.model_initialised = False

    def lazy_load_model(self):
        if not self.model_initialised:
            self.train_model()
            self.model_initialised = True

    def train_model(self):
        questions_path = "New Questions.xlsx"
        answers_path  = "New Answers.xlsx"

        # Preprocess questions
        features, labels = self.data_preprocessor.load_and_prepare_data(questions_path, answers_path)

        tfidf_vectorizer = SimpleTFIDFVectorizer(max_features=1000)  # Adjust max_features as needed

        # Fit and transform data using TF-IDF vectorizer
        tfidf_features = tfidf_vectorizer.fit_transform(features)

        # Initialize and train VAE model
        self.vae_model = VAE(input_dim=tfidf_features.shape[1], latent_dim=64, hidden_dims=[256, 128])
        self.vae_model.train(tfidf_features, epochs=100, batch_size=32)

    def setup_initial_menu(self):
        tk.Button(self, text="Google Search", command=self.initiate_google_search).pack(pady=10)
        tk.Button(self, text="Neural Network Query", command=self.setup_nn_query_ui).pack(pady=10)

    def initiate_google_search(self):
        self.destroy()  # Close Tkinter window before PyQt5 execution
        # PyQt5 app is started in a separate process to avoid conflicts
        app = QApplication(sys.argv)
        api_key = "AIzaSyCbtBcWEqXQXjK4yaOkN24TIsnkgrsKxkg"
        cse_id = "42059fbb7eb1e44ad"
        mainWindow = GoogleGUI(api_key, cse_id)
        mainWindow.show()
        sys.exit(app.exec_())

    def setup_nn_query_ui(self):
        self.destroy()  # Close the main window
        nn_query_window = tk.Tk()
        nn_query_window.title("NN Query")

        tk.Label(nn_query_window, text="Enter your query:").pack(pady=5)
        query_entry = tk.Entry(nn_query_window, width=50)
        query_entry.pack(pady=5)

        tk.Button(nn_query_window, text="Submit Query",
                  command=lambda: self.neural_network_query(query_entry.get(), nn_query_window)).pack(pady=10)

        tk.Button(nn_query_window, text="Back to Main Menu",
                  command=lambda: self.back_to_main_menu(nn_query_window)).pack(side=tk.LEFT, padx=(20, 10), pady=20)

        tk.Button(nn_query_window, text="Exit", command=lambda: self.exit_app(nn_query_window)).pack(side=tk.RIGHT,
                                                                                                     padx=(10, 20),
                                                                                                     pady=20)
    def change_window(self, gui_setup_func, title):
        self.root.withdraw()
        new_window = tk.Toplevel()
        new_window.title(title)
        new_window.geometry("800x600")
        if gui_setup_func == GoogleGUI:
            gui_setup_func("AIzaSyCbtBcWEqXQXjK4yaOkN24TIsnkgrsKxkg", "42059fbb7eb1e44ad", self.back_to_main_menu)
        else:
            gui_setup_func(new_window)

    def neural_network_query(self, query, window):
        # Vectorize the query
        processed_query = self.data_preprocessor.clean_text(query)  # Preprocess the text
        query_vector = self.tfidf_vectorizer.transform([processed_query])[0]  # Vectorize the query
        # Generate a response using the VAE
        response = self.vae_model.generate_response(np.array([query_vector]))  # Assuming the VAE expects a numpy array
        messagebox.showinfo("Generated Response", str(response), parent=window)  # Display the response


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
    app = MainApplication()
    app.mainloop()



