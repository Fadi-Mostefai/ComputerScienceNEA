import tkinter as tk
from tkinter import messagebox, simpledialog
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication
import sys
from GoogleAPI import GoogleGUI
from preprocessing import DataPreprocessor
from SimpleTFIDF import SimpleTFIDFVectorizer
from VAE import VAE


# Defines the main application window and its behavior
class MainApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Query Application")
        self.geometry("400x300")

        self.data_preprocessor = DataPreprocessor()
        self.tfidf_vectorizer = SimpleTFIDFVectorizer()
        self.glove_embeddings = self.load_glove_embeddings("glove.6B.50d.txt")
        self.model_path = "VAE_model"
        self.vae_model = None
        self.setup_initial_menu()

    # Loads word embeddings from a GloVe file
    def load_glove_embeddings(self, path):
        embeddings_dict = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embeddings_dict[word] = vector
        return embeddings_dict

    # Initializes the main menu
    def setup_initial_menu(self):
        tk.Button(self, text="Google Search", command=self.initiate_google_search).pack(pady=10)
        tk.Button(self, text="Neural Network Query", command=self.setup_nn_query_ui).pack(pady=10)

    # Closes the current GUI and opens the Google GUI
    def initiate_google_search(self):
        self.destroy()
        app = QApplication(sys.argv)
        api_key = "AIzaSyCbtBcWEqXQXjK4yaOkN24TIsnkgrsKxkg"
        cse_id = "42059fbb7eb1e44ad"
        mainWindow = GoogleGUI(api_key, cse_id)
        mainWindow.show()
        sys.exit(app.exec_())

    # Sets up the main menu for the neural network query
    def setup_nn_query_ui(self):
        self.destroy()
        nn_query_window = tk.Tk()
        nn_query_window.title("NN Query")
        nn_query_window.configure(bg="#FFC0CB")

        query_label = tk.Label(nn_query_window, text="Enter your query:", bg="#FFC0CB")
        query_label.pack(pady=5)
        query_entry = tk.Entry(nn_query_window, width=50)
        query_entry.pack(pady=5)

        submit_button = tk.Button(nn_query_window, text="Submit Query",
                                  command=lambda: self.neural_network_query(query_entry.get(), nn_query_window),
                                  bg="#FF69B4", fg="white", relief="flat")
        submit_button.pack(pady=10)

        train_model_button = tk.Button(nn_query_window, text="Train Model", command=self.train_model)
        train_model_button.pack(pady=10)

    # Trains the VAE model using data prepared through text preprocessing and TF-IDF vectorization
    def train_model(self):
        questions_path = "New Questions.xlsx"
        answers_path = "New Answers.xlsx"

        qa_pairs = self.data_preprocessor.load_and_prepare_data(questions_path, answers_path)

        vectorized_data = self.tfidf_vectorizer.fit_transform(qa_pairs)

        vectorized_data = np.array(vectorized_data)

        self.vae_model = VAE(input_dim=vectorized_data.shape[1], latent_dim=50, hidden_dims=[50, 50])
        self.vae_model.train(vectorized_data, epochs=100, batch_size=32, learning_rate=0.001, lambda_param=0.0001)

    # Finding the closest matching words in the GloVe embeddings space and the vae response vector
    def find_closest_embeddings(self, embedding, embeddings_dict, top_k=5):
        embedding = embedding.reshape(-1)

        cosine_similarities = {word: self.tfidf_vectorizer.cosine_similarity(embedding, embeddings_dict[word]) for word
                               in embeddings_dict}

        closest_embeddings = sorted(cosine_similarities, key=cosine_similarities.get, reverse=True)[:top_k]

        return closest_embeddings

    # Convert VAE-generated vectors back into text
    def vector_to_text(self, vae_response_vector, glove_embeddings):
        vae_response_vector = vae_response_vector.reshape(-1)
        closest_words = self.find_closest_embeddings(vae_response_vector, glove_embeddings, top_k=5)
        response_text = ' '.join(closest_words)
        return response_text

    # Handles the process of receiving a query, converting it into a form suitable for the VAE, generating a response,
    # and displaying it to the user.
    def neural_network_query(self, query, window):
        if not self.vae_model:
            messagebox.showwarning("Model not trained", "Please train the model first.")
            return
        processed_query = self.data_preprocessor.clean_text(query)
        query_vector = self.tfidf_vectorizer.transform([processed_query])[0]
        vae_response_vector = self.vae_model.generate_response(np.array([query_vector]))

        response_text = self.vector_to_text(vae_response_vector, self.glove_embeddings)

        messagebox.showinfo("Generated Response", response_text, parent=window)

        score = simpledialog.askinteger("Rate the Answer", "Please rate the answer on a scale of 0-100", parent=window,
                                        minvalue=0, maxvalue=100)

        if score is not None:
            print(f"User score: {score}")
            self.update_excel_files(query, response_text, score)
        else:
            print("User cancelled the rating.")

    # Stores the newly generated answer, the question, and the score into the excel files for questions and answers.
    def update_excel_files(self, question, answer, score):
        questions_df = pd.read_excel("New Questions.xlsx")
        answers_df = pd.read_excel("New Answers.xlsx")
        new_question_id = questions_df['Id'].max() + 1
        new_question_df = pd.DataFrame([{'Id': new_question_id, 'Score': score, 'Title': question, 'Body': question}])
        questions_df = pd.concat([questions_df, new_question_df], ignore_index=True)
        new_answer_id = answers_df['Id'].max() + 1
        new_answer_df = pd.DataFrame(
            [{'Id': new_answer_id, 'ParentId': new_question_id, 'Score': score, 'Body': answer}])
        answers_df = pd.concat([answers_df, new_answer_df], ignore_index=True)
        questions_df.to_excel("New Questions.xlsx", index=False)
        answers_df.to_excel("New Answers.xlsx", index=False)

    # Go back to the main menu
    def back_to_main_menu(self, root):
        root.destroy()
        main = MainApplication()
        main.mainloop()


if __name__ == "__main__":
    app = MainApplication()
    app.mainloop()
