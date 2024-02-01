# main.py
import tkinter as tk
from tkinter import simpledialog, messagebox, Frame, Label, Entry, Button, Listbox, Toplevel
from LogisticRegression import LogisticRegression
from GoogleAPI import GoogleGUI, GoogleSearch
from preprocessing import DataPreprocessor

class MainApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("Query Application")

        # Initialize data preprocessor and logistic regression model
        self.data_preprocessor = DataPreprocessor()
        self.model = LogisticRegression()  # Assuming the model is already defined elsewhere

        # Load and prepare data
        questions_path = 'New Questions.xlsx'
        answers_path = 'New Answers.xlsx'
        self.X, self.y = self.data_preprocessor.load_and_prepare_data(questions_path, answers_path)

        # Initialize Google Search with API keys
        self.google_search = GoogleSearch("AIzaSyCbtBcWEqXQXjK4yaOkN24TIsnkgrsKxkg", "42059fbb7eb1e44ad")  # Replace with your actual keys

        self.setup_ui()

    def load_model(self):
        # Load and prepare your model here. This is a placeholder for the actual model loading
        X, y = self.data_preprocessor.load_and_prepare_data()  # Assuming this method exists and works as intended
        model = LogisticRegression()
        model.fit(X, y)
        return model

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
        results = self.google_search.search(query)  # Assuming this method returns a list of results
        self.display_results(results)

    def display_results(self, results):
        result_window = Toplevel(self.root)
        result_window.title("Search Results")
        listbox = Listbox(result_window, width=100, height=20)
        listbox.pack(fill=tk.BOTH, expand=True)
        for result in results:
            listbox.insert(tk.END, result['title'])  # Modify as needed based on actual result structure

    def neural_network_query(self):
        query = self.nn_query_entry.get()
        query_processed = self.data_preprocessor.vectorize_text(self.data_preprocessor.clean_text(query))  # Example processing steps
        prediction = self.model.predict(query_processed.reshape(1, -1))
        messagebox.showinfo("Neural Network Prediction", f"The predicted response is: {prediction}")

if __name__ == "__main__":
    root = tk.Tk()
    app = MainApplication(root)
    root.mainloop()

