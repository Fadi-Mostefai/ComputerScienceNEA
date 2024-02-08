import tkinter as tk
from tkinter import ttk, scrolledtext
from tkhtmlview import HTMLLabel
import pandas as pd
import requests
from bs4 import BeautifulSoup
from googleapiclient.discovery import build

class GoogleSearch:
    def __init__(self, api_key, cse_id):
        self.api_key = api_key
        self.cse_id = cse_id

    def search(self, search_term, num_results=10):
        service = build("customsearch", "v1", developerKey=self.api_key)
        res = service.cse().list(q=search_term, cx=self.cse_id, num=num_results).execute()
        return res['items'] if 'items' in res else []

class GoogleGUI:
    def __init__(self, root, api_key, cse_id):
        self.root = root
        self.search_engine = GoogleSearch(api_key, cse_id)
        self.setup_ui()

    def setup_ui(self):
        self.label = tk.Label(self.root, text="Enter your query:")
        self.label.pack()

        self.query_entry = tk.Entry(self.root)
        self.query_entry.pack()

        self.search_button = tk.Button(self.root, text="Search", command=self.perform_search)
        self.search_button.pack()

        self.result_text = scrolledtext.ScrolledText(self.root, height=15, width=80, wrap=tk.WORD)
        self.result_text.pack()

    def perform_search(self):
        query = self.query_entry.get()
        search_results = self.search_engine.search(query)
        self.format_google_results(search_results)

    def format_google_results(self, results):
        self.result_text.delete(1.0, tk.END)
        for index, result in enumerate(results, start=1):
            title = result.get('title', 'No title')
            link = result.get('link', 'No link')
            snippet = result.get('snippet', 'No snippet')
            display_text = f"{index}. {title}\n{link}\n{snippet}\n\n"
            self.result_text.insert(tk.END, display_text)
            # Note: The hyperlink functionality needs to be adapted for use in scrolledtext, which might not directly support HTMLLabel.

    def run(self):
        self.root.mainloop()




