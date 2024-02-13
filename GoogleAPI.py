import tkinter as tk
import webbrowser
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

        self.results_display = HTMLLabel(self.root, html="<html><body></body></html>")
        self.results_display.pack(fill="both", expand=True)
        self.results_display.bind("<Hyperlink>", self.on_link_click)

    def perform_search(self):
        query = self.query_entry.get()
        search_results = self.search_engine.search(query)
        self.display_results(search_results)

    def display_results_window(self, results):
        results_window = tk.Toplevel(self.root)
        results_window.title("Search Results")
        listbox = tk.Listbox(results_window, width=100, height=20)
        scrollbar = tk.Scrollbar(results_window, orient="vertical")
        listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=listbox.yview)
        scrollbar.pack(side="right", fill="y")
        listbox.pack(side="left", fill="both", expand=True)

        # Store links in a list for later retrieval
        self.links = []
        for result in results:
            listbox.insert(tk.END, result['title'])
            self.links.append(result['link'])

        # Bind the listbox select event to the open_link method
        listbox.bind('<<ListboxSelect>>', self.open_link)

    # Method to open clicked links in a web browser
    def open_link(self, event):
        # Get the index of the selected link
        widget = event.widget
        index = int(widget.curselection()[0])
        link = self.links[index]

        # Open the link in the default web browser
        webbrowser.open(link)

    def run(self):
        self.root.mainloop()




