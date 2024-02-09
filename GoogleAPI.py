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
        self.format_google_results(search_results)

    def display_results(self, results):
        html_content = "<html><body>"
        for item in results:
            title = item['title']
            link = item['link']
            html_content += f'<a href="{link}" target="_blank">{title}</a><br>'
        html_content += "</body></html>"

        self.results_display.set_html(html_content)

    # Method to open clicked links in a web browser
    def on_link_click(self, event):
        print(event)  # This will help you understand the structure of the event object.
        webbrowser.open(event.correct_attribute_for_url)

    def run(self):
        self.root.mainloop()




