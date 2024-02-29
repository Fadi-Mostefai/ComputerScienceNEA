import tkinter as tk
import webbrowser
from tkinter import messagebox, simpledialog, Listbox, Scrollbar, VERTICAL, END
from tkhtmlview import HTMLLabel
import pandas as pd
import requests
from bs4 import BeautifulSoup
from googleapiclient.discovery import build


# The following class handles the interaction with the Google Custom Search JSON API.
class GoogleSearch:
    def __init__(self, api_key, cse_id):
        self.api_key = api_key
        self.cse_id = cse_id

    # The following method performs a search using the provided search term.
    def search(self, search_term, num_results=10):
        # Using the googleapiclient.discovery's build function to create a service object for the API
        service = build("customsearch", "v1", developerKey=self.api_key)
        # Queries the API with the search term, number of results (num_results), and returns the items found. If no items are found, it returns an empty list.
        res = service.cse().list(q=search_term, cx=self.cse_id, num=num_results).execute()
        return res['items'] if 'items' in res else []

# The following class provides a GUI for entering search queries, invoking searches, and displaying results.
class GoogleGUI:
    def __init__(self, root, api_key, cse_id, main_app):
        self.root = root
        self.search_engine = GoogleSearch(api_key, cse_id)
        self.main_app = main_app
        self.url_map = {}
        self.setup_ui()

    def setup_ui(self):
        # Assuming you have an entry for search queries
        self.search_entry = tk.Entry(self.root, width=50)
        self.search_entry.pack()

        search_button = tk.Button(self.root, text="Search", command=self.perform_search)
        search_button.pack()

        # Setup scroll box for results
        self.results_listbox = Listbox(self.root, width=50, height=10)
        self.results_listbox.pack(padx=10, pady=10)

        scrollbar = Scrollbar(self.root, orient=VERTICAL, command=self.results_listbox.yview)
        scrollbar.pack(side="right", fill="y")

        self.results_listbox.config(yscrollcommand=scrollbar.set)
        self.results_listbox.bind('<<ListboxSelect>>', self.on_result_select)

        back_button = tk.Button(self.root, text="Back to Main Menu", command=self.back_to_main_menu)
        back_button.pack(side=tk.LEFT, padx=(20, 10), pady=20)

    def perform_search(self):
        query = self.search_entry.get()
        results = self.search_engine.search(query)  # Your method to fetch results
        self.display_results(results)

    def display_results(self, results):
        self.results_listbox.delete(0, END)
        self.url_map.clear()

        for item in results:
            title = item['title']
            url = item['link']
            self.results_listbox.insert(END, title)
            self.url_map[title] = url

    def on_result_select(self, evt):
        selection = evt.widget.curselection()
        if selection:
            index = selection[0]
            title = evt.widget.get(index)
            url = self.url_map[title]
            webbrowser.open(url)

    def back_to_main_menu(self):
        self.root.destroy()  # Close the current window
        main_root = tk.Tk()  # Create a new Tk root window
        self.main_app(main_root)  # Instantiate MainApplication with the new root
        main_root.mainloop()  # Start the Tkinter loop for the new window

    def exit_app(self):
        self.root.quit()  # Quit the application

    def run(self):
        self.root.mainloop()




