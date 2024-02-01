import tkinter as tk
from googleapiclient.discovery import build

class GoogleSearch:
    def __init__(self, api_key, cse_key):
        self.api_key = api_key
        self.cse_key = cse_key

    def search(self, query):
        service = build("customsearch", "v1", developerKey=self.api_key)
        res = service.cse().list(q=query, cx=self.cse_key).execute()
        return res['items']

class GoogleGUI:
    def __init__(self, api_key, cse_key):
        self.search = GoogleSearch(api_key, cse_key)
        self.root = tk.Tk()
        self.create_gui()

    def create_gui(self):
        self.root.title("Google Search API")
        self.search_entry = tk.Entry(self.root, width=50)
        self.search_entry.pack()
        self.search_button = tk.Button(self.root, text="Search", command=self.perform_search)
        self.search_button.pack()
        self.results_listbox = tk.Listbox(self.root, width=100)
        self.results_listbox.pack()
        self.results_listbox.bind("<<ListboxSelect>>", self.on_result_select)

    def perform_search(self):
        query = self.search_entry.get()
        results = self.search.search(query)
        self.results_listbox.delete(0, tk.END)
        for result in results:
            self.results_listbox.insert(tk.END, result['title'])

    def on_result_select(self, event):
        selection = event.widget.curselection()
        if selection:
            index = selection[0]
            data = event.widget.get(index)
            tk.messagebox.showinfo("Selected Result", data)

    def run(self):
        self.root.mainloop()




