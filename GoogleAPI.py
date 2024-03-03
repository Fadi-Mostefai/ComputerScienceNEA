import sys
import logging
import sqlite3
from PyQt5.QtCore import QUrl, Qt
from PyQt5.QtWidgets import (QMainWindow, QWidget, QListWidget, QListWidgetItem, QDialog, QLineEdit, QPushButton, QVBoxLayout, QLabel, QDialogButtonBox, QMessageBox)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from googleapiclient.discovery import build

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# Assuming the database and tables are already created.
# If not, you'll need to ensure the database creation scripts are executed beforehand.

class GoogleSearch:
    def __init__(self, api_key, cse_id):
        self.api_key = api_key
        self.cse_id = cse_id

    def search(self, search_term, num_results=10):
        try:
            service = build("customsearch", "v1", developerKey=self.api_key)
            res = service.cse().list(q=search_term, cx=self.cse_id, num=num_results).execute()
            return res['items'] if 'items' in res else []
        except Exception as e:
            print(f"Error during Google Search API call: {e}")
            return []

class DatabaseManager:
    def __init__(self, db_name="search_app.db"):
        self.conn = sqlite3.connect(db_name)
        self.cur = self.conn.cursor()
        self.setup_database()

    def setup_database(self):
        self.cur.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT)''')
        self.cur.execute('''CREATE TABLE IF NOT EXISTS search_history (id INTEGER PRIMARY KEY, user_id INTEGER, link TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        self.conn.commit()

    def create_user(self, username, password):
        try:
            self.cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def validate_login(self, username, password):
        self.cur.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        return self.cur.fetchone() is not None

    def save_search_history(self, user_id, link):
        self.cur.execute("INSERT INTO search_history (user_id, link) VALUES (?, ?)", (user_id, link))
        self.conn.commit()

    def get_search_history(self, user_id):
        self.cur.execute("SELECT link FROM search_history WHERE user_id=? ORDER BY timestamp DESC", (user_id,))
        return self.cur.fetchall()

    def get_user_id(self, username):
        self.cur.execute("SELECT id FROM users WHERE username=?", (username,))
        result = self.cur.fetchone()
        return result[0] if result else None

class LoginDialog(QDialog):
    def __init__(self, db_manager, parent=None):
        super(LoginDialog, self).__init__(parent)
        self.db_manager = db_manager
        self.setupUI()

    def setupUI(self):
        self.setWindowTitle("Login")
        layout = QVBoxLayout()

        # Username input
        self.usernameInput = QLineEdit(self)
        layout.addWidget(QLabel("Username"))
        layout.addWidget(self.usernameInput)

        # Password input
        self.passwordInput = QLineEdit(self)
        self.passwordInput.setEchoMode(QLineEdit.Password)
        layout.addWidget(QLabel("Password"))
        layout.addWidget(self.passwordInput)

        # Login button
        loginButton = QPushButton("Login", self)
        loginButton.clicked.connect(self.checkCredentials)
        layout.addWidget(loginButton)

        self.setLayout(layout)

    def checkCredentials(self):
        username = self.usernameInput.text()
        password = self.passwordInput.text()

        if self.db_manager.validate_login(username, password):
            QMessageBox.information(self, "Login Successful", "You are now logged in.")
            self.accept()
        else:
            QMessageBox.warning(self, "Login Failed", "Incorrect username or password.")

class RegisterDialog(QDialog):
    def __init__(self, db_manager, parent=None):
        super(RegisterDialog, self).__init__(parent)
        self.db_manager = db_manager
        self.setupUI()

    def setupUI(self):
        self.setWindowTitle("Register")
        layout = QVBoxLayout()

        # Username input
        self.usernameInput = QLineEdit(self)
        layout.addWidget(QLabel("Username"))
        layout.addWidget(self.usernameInput)

        # Password input
        self.passwordInput = QLineEdit(self)
        self.passwordInput.setEchoMode(QLineEdit.Password)
        layout.addWidget(QLabel("Password"))
        layout.addWidget(self.passwordInput)

        # Register button
        registerButton = QPushButton("Register", self)
        registerButton.clicked.connect(self.registerUser)
        layout.addWidget(registerButton)

        self.setLayout(layout)

    def registerUser(self):
        username = self.usernameInput.text()
        password = self.passwordInput.text()

        if self.db_manager.create_user(username, password):
            QMessageBox.information(self, "Registration Successful", "Your account has been created.")
            self.accept()
        else:
            QMessageBox.warning(self, "Registration Failed", "Username already taken or other error.")

class GoogleGUI(QMainWindow):
    def __init__(self, api_key, cse_id):
        super().__init__()
        self.search_engine = GoogleSearch(api_key, cse_id)
        self.db_manager = DatabaseManager()
        self.current_user_id = None
        self.showLoginOrRegisterDialog()

    def initUI(self):
        self.setWindowTitle("Google API Search")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.searchBar = QLineEdit()
        self.searchBar.setPlaceholderText("Enter search query...")
        self.searchButton = QPushButton("Search")
        self.searchButton.clicked.connect(self.performSearch)
        self.historyButton = QPushButton("Show History")
        self.historyButton.clicked.connect(self.showHistory)

        self.resultsList = QListWidget()
        self.resultsList.itemDoubleClicked.connect(self.openLink)

        layout.addWidget(self.searchBar)
        layout.addWidget(self.searchButton)
        layout.addWidget(self.historyButton)
        layout.addWidget(self.resultsList)

    # Inside the GoogleGUI class
    def showLoginOrRegisterDialog(self):
        choice = QMessageBox.question(self, "Login or Register",
                                      "Do you want to login? Click No to register.",
                                      QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)

        if choice == QMessageBox.Yes:
            self.showLoginDialog()
        elif choice == QMessageBox.No:
            self.showRegisterDialog()
        else:
            self.close()  # Close the application if the user cancels the operation

    def showLoginDialog(self):
        loginDialog = LoginDialog(self.db_manager, self)
        if loginDialog.exec_() == QDialog.Accepted:
            username = loginDialog.usernameInput.text()
            self.current_user_id = self.db_manager.get_user_id(username)
            self.initUI()  # Initialize the UI only after successful login
        else:
            self.close()  # Close the app if the login dialog is closed without logging in

    def showRegisterDialog(self):
        registerDialog = RegisterDialog(self.db_manager, self)
        if registerDialog.exec_() == QDialog.Accepted:
            username = registerDialog.usernameInput.text()
            self.current_user_id = self.db_manager.get_user_id(username)
            self.initUI()  # Initialize the UI only after successful registration
        else:
            self.close()  # Close the app if the registration dialog is closed without registering



    def performSearch(self):
        query = self.searchBar.text()
        if not query.strip():
            QMessageBox.warning(self, "Warning", "Please enter a query.")
            return

        try:
            results = self.search_engine.search(query)
            self.displaySearchResults(results)
            if self.current_user_id:
                # Assuming you're saving the query or link correctly in your database.
                self.db_manager.save_search_history(self.current_user_id, query)
        except Exception as e:
            logging.error(f"Failed to perform search: {e}")
            QMessageBox.critical(self, "Error", "Failed to perform search.")

    def displaySearchResults(self, results):
        self.resultsList.clear()
        if results:
            for item in results:
                listItem = QListWidgetItem(f"{item['title']} - {item['link']}")
                listItem.setData(Qt.UserRole + 1, "search_result")  # Custom role for source
                listItem.setData(Qt.UserRole, item['link'])
                self.resultsList.addItem(listItem)

    def openLink(self, item):
        url = item.data(Qt.UserRole)
        source = item.data(Qt.UserRole + 1)  # Retrieve the source information
        if url:
            if source == "search_result" and self.current_user_id:
                # Save the click into history only if it's from search results
                self.db_manager.save_search_history(self.current_user_id, url)

            self.browserWindow = QWebEngineView()
            self.browserWindow.load(QUrl(url))
            self.browserWindow.show()

    def showHistory(self):
        if self.current_user_id:
            history = self.db_manager.get_search_history(self.current_user_id)
            self.resultsList.clear()
            for link in history:
                listItem = QListWidgetItem(link[0])
                listItem.setData(Qt.UserRole + 1, "history")  # Custom role for source
                listItem.setData(Qt.UserRole, link[0])
                self.resultsList.addItem(listItem)






