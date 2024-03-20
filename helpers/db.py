import sqlite3
from contextlib import contextmanager

@contextmanager
def get_db():
    conn = sqlite3.connect('storage.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS LoanUser (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            loanID TEXT NOT NULL,
            status TEXT NOT NULL,
            dueDate INTEGER NOT NULL,
            amount INTEGER NOT NULL DEFAULT 100,
            borrower TEXT,
            lender TEXT,
            feature1 INTEGER NOT NULL,
            feature2 INTEGER NOT NULL,
            feature3 INTEGER NOT NULL,
            interestRate INTEGER NOT NULL DEFAULT 10,
            isRepaid BOOLEAN NOT NULL DEFAULT FALSE,
            createdAt DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    ''')

    try:
        yield conn
    finally:
        conn.close()