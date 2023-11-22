from model import *
import html
import sqlite3

def trainText(text):
    etext = html.escape(text)

    x_bag = np.zeros(tokensBag)

    for ch in etext:
        x_pred = np.zeros(len(nChars))
        x_pred[ord(ch)] = 1


# Connect to the SQLite database (replace 'your_database.db' with your actual database file)
conn = sqlite3.connect('/Volumes/AirUSB/Datasets/docs.db')

# Create a cursor object to interact with the database
cursor = conn.cursor()

# Replace 'your_table' with the actual table name
table_name = 'documents'

# Select all rows from the table
cursor.execute(f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT 1")

# Fetch all rows from the result set
rows = cursor.fetchall()

# Iterate through the rows and print the data (adjust as needed)
for row in rows:
    print(row)

# Close the cursor and the connection
cursor.close()
conn.close()
