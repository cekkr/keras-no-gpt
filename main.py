from model import *
import html
import sqlite3

def trainText(text):
    etext = html.escape(text)

    for ch in etext:
        pushChar(ch)


# Connect to the SQLite database (replace 'your_database.db' with your actual database file)
conn = sqlite3.connect('/Volumes/AirUSB/Datasets/docs.db')

# Create a cursor object to interact with the database
cursor = conn.cursor()

def count_rows(table_name):
    try:
        # Execute a SQL query to count the rows in the specified table
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")

        # Fetch the result of the query
        row_count = cursor.fetchone()[0]

        # Print or use the row count as needed
        print(f'Total number of rows in {table_name}: {row_count}')

    except sqlite3.Error as e:
        print(f"Error: {e}")

    finally:
        # Close the cursor and the connection
        cursor.close()
        conn.close()

# Replace 'your_table' with the actual table name
table_name = 'documents'

totalRows = count_rows(table_name)

engrave = 10
maxCycles = engrave * totalRows

cycles = 0

while cycles < maxCycles:

    # Select all rows from the table
    cursor.execute(f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT 1")

    # Fetch all rows from the result set
    rows = cursor.fetchall()

    # Iterate through the rows and print the data (adjust as needed)
    for row in rows:
        trainText(row.text)

# Close the cursor and the connection
cursor.close()
conn.close()
