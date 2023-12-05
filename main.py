from model import *
import html
import requests
import json
import random
import psutil

def get_memory_info():
    # Get the memory information
    memory = psutil.virtual_memory()

    # Extract and print used and total memory
    used_memory = memory.used
    total_memory = memory.total

    print(f"Used Memory: {used_memory / (1024 ** 3):.2f} GB")
    print(f"Total Memory: {total_memory / (1024 ** 3):.2f} GB")


def trainText(text):
    etext = html.escape(text)
    etext = etext.replace('\n', '<br>')

    for ch in etext:
        pushChar(ch)

def count_rows():
    # Replace the URL with the actual API endpoint you want to call
    url = "http://eswayer.com/api/ml/wiki_api.php"

    # Make the HTTP GET request
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Decode the JSON response
        json_data = response.json()

        return int(json_data['count'])
    else:
        # Print an error message if the request was not successful
        print(f"Error: {response.status_code}")

def getNRow(n):
    # Replace the URL with the actual API endpoint you want to call
    url = "http://eswayer.com/api/ml/wiki_api.php?n=" + str(n)

    # Make the HTTP GET request
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Decode the JSON response
        json_data = response.json()

        return json_data
    else:
        # Print an error message if the request was not successful
        print(f"Error: {response.status_code}")

totalRows = count_rows()

engrave = 10
maxCycles = engrave * totalRows

cycles = 0

while cycles < maxCycles:
    try:
        # Fetch all rows from the result set
        n = random.randrange(totalRows)
        row = getNRow(n)

        print(f"Working on {row['name']}")

        trainText(row['text'])
        cycles += 1
        print(f"Current cycle: {cycles} / {maxCycles} \t {row['name']}")
    except Exception as e:
        print("Row error, jumped: ", e)
        #raise e


print("Done")

