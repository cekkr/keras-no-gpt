from model import *
import html
import requests
import json
import random
import psutil
import threading
import time
import random

def get_memory_info():
    # Get the memory information
    memory = psutil.virtual_memory()

    # Extract and print used and total memory
    used_memory = memory.used
    total_memory = memory.total

    print(f"Used Memory: {used_memory / (1024 ** 3):.2f} GB")
    print(f"Total Memory: {total_memory / (1024 ** 3):.2f} GB")

    return used_memory / total_memory


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

def wdata_count_rows():
    # Replace the URL with the actual API endpoint you want to call
    url = "http://eswayer.com/api/ml/wikidata_api.php"

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

def wdata_getNRow(n):
    # Replace the URL with the actual API endpoint you want to call
    url = "http://eswayer.com/api/ml/wikidata_api.php?n=" + str(n)

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
totalWRows = wdata_count_rows()

engrave = 10
maxCycles = engrave * totalRows

cycles = 0

curBatchSize = 1
curBatch = []

def collectBatch():
    global curBatchSize
    global curBatch
    global totalRows
    global totalWRows

    while True:
        while len(curBatch) < curBatchSize:
            rnum = random.randint(0, 2)

            row = None
            if rnum == 0:
                n = random.randrange(totalRows)
                row = getNRow(n)
            else:
                n = random.randrange(totalWRows)
                row = wdata_getNRow(n)
                row['text'] = row['sentence']

            print("New data row fetched")

            curBatch.append(row)

        time.sleep(1)

threadBatch = threading.Thread(target=collectBatch)
threadBatch.start()

lastRamUsage = 0

while True:
    try:
        while batch.size() < curBatchSize:
            print("check batch")
            if len(curBatch) == 0:
                # you'll be more lucky next time
                time.sleep(1)
            else:
                cont = curBatch.pop(0)
                op = Batch.Operation(cont['text'])
                batch.addOp(op)

        fitSeq()
        lastRamUsage = get_memory_info()

        if lastRamUsage < 0.75:
            curBatchSize += 1

        if lastRamUsage > 0.85:
            curBatchSize -= 1

        if curBatchSize == 0:
            print("seems there is no enough ram available...")

        cycles += 1
        print(f"Current cycle: {cycles}")

    except Exception as e:
        print("Row error, jumped: ", e)
        #raise e


print("Done")

