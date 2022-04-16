import os
import numpy as np
import pandas as pd
import pickle
import re
import json
from sklearn import preprocessing
from alive_progress import alive_bar

cwd = os.getcwd()

# Load data
with open(cwd + "/courses/*.csv", "r") as f:
    df = pd.read_csv(f, names=["Code", "Name", "Description"], index_col="Code")

# Term labels
with open(cwd + "/terms.json", "r") as f:
    frequencies = json.load(f)
terms = list(frequencies.keys())

# Document labels
documents = df.index.to_list()

# Preallocate space for terms x documents matrix
matrix = np.zeros([len(terms), len(documents)])

# Load english dictionary
with open(cwd + "/dictionary.json", "r") as f:
    dict = json.load(f)
root_lookup = dict.get("terms", {})

# Get count of each word in each document
print("Counting terms in each document...")
with alive_bar(len(df.values)) as bar:
    for i, row in enumerate(df.values):
        for r in row:
            r = re.sub(r"\'s?", "", str(r).lower())
            r = re.sub(r"[^\w]|\d|_", " ", r)
            words = r.split()

            for w in words:
                root = root_lookup.get(w, w)
                if len(root) > 1:
                    count = matrix[terms.index(root), i]
                    count += 1
                    matrix[terms.index(root), i] = count
        
        bar()

# Divide each term by its global frequency
weights = np.zeros([len(terms), len(terms)])
for i, term in enumerate(terms):
    weights[i, i] = 1 / float(frequencies.get(term))
matrix = np.matmul(weights, matrix)

# Normalize columns
matrix = preprocessing.normalize(matrix, axis=0)

# Output matrix to binary file
with open(cwd + "/matrices/A.pickle", "wb") as f:
    pickle.dump(matrix, f)
