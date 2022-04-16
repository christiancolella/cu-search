import os
import numpy as np
import csv
import pickle
import re
import json

class IR:
    def __init__(self, dictionary, terms, matrix):         # Constructor
        self.dictionary = dictionary
        self.terms = terms
        self.matrix = matrix
        return

    def format_query(self, query):
        Q = np.zeros([len(self.terms), 1])                 # Allocate space for query vector
        
        query = re.sub(r"\'s?", "", str(query).lower())    # Trim input
        query = re.sub(r"[^\w]|\d|_", " ", query)
        words = query.split()

        for w in words:                                    # Get count of terms in query
            root = self.dictionary.get(w, w)
            if len(root) > 1:
                try:
                    index = self.terms.index(root)
                    count = Q[index, 0]
                    count += 1
                    Q[index, 0] = count
                except ValueError:                         # Ignore word if not
                    pass                                   # in terms list

        return (1 / np.linalg.norm(Q)) * Q                 # Normalize query vector

    def search(self, query, database, num_results):
        print("\nTop " + str(num_results)
            + " results for \"" + query + "\":")
        
        COS = np.matmul(self.matrix.T, self.format_query(query))     # Compute cosine
        COS = COS.reshape(self.matrix.shape[1])      

        results = COS.argsort()[-num_results:][::-1]       # Get indices of top results

        with open(database, "r") as f:                     # Load database (list of courses)
            csvreader = csv.reader(f)
            courses = list(csvreader)

        for i, r in enumerate(results):                    # Print results
            if COS[r] > 0.01:
                course = courses[r]
                print(str(i + 1) + ") " + course[0]
                    + " - " + course[1])
            else:
                print("-- No more matches --")
                break

        print("\n")
        return

def main():
    cwd = os.getcwd()

    with open(cwd + "/matrices/U.pickle", "rb") as f:      # Load matrix factors
        U = pickle.load(f)
    with open(cwd + "/matrices/SIGMA.pickle", "rb") as f:
        SIGMA = pickle.load(f)
    with open(cwd + "/matrices/V.pickle", "rb") as f:
        V = pickle.load(f)

    A = np.matmul(U, np.matmul(SIGMA, V.T))                # Multiply factors together

    with open(cwd + "/terms.json", "r") as f:              # Load list of terms
        temp = json.load(f)
        T = list(temp.keys())

    with open(cwd + "/dictionary.json", "r") as f:         # Load dictionary of root words
        temp = json.load(f)
        D = temp.get("terms", {})

    ir = IR(D, T, A)
    ir.search("linear algebra", "courses/*.csv", 15)

main()
