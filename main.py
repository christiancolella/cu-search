import os
import numpy as np
import pickle
import csv
import re
import json
from config import config as cfg

class IR:
    def __init__(self, config):
        """Information Retrieval object using parameters from config"""

        NAME = config.get("database", {}).get("name")           # Load config
        DICT_PATH = config.get("dictionary_path", "")
        DATABASE_PATH = NAME + "/" + config.get("database",
            {}).get("filepath", "")

        if os.path.isdir(NAME):                                 # Attempt to load matrix factors
            if os.path.isdir(NAME + "/matrices"):
                m_path = NAME + "/matrices/"

                u_exists = os.path.isfile(m_path + "/U.pickle")
                s_exists = os.path.isfile(m_path + "/SIGMA.pickle")
                v_exists = os.path.isfile(m_path + "/V.pickle")

                if u_exists and s_exists and v_exists:
                    temp_path = NAME + "/matrices/"
                    with open(temp_path + "U.pickle", "rb") as f:
                        U = pickle.load(f)
                    with open(temp_path + "SIGMA.pickle", "rb") as f:
                        SIGMA = pickle.load(f)
                    with open(temp_path + "V.pickle", "rb") as f:
                        V = pickle.load(f)
                else:
                    print("Error: SVD matrices do not exist.")
                    exit()
            else:
                print("Error: Matrices folder does not exist.")
                exit()
        else:
            print("Error: No folder with name: \'" + NAME + "\'")
            exit()

        self.matrix = np.matmul(U, np.matmul(SIGMA, V.T))       # Multiply factors together

        if os.path.isfile(DICT_PATH):                           # Load and set dictionary of root words
            with open(DICT_PATH, "r") as f:
                temp = json.load(f)
                self.dictionary = temp.get("terms", {})
        else:
            print("Error: No file found at: \'" + DICT_PATH 
                    + "\'")
            exit()

        if os.path.isfile(NAME + "/terms.json"):                # Load and set list of terms
            with open(NAME + "/terms.json", "r") as f:
                temp = json.load(f)
                self.terms = list(temp.keys())
        else:
            print("Error: No file found at: \'" + NAME 
                    + "/terms.json\'")
            exit()

        if os.path.isfile(DATABASE_PATH):                       # Load and set documents from database
            with open(DATABASE_PATH, "r") as f:
                csvreader = csv.reader(f)
                self.documents = list(csvreader)
        else:
            print("Error: No file found at: \'" 
                + DATABASE_PATH  + "\'")
            exit()


    def format_query(self, query):
        """Convert a string query into a unit vector of term frequencies"""

        Q = np.zeros([len(self.terms), 1])                      # Allocate space for query vector
        
        query = re.sub(r"\'s?", "", str(query).lower())         # Trim non-alphabetic characters from input.
        query = re.sub(r"[^\w]|\d|_", " ", query)               # This is the same procedure used to
        words = query.split()                                   # generate the list of terms.

        for w in words:                                         # Get count of terms in query
            root = self.dictionary.get(w, w)
            if len(root) > 1:
                try:
                    index = self.terms.index(root)
                    count = Q[index, 0]
                    count += 1
                    Q[index, 0] = count

                except ValueError:                              # Ignore word if not
                    pass                                        # in terms list

        return (1 / np.linalg.norm(Q)) * Q                      # Normalize query vector


    def search(self, query, num_results):
        """Search database for query"""

        print("\nTop " + str(num_results)
            + " results for \"" + query + "\":")
        
        COS = np.matmul(self.matrix.T, 
            self.format_query(query))                           # Compute cosine
        COS = COS.reshape(self.matrix.shape[1])      

        results = COS.argsort()[-num_results:][::-1]            # Get indices of top results

        for i, r in enumerate(results):                         # Print results
            if COS[r] > 0.01:
                result = self.documents[r]
                print(str(i + 1) + ") " + result[0]
                    + " - " + result[1])
            else:
                print("-- No more matches --")
                break

        print("\n")


def main():
    ir = IR(cfg)                                                # Create information retrieval object
    ir.search("linear algebra", 15)                             # Make search here

main()
