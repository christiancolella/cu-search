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

            if os.path.isdir(NAME + "/mat/td"):                 # Terms x documents
                temp_path = NAME + "/mat/td/"

                u_exists = os.path.isfile(temp_path + "u.pickle")
                s_exists = os.path.isfile(temp_path + "s.pickle")
                v_exists = os.path.isfile(temp_path + "v.pickle")

                if u_exists and s_exists and v_exists:
                    with open(temp_path + "u.pickle", "rb") as f:
                        self.td_U = pickle.load(f)
                    with open(temp_path + "s.pickle", "rb") as f:
                        self.td_S = pickle.load(f)
                    with open(temp_path + "v.pickle", "rb") as f:
                        self.td_V = pickle.load(f)
                else:
                    print("Error: Terms x Documents SVD matrices do not exist.")
                    exit()
            else:
                print("Error: Matrices do not exist.")
                exit()
            
            if os.path.isdir(NAME + "/mat/tt"):                 # Term comparison
                temp_path = NAME + "/mat/tt/"

                u_exists = os.path.isfile(temp_path + "u.pickle")
                s_exists = os.path.isfile(temp_path + "s.pickle")
                v_exists = os.path.isfile(temp_path + "v.pickle")

                if u_exists and s_exists and v_exists:
                    with open(temp_path + "u.pickle", "rb") as f:
                        self.tt_U = pickle.load(f)
                    with open(temp_path + "s.pickle", "rb") as f:
                        self.tt_S = pickle.load(f)
                    with open(temp_path + "v.pickle", "rb") as f:
                        self.tt_V = pickle.load(f)
                else:
                    print("Error: Term comparison SVD matrices do not exist.")
                    exit()
            else:
                print("Error: Matrices do not exist.")
                exit()

        else:
            print("Error: No folder with name: \'" + NAME + "\'")
            exit()

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
        TEMP = np.zeros([len(self.terms), 1])                   # and temp query vector
        
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

        """
        for i in range(len(Q)):                                 # Compute term comparison
            V = np.matmul(self.tt_V[i, :], self.tt_S)           # This is space efficient but very slow
            W = np.matmul(self.tt_U.T, Q)
            TEMP[i, 0] = np.dot(V, W)

        Q = TEMP
        """

        A = np.matmul(self.tt_U, 
            np.matmul(self.tt_S, self.tt_V.T))                  # Compute term comparison
        for i, row in enumerate(A):                             # This is fast but not space efficient
            row[i] = 1.0                                        # Force diagonal entries to be 1
        Q = np.matmul(A, Q)
        
        """
        with open("cu_search/mat/tt/" + "a.pickle", "rb") as f: # Use uncompressed term comparison matrix
            A = pickle.load(f)
        Q = np.matmul(A, Q)
        """

        return (1 / np.linalg.norm(Q)) * Q                      # Normalize query vector


    def search(self, query, num_results):
        """Search database for query"""

        print("\nTop " + str(num_results)
            + " results for \"" + query + "\":")
        
        Q = self.format_query(query)                            # Format query as vector
        # COS = np.zeros(len(Q))                                # Allocate space for cosines

        """
        for i in range(len(A.shape[1])):                        # Compute cosines
            V = np.matmul(self.td_V[i, :], self.td_S)           # This is space efficient but very slow
            W = np.matmul(self.td_U.T, Q)
            COS[i] = np.dot(V, W)
        """

        A = np.matmul(self.td_U, 
            np.matmul(self.td_S, self.td_V.T))                  # Compute cosines

        """
        with open("cu_search/mat/td/" + "a.pickle", "rb") as f: # Use uncompressed term x doc matrix
            A = pickle.load(f)
        """

        COS = np.matmul(A.T, Q)                                 # This is fast but not space efficient
        COS = COS.reshape(A.shape[1])

        results = COS.argsort()[-num_results:][::-1]            # Get indices of top results

        for i, r in enumerate(results):                         # Print results
            if COS[r] > 0.01:
                result = self.documents[r]
                print(str(i + 1) + ") " + result[0]
                    + " - " + result[1] + "; " + str(round(COS[r], 3)))
            else:
                print("-- No more matches --")
                break

        print("\n")


def main():
    ir = IR(cfg)                                                # Create information retrieval object
    ir.search("linear algebra", 15)                             # Make search here

main()
