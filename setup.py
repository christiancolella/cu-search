import csv
import os
import numpy as np
import pandas as pd
import pickle
import re
import json
import math
from sklearn import preprocessing
from alive_progress import alive_bar
from config import config
import utils

NAME = config.get("database", {}).get("name")                   # Load config
WORDS_PATH = config.get("words_path", "")
DICT_PATH = config.get("dictionary_path", "")
DATABASE_PATH = NAME + "/" + config.get("database", {}).get(
    "filepath", "")

def generate_dictionary():
    """Generate a dictionary of root words from a text file of words"""

    if os.path.isfile(WORDS_PATH):
        with open(WORDS_PATH, "r") as f:                        # Load words from file
            words = f.read().split("\n")
    else:
        print("Error: No file found at: \'" + WORDS_PATH + "\'")
        exit()

    dict = {"terms": {}, "roots": {}}

    PRE = r"((?:in)|(?:im)|(?:il)|(?:ir)|(?:un)|(?:re)      \
    |(?:pre)|(?:de)|(?:dis)|(?:sub)|(?:super)|(?:under)     \
    |(?:over)|(?:trans)|(?:con))?"                              # Common prefixes

    SUF = r"(?:(?:ed)|(?:ings?)|(?:ers?)|(?:ors?)|(?:est)   \
    |(?:ly)|(?:ily)|(?:ions?)|(?:ations?)|(?:ness)          \
    |(?:liness)|(?:less)|(?:lessness)|(?:ments?)|(?:able)   \
    |(?:ability)|(?:ible)|(?:ibility)|(?:i?al)|(?:i?ally)   \
    |(?:ant)|(?:ent)|(?:ance)|(?:ence)|(?:ive)|(?:ative)    \
    |(?:ic)|(?:ist)|(?:ism))*(?:s)?$"                           # Common suffixes

    Y_SUF = r"(?:(?:ies)|(?:ied)|(?:yings?)|(?:iers?)       \
    |(?:iest)|(?:ily)|(?:iness)|(?:iless)|(?:able)          \
    |(?:ability)|(?:ial))$"                                     # Common suffixes for words ending in 'y'

    with alive_bar(len(words)) as bar:
        print("Grouping words...")
        for word in words:
            ends_in_y = False
            try:
                match = re.match(re.compile(PRE + r"(\w+?)" \
                    + Y_SUF), word)                             # Attempt to determine if word ends in 'y'
                if match:                                       # so that the corresponding suffixes can be
                    if (match.group(2) + 'y') in words:         # applied
                        ends_in_y = True
                else:
                    match = re.match(re.compile(PRE + \
                        r"(\w+?)" + SUF), word)                 # Default to assuming that word does not
                                                                # end in 'y'
                if match:
                    root = match.group(2)
                else:
                    root = word
            except IndexError:
                pass

            terms = dict.get("terms", {})                       # Attempt to determine root word by matching
            roots = dict.get("roots", {})                       # a guess with an entry in the list of words

            if root in words:                                   # Guess root alone
                stems = roots.get(root, [])
                stems.append(word)
                roots.update({root: stems})

            elif (root + 'e') in words:                         # Guess root + 'e'
                r = root + 'e'
                stems = roots.get(r, [])
                stems.append(word)
                roots.update({r: stems})

            elif (root + 'y') in words and ends_in_y:           # Guess root + 'y' if the word has been 
                r = root + 'y'                                  # determined to end in 'y'
                stems = roots.get(r, [])
                stems.append(word)
                roots.update({r: stems})

            elif len(root) >= 2:
                if root[-1] == root[-2]:                        # Try removing a repeated letter
                    r = root[:-1]
                    if r in words:
                        stems = roots.get(r, [])
                        stems.append(word)
                        roots.update({r: stems})

                elif root[-2:] == "ss":                         # Try substituting 'ss' with 't'
                    r = root[:-2] + 't'
                    if r in words:
                        stems = roots.get(r, [])
                        stems.append(word)
                        roots.update({r: stems})

            else:                                               # Default to assuming the root word is valid
                stems = dict.get(root, [])                      # on its own
                stems.append(word)
                roots.update({root: stems})

            terms.update({word: root})                          # Update the dictionary with the new word
            dict.update({"terms": terms})                       # and its determined root
            dict.update({"roots": roots})

            bar()

    print("Grouped " + str(len(words)) + " total words with " \
        + str(len(dict.keys())) + " root words.")

    print("Creating JSON file...")                              # Output to JSON file
    if not utils.path_exists(DICT_PATH):
        utils.make_path(DICT_PATH)
    with open(DICT_PATH, "w") as f:
        json.dump(dict, f, sort_keys=True, indent=4)


def generate_terms():
    """Generate a list of terms from a csv file of documents"""

    if os.path.isfile(DATABASE_PATH):
        with open(DATABASE_PATH, "r") as f:                     # Load course names and descriptions
            df = pd.read_csv(f)                                 # into dataframe
    else:
        print("Error: No file found at: \'" 
            + DATABASE_PATH + "\'")
        exit()

    if os.path.isfile(DICT_PATH):
        with open(DICT_PATH, "r") as f:                         # Load dictionary
            temp = json.load(f)
        root_lookup = temp.get("terms", {})
    else:
        print("Error: No file found at: \'" 
            + DICT_PATH + "\'")
        exit()

    print("Generating terms...")
    with alive_bar(len(df.values) * len(df.columns)) as bar:
        terms = {}

        for row in df.values:
            for i in row:
                i = re.sub(r"\'s?", "", str(i).lower())         # Remove apostrophes, numbers, underscores
                i = re.sub(r"[^\w]|\d|_", " ", i)               # and other non-alphabetic characters
                words = i.split()

                for w in words:
                    root = root_lookup.get(w, w)                # Look up root word from dictionary to use
                    if len(root) > 1:                           # as a term. If no root word found, then the
                        count = terms.get(root, 0)              # word itself is a new term.
                        count += 1
                        terms.update({root: count})
                
                bar()

    if not os.path.isdir(NAME):
        os.mkdir(NAME)
    with open(NAME + "/terms.json", "w") as f:                  # Print terms to JSON file
        json.dump(terms, f, sort_keys=True, indent=4)


def generate_matrices(csv_filepath, col_weights):
    """Generate uncompressed terms x documents matrix and term comparison matrix"""

    if os.path.isfile(csv_filepath):
        with open(csv_filepath, "r") as f:                      # Load database from csv file
            df = pd.read_csv(f)
        documents = df.index.to_list()
    else:
        print("Error: No file found at: \'" 
            + csv_filepath + "\'")
        exit()

    if os.path.isfile(NAME + "/terms.json"):
        with open(NAME + "/terms.json", "r") as f:              # Load terms from file
            freq = json.load(f)
        terms = list(freq.keys())
    else:
        print("Error: No file found at: \'" + NAME 
            + "/terms.json" + "\'")
        exit()

    td_matrix = np.zeros([len(terms), len(documents)])          # Allocate space for term x doc matrix

    if os.path.isfile(DICT_PATH):
        with open(DICT_PATH, "r") as f:                         # Load dictionary of root words
            dict = json.load(f)
        root_lookup = dict.get("terms", {})
    else:
        print("Error: No file found at: \'" 
            + DICT_PATH + "\'")
        exit()

    print("Counting terms in each document...")                 # Get count of each term in each document
    with alive_bar(len(df.values)) as bar:
        for i, row in enumerate(df.values):
            for j, r in enumerate(row):
                r = re.sub(r"\'s?", "", str(r).lower())         # Remove non-alphabetic characters
                r = re.sub(r"[^\w]|\d|_", " ", r)
                words = r.split()

                for w in words:
                    root = root_lookup.get(w, w)                # Get root word
                    if len(root) > 1:
                        count = td_matrix[terms.index(root), i] # Add weighted count according to by column
                        count += col_weights[j]                 # according to 'col_weights'
                        td_matrix[terms.index(root), i] = count
            
            bar()

    weights = np.zeros([len(terms), len(terms)])                # Divide each term by its frequency in the
    for i, term in enumerate(terms):                            # database in order to rank its importance
        weights[i, i] = 1 / float(freq.get(term))
    td_matrix = np.matmul(weights, td_matrix)
    temp_matrix = preprocessing.normalize(td_matrix, axis=1)

    td_matrix = preprocessing.normalize(td_matrix, axis=0)      # Normalize the columns of the matrix

    output_path = NAME + "/mat/td/a.pickle"                     # Create directory for term x doc matrix  
    if not utils.path_exists(output_path):                      # if it doesn't already exist
        utils.make_path(output_path)
    
    with open(output_path, "wb") as f:                          # Output term x doc matrix to binary file
        pickle.dump(td_matrix, f)

    tt_matrix = np.matmul(temp_matrix, temp_matrix.T)           # Create term comparison matrix
    print(tt_matrix[0:3, 0:3])

    output_path = NAME + "/mat/tt/a.pickle"                     # Create directory for term comp matrix  
    if not utils.path_exists(output_path):                      # if it doesn't already exist
        utils.make_path(output_path)

    with open(output_path, "wb") as f:                          # Output term comp matrix to binary file
        pickle.dump(tt_matrix, f)


def incomplete_svd(matrix, k):
    """Compute the singular value decomposition of a matrix up to rank 'k'"""

    THRESHOLD = 0.005                                           # Threshold for eigenvector algorithm
    m, n = matrix.shape

    ATA = np.matmul(matrix.T, matrix)                           # Store A-transpose A

    U = np.zeros([m, k])                                        # Allocate space for factors
    SIGMA = np.zeros([k, k])
    V = np.zeros([n, k])

    print("Calculating eigenvectors...")                        # Calculate 'k' largest eigenvectors of ATA
    with alive_bar(k) as bar:                                   
        for i in range(k):
            u = np.random.rand(n, 1)                            # Generate random unit vector
            u = (1 / np.linalg.norm(u)) * u
            u_last = np.zeros([n, 1])

            while np.linalg.norm(np.subtract(u,
            u_last)) > THRESHOLD:                               # Loop while ||u_n+1 - u_n|| is below the
                u_last = u                                      # specified threshold
                u = np.matmul(ATA, u)
                u = (1 / np.linalg.norm(u)) * u
                                                                # Gram-Schmidt algorithm for generating
                for j in range(i - 1):                          # orthonormal eigenvectors
                    u = np.subtract(u, 
                    np.dot(u.T, V[:, [j]]) * V[:, [j]])

                u = (1 / np.linalg.norm(u)) * u

            V[:, [i]] = u                                       # Store eigenvector in V matrix and
            SIGMA[i, i] = math.sqrt(np.linalg.norm(             # compute corresponding singular value
                np.matmul(ATA, u)) / np.linalg.norm(u))

            bar()

    for i in range(k):                                          # Compute the columns of U
        U[:, [i]] = (1 / SIGMA[i, i]) * np.matmul(
            matrix, V[:, [i]])

    return U, SIGMA, V

def compress(matrix_filepath, rank, type):
    """Compress a matrix to a specified rank with the incomplete SVD.
    'type' can be either 'td' or 'tt'."""

    if os.path.isfile(matrix_filepath):
        with open(matrix_filepath, "rb") as f:                  # Load matrix from file
            A = pickle.load(f)
    else:
        print("Error: No file found at: \'" + matrix_filepath
            + "\'")

    print("Calculating SVD factorization...")                   # Compute incomplete SVD
    U, S, V = incomplete_svd(A, rank)

    if type == "td":
        output_path = NAME + "/mat/td/"
    if type == "tt":
        output_path = NAME + "/mat/tt/"
    else:
        print("Error: Invalid type. Must be either 'td' or 'tt'.")
        return
    
    if not utils.path_exists(output_path + "a.txt"):            # Create folder if it doesn't exist
        utils.make_path(output_path + "a.txt")

    with open(output_path + "u.pickle", "wb") as f:             # Output matrix factors to binary files
        pickle.dump(U, f)
    with open(output_path + "s.pickle", "wb") as f:
        pickle.dump(S, f)
    with open(output_path + "v.pickle", "wb") as f:
        pickle.dump(V, f)

def main():
    """Setup from config"""
    
    # generate_dictionary()
    # generate_terms()
    # generate_matrices(DATABASE_PATH, [0, 1, 1])
    # compress("cu_search/mat/td/a.pickle", 100, "td")
    # compress("cu_search/mat/tt/a.pickle", 100, "tt")

main()
