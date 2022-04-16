import os
import pandas as pd
import re
import json
from alive_progress import alive_bar

cwd = os.getcwd()

# Load course names and descriptions into data frame
with open(cwd + "/*.csv", "r") as f:
    df = pd.read_csv(f, names=["Code", "Name", "Description"], index_col="Code")

# Load english dictionary
with open(cwd + "/dictionary.json", "r") as f:
    dict = json.load(f)

root_lookup = dict.get("terms", {})

print("Generating terms...")
with alive_bar(len(df.values) * len(df.columns)) as bar:
    terms = {}

    for row in df.values:
        for i in row:
            i = re.sub(r"\'s?", "", str(i).lower())
            i = re.sub(r"[^\w]|\d|_", " ", i)
            words = i.split()

            for w in words:
                root = root_lookup.get(w, w)
                if len(root) > 1:
                    count = terms.get(root, 0)
                    count += 1
                    terms.update({root: count})
            
            bar()

# Print terms to JSON file
with open(cwd + "/terms.json", "w") as f:
    json.dump(terms, f, sort_keys=True, indent=4)
