import os
import re
import json
from alive_progress import alive_bar

cwd = os.getcwd()
with open(cwd + "/english_words.txt", "r") as f:
    words = f.read().split("\n")

dict = {"terms": {}, "roots": {}}

prefixes = r"((?:in)|(?:im)|(?:il)|(?:ir)|(?:un)|(?:re)|(?:pre)|(?:de)|(?:dis)|(?:sub)|(?:super)|(?:under)|(?:over)|(?:trans)|(?:con))?"
suffixes = r"(?:(?:ed)|(?:ings?)|(?:ers?)|(?:ors?)|(?:est)|(?:ly)|(?:ily)|(?:ions?)|(?:ations?)|(?:ness)|(?:liness)|(?:less)|(?:lessness)|(?:ments?)|(?:able)|(?:ability)|(?:ible)|(?:ibility)|(?:i?al)|(?:i?ally)|(?:ant)|(?:ent)|(?:ance)|(?:ence)|(?:ive)|(?:ative)|(?:ic)|(?:ist)|(?:ism))*(?:s)?$"
y_suffixes = r"(?:(?:ies)|(?:ied)|(?:yings?)|(?:iers?)|(?:iest)|(?:ily)|(?:iness)|(?:iless)|(?:able)|(?:ability)|(?:ial))$"

with alive_bar(len(words)) as bar:
    print("Grouping words...")
    for word in words:
        ends_in_y = False

        # Match prefix and suffix
        try:
            match = re.match(re.compile(prefixes + r"(\w+?)" + y_suffixes), word)
            if match:
                if (match.group(2) + 'y') in words:
                    ends_in_y = True
            else:
                match = re.match(re.compile(prefixes + r"(\w+?)" + suffixes), word)

            if match:
                root = match.group(2)
            else:
                root = word
        except IndexError:
            pass

        # Determine root word
        terms = dict.get("terms", {})
        roots = dict.get("roots", {})

        if root in words:
            stems = roots.get(root, [])
            stems.append(word)
            roots.update({root: stems})
        elif (root + 'e') in words:
            r = root + 'e'
            stems = roots.get(r, [])
            stems.append(word)
            roots.update({r: stems})
        elif (root + 'y') in words and ends_in_y:
            r = root + 'y'
            stems = roots.get(r, [])
            stems.append(word)
            roots.update({r: stems})
        elif len(root) >= 2:
            if root[-1] == root[-2]:
                r = root[:-1]
                if r in words:
                    stems = roots.get(r, [])
                    stems.append(word)
                    roots.update({r: stems})
            elif root[-2:] == "ss":
                r = root[:-2] + 't'
                if r in words:
                    stems = roots.get(r, [])
                    stems.append(word)
                    roots.update({r: stems})
        else:
            stems = dict.get(root, [])
            stems.append(word)
            roots.update({root: stems})

        # Update main dict
        terms.update({word: root})
        dict.update({"terms": terms})
        dict.update({"roots": roots})

        # Progress bar
        bar()

print("Grouped " + str(len(words)) + " total words with " + str(len(dict.keys())) + " root words.")

# Output to JSON file
print("Creating JSON file...")
with open(cwd + "/dictionary.json", "w") as f:
    json.dump(dict, f, sort_keys=True, indent=4)

# Print to console when complete
print("Done!")
