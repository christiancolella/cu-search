import os
import requests
from bs4 import BeautifulSoup
import re
import csv
from alive_progress import alive_bar

LINK = "https://catalog.colorado.edu/courses-a-z/"
cwd = os.getcwd()

# Empty the big csv file on execution so that it doesn't become enormous by accident
with open(cwd + "/courses/*.csv", "w") as f:
    pass

# Get links to all course subject pages
html = requests.get(LINK).text
soup = BeautifulSoup(html, "html.parser")

lines = soup.find_all("a", attrs={"href": re.compile("/courses-a-z/\w{4}/")})
links = [i["href"] for i in lines]
subjects = [i["href"][-5:-1].upper() for i in lines] # Get 4 letter subject codes for each subject

# Create individual files for all subjects
with alive_bar(len(links)) as bar:
    for i, link in enumerate(links):

        # Get text from subject page
        html = requests.get("https://catalog.colorado.edu/" + link).text
        soup = BeautifulSoup(html, "html.parser")

        # Temporary lists
        codes = []
        names = []
        descs = []

        # Get lines containing course codes/names
        lines = soup.find_all(class_="courseblock")
        for j in lines:
            course = j.find("strong").text

            # Course code
            code = course[:9]
            code = re.sub(r"\xa0|\"+", "", code)
            code = re.sub(r"\s\s+", " ", code)
            code = re.sub(r"(\w)(\d)", r"\1 \2", code)
            code = re.sub(r"(\d) (\d)", r"\1\2", code)

            # Course name
            name = course[(course.index(")") + 2):]
            name = re.sub(r"\xa0|\"+", "", name)
            name = re.sub(r"\s\s+", " ", name)

            codes.append(code)
            names.append(name)

        # Get lines containing course descriptions
        lines = soup.find_all(class_="courseblockdesc noindent")
        for j in lines:
            desc = j.text[1:]
            desc = re.sub(r"\xa0|\"+", "", desc)
            desc = re.sub(r"\s\s+", " ", desc)
            desc = re.sub(r"(\w)(\d)", r"\1 \2", desc)
            desc = re.sub(r"(\d) (\d)", r"\1\2", desc)
            descs.append(desc)

        # Join temporary lists
        courses = [[codes[j], names[j], descs[j]] for j in range(len(codes))]

        # Write courses to subject csv file
        with open(cwd + "/courses/" + subjects[i] + ".csv", "w") as f:
            csvwriter = csv.writer(f)
            for c in courses:
                csvwriter.writerow(c)
        
        # Append courses to main csv file
        with open(cwd + "/courses/*.csv", "a") as f:
            csvwriter = csv.writer(f)
            for c in courses:
                csvwriter.writerow(c)
        
        # Update progress bar
        bar()
