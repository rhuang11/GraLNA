import csv
import glob
import re
import string
import sys
import datetime as dt


def utf8len(s):
    """helper function to get the size of string"""
    return len(s.encode("utf-8"))


# Load your master dictionary file. This file requires a
# Word column and a Syllables column. Other columns are optional
# and should be defined in the SENTIMENT_OUTPUT_FIELDS Python dictionary below.
master_dictionary_file = "Loughran-McDonald_MasterDictionary_1993-2021.csv"

# The SENTIMENT_OUTPUT_FIELDS dictionary below contains the sentiment fields we want
# to include. The names below must exactly match the column names in the master
# dictionary file.
SENTIMENT_OUTPUT_FIELDS = {
    "Negative": 1,
    "Positive": 1,
    "Uncertainty": 1,
    "Litigious": 1,
    "Strong_Modal": 1,
    "Weak_Modal": 1,
    "Constraining": 1,
}

# Load the master dictionary CSV file into a Python dictionary
# with Word as the key.
master_dictionary = {}
with open(master_dictionary_file) as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=",")
    line_count = 0
    for row in csv_reader:
        master_dictionary[row["Word"]] = row
        line_count += 1
print(f"master dictionary has {len(master_dictionary)} words.")

# The following output fields are available by default.
FIXED_OUTPUT_FIELDS = [
    "Accession_No",  # 0
    "CIK",  # 1
    "Filing_Date",  # 2
    "Text_Size (Bytes)",  # 3
    "Number_of_Words",  # 4
    "Number_of_Alphabetic",  # 5
    "Number_of_Digits",  # 6
    "Number_of_Numbers",  # 7
    "Average_Syllables",  # 8
    "Average_Word_Length",  # 9
    "Vocabulary",  # 10
]
original_len_output = len(FIXED_OUTPUT_FIELDS)

# The sentiment columns are added dynamically.
# (As of Python 3.6, for the CPython implementation of Python, dictionaries remember the order of items inserted.)
for key, item in SENTIMENT_OUTPUT_FIELDS.items():
    FIXED_OUTPUT_FIELDS.append(f"{key}")

# Create all of the column names.
data = []
data.append(FIXED_OUTPUT_FIELDS)

for result in results:
    text = result[5]
    cik = result[6][0]["cik"]
    filing_date = result[2]
    # Customize tokenization here.
    tokens = re.findall("\w+", text)  # Note that \w+ splits hyphenated words.
    vocabulary = {}
    # Setup initial placeholders.
    output_data = [0] * len(FIXED_OUTPUT_FIELDS)
    output_data[0] = result[0]  # Accession_No
    output_data[1] = cik  # CIK
    output_data[2] = filing_date  # Filing_Date
    total_tokens = 0
    output_data[3] = utf8len(text)  # Text_Size

    output_data[5] = len(re.findall("[A-Z]", text))  # Number_of_Alphabetic
    output_data[6] = len(re.findall("[0-9]", text))  # Number_of_Digits
    # Drop punctuation within numbers for number count.
    number_doc = re.sub("(?!=[0-9])(\.|,)(?=[0-9])", "", text)
    number_doc = number_doc.translate(
        str.maketrans(string.punctuation, " " * len(string.punctuation))
    )
    output_data[7] = len(
        re.findall(r"\b[-+\(]?[$€£]?[-+(]?\d+\)?\b", number_doc)
    )  # Number_of_Numbers

    total_syllables = 0
    word_length = 0
    number_of_words = 0

    for token in tokens:
        if (
            not token.isdigit()
            and len(token) > 1
            and master_dictionary.get(token) is not None
        ):
            total_tokens += 1
            word_length += len(token)

            if token not in vocabulary:
                vocabulary[token] = 1

            total_syllables += int(master_dictionary[token]["Syllables"])

            for key, item in SENTIMENT_OUTPUT_FIELDS.items():
                if (
                    master_dictionary[token][key] != "0"
                    and master_dictionary[token][key] != 0
                    and master_dictionary[token][key] != None
                ):
                    output_data[FIXED_OUTPUT_FIELDS.index(key)] += item

    output_data[4] = total_tokens  # Number_of_Words
    output_data[8] = total_syllables / total_tokens  # Average_Syllables
    output_data[9] = word_length / total_tokens  # Average_Word_Length
    output_data[10] = len(vocabulary)  # Vocabulary

    # Convert values for various columns to %
    for i in range(original_len_output, len(output_data)):
        output_data[i] = (output_data[i] / total_tokens) * 100

    print(f"finished {result[0]}")
    data.append(output_data)