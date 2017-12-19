from sklearn.feature_extraction.text import TfidfVectorizer
import os
import math
from shutil import copy2
from nltk import word_tokenize
import re
import string
import numpy as np
np.set_printoptions(threshold=np.inf)

texts_folder = "Texts/"
proc_folder = "Processed/"
comb_folder = "Combined/"
samples_folder = "Samples/"
n = 10000
# n = 10

def parse(folder, filename):
    """Perform a variety of tasks to parse the raw text into usable format
    1. Ignore first 10% and last 6% of the book, as per Bamman et al.
    2. Remove first and last sentences from each page of each book.
    """
    # Check if the book has actually been downloaded
    path = texts_folder + folder + "/" + filename
    proc_path = proc_folder + folder + "/" + filename
    if len(os.listdir(path)) == 0:
        print(filename + " has not been downloaded")
        return

    # We check if the file already has been parsed
    if os.path.exists(proc_path) and len(os.listdir(proc_path)) > 0:
        print(filename + " has already been processed.")
        return

    # Compute cutoffs
    print("Performing cutoff for book " + filename + " in folder " + folder)
    numpages = len(os.listdir(path))
    start_cut = int(math.ceil(0.1 * numpages))
    end_cut = int(math.ceil(0.06 * numpages))

    # Delete docs before start_cut and after end_cut
    if not os.path.exists(proc_path):
        os.makedirs(proc_path)

    for i in range(start_cut, numpages - end_cut):
        copy2(path + "/" + str(i) + '.txt', proc_path)

    # Remove first and last sentence from each document
    for i in range(start_cut, numpages - end_cut):
        with open(proc_path + '/' + str(i) + '.txt', encoding='utf8') as fin:
            page = fin.read().splitlines(True)
        with open(proc_path + '/' + str(i) + '.txt', 'w', encoding='utf8') as fout:
            fout.writelines(page[1:-1])


def create_combined(folder):
    """Combines all books from a time period into a single document in the 'Combined' directory"""
    # Check if document already exists
    if os.path.exists(comb_folder + folder + '/document.txt'):
        print("Combined document already exists for " + folder)
        return

    # Open file and write all page from all books to it
    print("Combining files for folder " + folder)

    path = comb_folder + folder + "/document.txt"
    if os.path.exists(path):
        append_write = 'a'
    else:
        append_write = 'w'

    f = open(path, append_write, encoding='utf8')
    for book in os.listdir(proc_folder + folder):
        for page in os.listdir(proc_folder + folder + "/" + book):
            with open(proc_folder + folder + "/" + book + "/" + page, encoding='utf8') as p:
                lines = p.read()
                tok_lines = tokenize(lines)  # Tokenize before writing
                f.writelines(["%s " % word for word in tok_lines])
    f.close()


def tokenize(text):
    # Remove punctuation and digits
    no_dig = re.sub(u'[0-9]', "", text)

    punc = string.punctuation
    pattern = r"[{}]".format(punc)

    no_punc = re.sub(pattern, "", no_dig)
    no_punc = no_punc.lower()
    return word_tokenize(no_punc)


def create_vocabulary():
    """Process the words with TFIDF across the time domains and get top tfidf terms to make our vocabulary"""
    vectorizer = TfidfVectorizer(input='filename',
                                 encoding='utf-8',
                                 strip_accents='unicode',
                                 decode_error='ignore',
                                 analyzer='word',
                                 tokenizer=tokenize,
                                 lowercase=True,
                                 ngram_range=(1, 1),
                                 stop_words='english',
                                 max_features=100000,
                                 norm='l2',
                                 smooth_idf=True)

    # Create all the file paths
    docs = []
    for i in range(1725, 1925, 25):
        foldername = comb_folder + "{}-{}/document.txt".format(i, i + 25)
        docs.append(foldername)

    # Compute TFIDF
    print("Beginning TFIDF computations")
    X = vectorizer.fit_transform(docs)

    # Get top words
    feature_array = np.array(vectorizer.get_feature_names())
    tfidf_sorting = np.argsort(X.toarray()).flatten()[::-1]

    top_n = feature_array[tfidf_sorting][:n]
    print ("finished TFIDF computations")
    return top_n


def get_vocab_len():
    return n

# ------------- Main script begins --------------- #
if __name__ == "__main__":
    for folder in os.listdir(texts_folder):
        for bookid in os.listdir(texts_folder + '/' + folder):
            parse(folder, bookid)

        create_combined(folder)

