from sklearn.feature_extraction.text import TfidfVectorizer
import os
import math
from shutil import copy2
import numpy as np

texts_folder = "Texts/"
proc_folder = "Processed/"
comb_folder = "Combined/"
samples_folder = "Samples/"


def parse(folder, filename):
    """Perform a variety of tasks to parse the raw text into usable format
    1. Ignore first 10% and last 6% of the book, as per Bamman et al.
    2. Remove first and last sentences from each page of each book.
    """
    # Check if the book has actually been downloaded
    path = texts_folder + folder + "/" + filename
    if len(os.listdir(path)) == 0:
        print(filename + " has not been downloaded")
        return

    # We check if the file already has been parsed
    if len(os.listdir(proc_folder + "/parsed.txt")) > 0:
        print(filename + " has already been processed.")
        return

    # Compute cutoffs
    print("Performing cutoff for book " + filename + " in folder " + folder)
    numpages = len(os.listdir(path))
    start_cut = int(math.ceil(0.1 * numpages))
    end_cut = int(math.ceil(0.06 * numpages))

    # Delete docs before start_cut and after end_cut
    for i in range(start_cut, end_cut):
        copy2(path + "/" + str(i) + '.txt', proc_folder + folder + "/" + filename)

    # Remove first and last sentence from each document
    for i in range(start_cut + 1, numpages - end_cut - 1):
        with open(path + '/' + str(i) + '.txt') as fin:
            page = fin.read().splitlines(True)
        with open(path + '/' + str(i) + '.txt') as fout:
            fout.writelines(page[1:])

    # Put a marker saying we have parsed this document
    f = open(path + "/parsed.txt", 'w')
    f.close()


def create_combined(folder):
    """Combines all books from a time period into a single document in the 'Combined' directory"""
    # Check if document already exists
    if os.path.exists(comb_folder + folder + '/document.txt'):
        print("Combined document already exists for " + folder)
        return

    # Open file and write all page from all books to it
    f = open(comb_folder + folder + '/document.txt', 'a')
    for book in os.listdir(proc_folder + folder):
        for page in os.listdir(proc_folder + book):
            f.write(page)
    f.close()


def create_vocabulary():
    """Process the words with TFIDF across the time domains and get top tfidf terms to make our vocabulary"""
    vectorizer = TfidfVectorizer(input='content',
                                 encoding='ignore',
                                 strip_accents='unicode',
                                 analyzer='word',
                                 ngram_range=(1, 1),
                                 token_pattern='(?u)\S\S+',
                                 max_features=100000,
                                 norm='l2',
                                 smooth_idf=True)

    # Create all the file paths
    docs = []
    for i in range(1750, 1925, 25):
        foldername = comb_folder + "/{}-{}/document.txt".format(i, i + 25)
        docs.append(foldername)

    # Compute TFIDF
    vectorizer.fit_transform(docs)

    # Get top words
    indices = np.argsort(vectorizer.idf_)[::-1]
    features = vectorizer.get_feature_names()
    top_features = [features[i] for i in indices[:10000]]

    return top_features


def create_samples(book_id, vocabulary):
    """For a given book, create samples for this book, i.e. create the one-hot matrix encoding"""
    pass

# ------------- Main script begins --------------- #
for folder in os.listdir(texts_folder):
    for bookid in os.listdir(texts_folder + '/' + folder):
        parse(folder, bookid)

    create_combined(folder)

vocabulary = create_vocabulary()

for folder in os.listdir(texts_folder):
    for bookid in folder:
        create_samples(bookid, vocabulary)
