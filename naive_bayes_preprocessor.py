from sample_creator import SampleCreator
from scipy import sparse
import numpy as np
import os
import pickle

class NaiveBayesSampleCreator(SampleCreator):
    """Similar to SampleCreator but we are preprocessing data in order to feed it into a Naive Bayes Classifier"""

    def correct_input(self, raw_sample):
        """Creates np array for a single sample"""
        sentence_array = [0] * len(self.vocabulary)
        for word in raw_sample:
            try:
                # This time increment to get the raw frequency of the word
                sentence_array[self.vocabulary[word]] += 1
            except KeyError:
                # Word was not found
                pass

        return np.array(sentence_array)


    def get_label(self, category):
        # return np.full(self.sample_size, category, dtype=np.int8)
        return [category] * self.size_mapping[category]

if __name__ == "__main__":
    nbsc = NaiveBayesSampleCreator(400, 8)

    # Create pickle folder
    if not os.path.exists("Pickles"):
        os.makedirs("Pickles")

    samples_list = []
    labels_list = []
    for i in range(0, nbsc.num_categories):
        print("Computing sample values at category " + str(i))
        samples_list += nbsc.get_samples(i)
        labels_list += nbsc.get_label(i)

    # PRODUCTION
    all_samples = sparse.coo_matrix(np.array(samples_list))
    all_labels = np.array(labels_list)
    pickle_all = open("Pickles/nb.pickle", "wb")

    # TESTING
    # all_samples = sparse.coo_matrix(np.array(nbsc.get_samples(8)))
    # all_labels = np.array(nbsc.get_label(8))
    # pickle_all = open("Pickles/nb_test.pickle", "wb")

    # Dump pickle
    pickle.dump((all_samples, all_labels), pickle_all, protocol=2)
