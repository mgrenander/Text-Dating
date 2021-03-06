import data_preprocessor
from keras.utils import to_categorical
import numpy as np
import pickle
import os
from scipy import sparse


class SampleCreator:
    vocabulary = dict()
    sample_size = 0
    mapping = []
    size_mapping = []
    num_categories = 0

    def __init__(self, sample_size, num_categories):
        vocab_list = data_preprocessor.create_vocabulary()
        self.sample_size = sample_size
        self.num_categories = num_categories
        self.size_mapping = [0] * num_categories

        for i in range(0, 8):
            start = i*25 + 1725
            end = (i+1)*25 + 1725
            self.mapping.append("{}-{}".format(start, end))

        for index, item in enumerate(vocab_list):
            self.vocabulary[item] = index

        # self.mapping.append("1650-1675")
        # So now mapping[8] = "1650..."

    def correct_input(self, raw_sample):
        """Creates sparse matrix for a single entry"""
        sentence_matrix = []
        for word in raw_sample:
            word_vec = [0] * len(self.vocabulary)
            try:
                word_vec[self.vocabulary[word]] = 1
            except KeyError:
                # Word was not found
                pass

            sentence_matrix.append(word_vec)
        return sparse.coo_matrix(np.array(sentence_matrix, dtype=np.int16))

    def get_samples(self, category):
        # Load data from category
        with open("Combined/" + str(self.mapping[category]) + "/document.txt") as f:
            all_words = data_preprocessor.tokenize(f.read().decode("UTF-8"))
            samples = []
            for i in range(0, len(all_words), self.sample_size):
                # Ensure we don't include the last one, we may not be of size sample_size
                if i + self.sample_size >= len(all_words):
                    break

                pre_sample = all_words[i:i+self.sample_size]
                samples.append(self.correct_input(pre_sample))

        # Update the size mapping
        self.size_mapping[category] = len(samples)
        return samples

    def get_label(self, category):
        # Returns label for the samples
        num_samples = self.size_mapping[category]
        one_label = to_categorical(category, self.num_categories).astype(np.int16, copy=False)

        labels = []
        for i in range(0, num_samples):
            labels.append(one_label)
        return labels

    def get_vocab_len(self):
        """Returns length of vocabulary"""
        return len(self.vocabulary)


if __name__ == "__main__":
    # Create samples and pickle data
    sc = SampleCreator(400, 8)

    # Create pickle folder
    if not os.path.exists("Pickles"):
        os.makedirs("Pickles")

    samples_list = []
    labels_list = []
    for i in range(0, sc.num_categories):
        print("Computing sample values at category " + str(i))
        samples_list += sc.get_samples(i)
        labels_list += sc.get_label(i)

    # TESTING
    # all_samples = np.array(sc.get_samples(8))
    # all_labels = sparse.coo_matrix(np.array(sc.get_label(8)))
    # pickle_all = open("Pickles/test_pickle.pickle", "wb")

    # PRODUCTION
    all_samples = np.array(samples_list)
    all_labels = sparse.coo_matrix(np.array(labels_list))
    pickle_all = open("Pickles/pickle_all.pickle", "wb")

    # Dump pickle
    pickle.dump((all_samples, all_labels), pickle_all, protocol=2)
