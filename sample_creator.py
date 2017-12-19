
import data_preprocessor
from keras.utils import to_categorical
import numpy as np
import pickle
import os

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

        for i in range(0,8): # TODO: change this after done testing with 1650-1675
            start = i*25 + 1725
            end = (i+1)*25 + 1725
            self.mapping.append("{}-{}".format(start,end))

        for index, item in enumerate(vocab_list):
            self.vocabulary[item] = index

        self.mapping.append("1650-1675")
        # So now mapping[8] = "1650..."


    def correct_input(self, raw_sample):
        """Given a list of words, creates an array where each entry represents the one-hot encoding of one word"""
        sentence_matrix = []
        for word in raw_sample:
            word_vec = [0] * len(self.vocabulary)
            try:
                word_vec[self.vocabulary[word]] = 1
            except KeyError:
                # Word was not found
                pass

            sentence_matrix.append(word_vec)
        return sentence_matrix


    def get_samples(self, category):
        # Load data from category
        with open("Combined/" + str(self.mapping[category]) + "/document.txt", encoding='utf8') as f:
            all_words = data_preprocessor.tokenize(f.read())
            samples = []
            for i in range(0, len(all_words), self.sample_size):
                pre_sample = all_words[i:i+self.sample_size]
                samples.append(self.correct_input(pre_sample))

        # Update the size mapping
        self.size_mapping[category] = len(samples)
        return samples


    def get_label(self, category):
        # Returns label for the samples
        num_samples = self.size_mapping[category]
        one_label = to_categorical(category, self.num_categories)

        labels = []
        for i in range(0, num_samples):
            labels.append(one_label)
        return labels


    def get_vocab_len(self):
        """Returns length of vocabulary"""
        return len(self.vocabulary)

if __name__ == "__main__":
    # Create samples and pickle data
    sc = SampleCreator(5, 9)

    # all_samples = []
    # all_labels = []
    # for i in range(0,7):
    #     print("Computing sample values at category " + str(i))
    #     all_samples += sc.get_samples(i)
    #     all_labels += sc.get_label(i)

    # # Convert all_labels to numpy array
    # all_labels = np.array(all_labels)

    # TESTING
    test_sample = sc.get_samples(8)
    test_labels = np.array(sc.get_label(8))

    ##### PICKLING
    # Create pickle folder
    if not os.path.exists("Pickles"):
        os.makedirs("Pickles")

    # Pickle only the test sample
    pickle_test = open("Pickles/test_pickle.pickle", "wb")
    pickle.dump((test_sample, test_labels), pickle_test)

    # Pickle all samples
    # pickle_all = open("Pickles/samples_labels.pickle", "wb")
    # pickle.dump((all_samples, all_labels), pickle_all)
