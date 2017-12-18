
import data_preprocessor
from keras.utils import to_categorical
import numpy as np


class SampleCreator:
    vocabulary = []
    sample_size = 0
    mapping = []
    size_mapping = []

    def __init__(self, sample_size):
        self.vocabulary = data_preprocessor.create_vocabulary()
        self.sample_size = sample_size
        self.size_mapping = [0] * 8

        for i in range(0,8):
            start = i*25 + 1725
            end = (i+1)*25 + 1725
            self.mapping.append("{}-{}".format(start,end))

        self.mapping.append("1650-1675")
        # So now mapping[8] = "1650..."


    def correct_input(self, sent):
        dic = {}
        for index, item in enumerate(self.vocabulary):
            dic[item] = index

        sentence_matrix = []
        for word in sent:
            temp = [0] * len(self.vocabulary)
            try:
                temp[dic[word]] = 1
            except KeyError:
                pass
            sentence_matrix.append(temp)
        matrix= np.array(sentence_matrix)
        # print matrix
        # sys.exit()
        # result_sentences=np.array(result_sentences)
        return matrix


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
        return [to_categorical([category],8)] * num_samples


    def get_vocab_len(self):
        """Returns length of vocabulary"""
        return len(self.vocabulary)