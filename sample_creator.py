
import data_preprocessor
from keras.utils import to_categorical

class SampleCreator:
    vocabulary = []
    sample_size = 0
    mapping = []

    def __init__(self, sample_size):
        self.vocabulary = data_preprocessor.create_vocabulary()
        self.sample_size = sample_size

        for i in range(0,8):
            start = i*25 + 1725
            end = (i+1)*25 + 1725
            self.mapping.append("{}-{}".format(start,end))

    def correct_input(self, sentences):
        dic = {}
        for index, item in enumerate(self.vocabulary):
            dic[item] = index
        result_sentences = []
        for sent in sentences:
            sentence_matrix = []
            for word in sent:
                temp = [0] * len(self.vocabulary)
                temp[dic[word]] = 1
                sentence_matrix.append(temp)
            # matrix= np.array(sentence_matrix)
            matrix = sentence_matrix
            result_sentences.append(matrix)
        # result_sentences=np.array(result_sentences)
        return result_sentences


    def get_samples(self, category):
        # Load data from category
        #with open("Combined/" + str(self.mapping[category]) + "/document.txt", encoding='utf8') as f:
        with open("Combined/" + str(self.mapping[category]) + "/document.txt", 'r') as f:
            #all_words = data_preprocessor.tokenize(f.read())
            all_words = data_preprocessor.tokenize(f.read().decode("UTF-8"))
            samples = []
            for i in range(0, len(all_words), self.sample_size):
                pre_sample = all_words[i:i+self.sample_size]
                samples.append(self.correct_input(pre_sample))

    def get_label(self, category):
        '''
        input category: a single int 
        '''
        # TODO: get Seara's help for this part
        # encoding_list = []
        # for i in range(0,8):
        #     if i == category:
        #         encoding_list.append(1)
        #     else:
        #         encoding_list.append(0)
        #return encoding_list
        return to_categorical([category],8)



