from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class TFIDF(object):

    def __init__(self, db, vocab, ivocab, stopwords=None):
        self.db = [' '.join(item[1]['abstract']) for item in db]
        self.tfidf_vec = TfidfVectorizer(stop_words=stopwords, 
                vocabulary=vocab, sublinear_tf=False)
        self.tfs = self.tfidf_vec.fit_transform(self.db).toarray()
        self.vocab = vocab
        self.ivocab = ivocab

    def get_tfs(self, w, start):
        '''
        NOTE: data should NOT be shuffled when using TFIDF

        @params
            w: numpy array, padded words in batch
            start: int, start index to retrieve tfidfs in self.tfs
        '''
        batch_size, max_len = w.shape
        tfs = self.tfs[start:start+batch_size]
        weights = np.zeros(w.shape)

        ws = w - 1
        for i in range(batch_size):
            idx = np.where(ws[i] >= 0)[0]
            weights[i,:len(idx)] = tfs[i][ws[i][idx]]
        return weights

