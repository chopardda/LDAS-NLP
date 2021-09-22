import numpy as np

from tqdm.auto import tqdm
from sklearn.neighbors import KDTree

from pba.get_bert_embedding import get_bert_embeddings

class ContextNeighborStorage:
    '''
    Class to build k-d tree from embeddings (create search index) and to query nearest neighbours
    (from [https://gist.github.com/avidale/c6b19687d333655da483421880441950])
    '''
    def __init__(self, sentences, n_labels, model, session):
        self.sentences = sentences
        self.n_labels = n_labels
        self.model = model
        self.session = session

    def process_sentences(self):
        result = get_bert_embeddings(self.sentences, self.n_labels, self.model, self.session)

        self.sentence_ids = []
        self.token_ids = []
        self.all_tokens = []
        all_embeddings = []
        for i, (toks, embs) in enumerate(tqdm(result)):
            for j, (tok, emb) in enumerate(zip(toks, embs)):
                self.sentence_ids.append(i)
                self.token_ids.append(j)
                self.all_tokens.append(tok)
                all_embeddings.append(emb)
        all_embeddings = np.stack(all_embeddings)
        # we normalize embeddings, so that euclidian distance is equivalent to cosine distance
        self.normed_embeddings = (all_embeddings.T / (all_embeddings**2).sum(axis=1) ** 0.5).T

    def build_search_index(self):
        # this takes some time
        self.indexer = KDTree(self.normed_embeddings)

    def query(self, query_sent, query_word, k=10, filter_same_word=False, different_neighbours=False):
        toks, embs = get_bert_embeddings([query_sent], self.n_labels, self.model, self.session)[0]

        found = False
        for tok, emb in zip(toks, embs):
            if tok == query_word:
                found = True
                break
        if not found:
            raise ValueError('The query word {} is not a single token in sentence {}'.format(query_word, toks))
        emb = emb / sum(emb**2)**0.5

        if filter_same_word:
            k_max = 100
            while True:
                initial_k = max(k, k_max)
                di, idx = self.indexer.query(emb.reshape(1, -1), k=initial_k)
                k_nn = [self.all_tokens[ii] for ii in idx.ravel()]
                if len(list(set(k_nn))) > 1:
                    break
                k_max += 100
        else:
            initial_k = k
            di, idx = self.indexer.query(emb.reshape(1, -1), k=initial_k)

        distances = []
        neighbours = []
        contexts = []
        for i, index in enumerate(idx.ravel()):
            token = self.all_tokens[index]
            if filter_same_word and (query_word in token or token in query_word):
                continue
            if different_neighbours and token in neighbours:
                continue
            distances.append(di.ravel()[i])
            neighbours.append(token)
            contexts.append(self.sentences[self.sentence_ids[index]])
            if len(distances) == k:
                break

        return distances, neighbours, contexts