import numpy as np
import os

from pba.data_preprocessing import BERT_preprocess
from pba.bert_features import convert_data_to_features
from pba.bert_features import load_vocab, wordpiece_tokenize

MAX_SEQ_LENGTH = 128
EMBEDDING_DIM = 768
BATCH_SIZE = 64

def convert_id_to_token(vocab, id):
    # from [https://stackoverflow.com/questions/8023306/get-key-by-value-in-dictionary]
    return list(vocab.keys())[list(vocab.values()).index(id)]


def get_bert_embeddings(list_of_sentences, n_labels, model, sess):
    # --- tokenize sentences
    tokens = BERT_preprocess(list_of_sentences)

    # --- get embeddings
    n_batches = int(len(tokens) / BATCH_SIZE)
    if len(tokens) % BATCH_SIZE != 0:
        n_batches += 1

    bert_vocab_file = os.path.join(os.path.realpath('../../..'),'datasets','pretrained_models','bert_base','vocab.txt')
    bert_vocab = load_vocab(bert_vocab_file)

    results = []
    for i in range(n_batches):
        batch_tokens = tokens[i * model.batch_size:(i + 1) * model.batch_size]


        dummy_labels = np.zeros((len(batch_tokens), n_labels))
        dummy_label_list = list(np.arange(n_labels))
        features = convert_data_to_features(batch_tokens, dummy_labels, label_list=dummy_label_list,
                                            max_seq_length=MAX_SEQ_LENGTH)

        input_ids, input_mask, token_type_ids = [], [], []
        for el in features:
            input_ids.append(el.input_ids)
            input_mask.append(el.input_mask)
            token_type_ids.append(el.input_type_ids)

        input_ids = np.array(input_ids, np.float32)
        input_mask = np.array(input_mask, np.float32)
        token_type_ids = np.array(token_type_ids, np.float32)

        noise_vector = np.zeros((len(batch_tokens), MAX_SEQ_LENGTH, EMBEDDING_DIM))

        batch_embeddings = sess.run(
            model.embedding_output,
            feed_dict={
                model.input_ids: input_ids,
                model.input_mask: input_mask,
                model.token_type_ids: token_type_ids,
                model.labels: dummy_labels,
                model.noise_vector: noise_vector
            })

        # Handle wordpiece tokens
        for sent_ii in range(len(batch_tokens)):
            tokens_list = []
            embeddings_list = []
            oov_len = 1

            wp_tokens = []
            for token in batch_tokens[sent_ii]:
                wp_tokens += wordpiece_tokenize(token, bert_vocab)

            sent_tokens = []
            sent_tokens.append("[CLS]")
            for token in wp_tokens:
                sent_tokens.append(token)
            if len(sent_tokens) >= MAX_SEQ_LENGTH:
                sent_tokens = sent_tokens[:MAX_SEQ_LENGTH-1]
            sent_tokens.append("[SEP]")

            assert(len(sent_tokens)==int(np.sum(input_mask[sent_ii])))

            for token_ii in range(len(sent_tokens)):
                token = sent_tokens[token_ii]
                if token == '[PAD]':
                    break
                if token in ['[CLS]','[SEP]']:
                    continue
                if token.startswith('##'):
                    token = token[2:]
                    tokens_list[-1] += token
                    embeddings_list[-1] += batch_embeddings[sent_ii, token_ii, :]
                    oov_len += 1
                else:
                    if oov_len > 1:
                        embeddings_list[-1] /= oov_len
                        oov_len = 1
                    tokens_list.append(token)
                    embeddings_list.append(batch_embeddings[sent_ii, token_ii, :])
            if oov_len > 1:  # if the whole sentence is one oov, handle this special case
                embeddings_list[-1] /= oov_len

            results.append((tokens_list, embeddings_list))

    return results
