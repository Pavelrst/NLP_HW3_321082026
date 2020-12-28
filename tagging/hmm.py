import os
import time
from data import *
from collections import defaultdict, Counter
import math


class Probabilities():
    def __init__(self, e_tag_counts,
                 e_word_tag_counts,
                 q_tri_counts,
                 q_bi_counts,
                 q_uni_counts,
                 total_tokens,
                 lambda1=0.6,
                 lambda2=0.3,
                 lambda3=0.1):
        self.e_tag_counts = e_tag_counts
        self.e_word_tag_counts = e_word_tag_counts
        self.q_tri_counts = q_tri_counts
        self.q_bi_counts = q_bi_counts
        self.q_uni_counts = q_uni_counts
        self.total_tokens = total_tokens
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

    def e_proba(self, word, tag):
        if (word, tag) in e_word_tag_counts.keys():
            return e_word_tag_counts[(word, tag)] / float(e_tag_counts[tag])
        else:
            return 0

    def q_proba(self, tag1, tag2, tag3):
        proba = self.unigram_prob(tag1)
        proba += self.bigram_prob(tag1, tag2)
        proba += self.trigram_prob(tag1, tag2, tag3)
        return proba

    def trigram_prob(self, tag1, tag2, tag3):
        assert 0 <= self.lambda3 < 1
        if (tag1, tag2, tag3) in q_tri_counts.keys():
            return self.lambda3 * q_tri_counts[(tag1, tag2, tag3)] / q_bi_counts[(tag1, tag2)]
        else:
            return 0

    def bigram_prob(self, tag1, tag2):
        assert 0 <= self.lambda2 < 1
        if (tag1, tag2) in q_bi_counts.keys():
            return self.lambda2 * q_bi_counts[(tag1, tag2)] / q_uni_counts[tag1]
        else:
            return 0

    def unigram_prob(self, tag1):
        assert 0 <= self.lambda1 < 1
        if tag1 in q_uni_counts.keys():
            return self.lambda1 * q_uni_counts[tag1] / total_tokens
        else:
            return 0

class Layer():
    def __init__(self):
        self.layer_dict = defaultdict(lambda: -math.inf)
        self.back_pointers = {}

    def set_proba(self, tag1, tag2, log_proba):
        self.layer_dict[(tag1, tag2)] = log_proba

    def get_proba(self, tag1, tag2):
        return self.layer_dict[(tag1, tag2)]

    def set_back_pointer(self, tag1, tag2, back_tag):
        self.back_pointers[(tag1, tag2)] = back_tag

    def get_back_pointers(self):
        return self.back_pointers

    def drop_lows(self):
        for key in self.layer_dict.keys():
            if self.layer_dict[key] < -math.inf:
                del self.layer_dict[key]



class ViterbiGraph():
    def __init__(self, proba_fn):
        self.proba_fn = proba_fn
        self.all_tags = e_tag_counts.keys()
        self.e_tag_counts = e_tag_counts
        self.total_back_pointers_list = []

    def compare_fn(self, input_tuple):
        log_proba = input_tuple[1]
        tags_tuple = input_tuple[0]
        tag1 = tags_tuple[0]
        tag2 = tags_tuple[1]
        return log_proba + math.log(self.proba_fn.q_proba(tag1, tag2, 'STOP'))

    def predict(self, sent):
        layer = Layer()
        layer.set_proba('START', 'START', 0)

        for word in sent:
            new_layer = Layer()
            for v in self.all_tags:
                if not self.proba_fn.e_proba(word=word, tag=v) == 0:
                    for (w, u), prev_log_prob in layer.layer_dict.items():
                        e_proba = self.proba_fn.e_proba(word=word, tag=v)
                        q_proba = self.proba_fn.q_proba(tag1=w, tag2=u, tag3=v)
                        log_prob = prev_log_prob + \
                                   math.log(q_proba) + \
                                   math.log(e_proba)

                        if log_prob > new_layer.get_proba(u, v):
                            new_layer.set_proba(u, v, log_prob)
                            new_layer.set_back_pointer(u, v, w)

            self.total_back_pointers_list.append(new_layer.get_back_pointers())
            new_layer.drop_lows()
            layer = new_layer

        final_pair = max(layer.layer_dict.items(), key=self.compare_fn)[0]
        predicted_tags = list(final_pair)

        # back track from the end, excluding first two 'start' tags
        reverse_pointers = self.total_back_pointers_list[::-1]
        reverse_pointers = reverse_pointers[:-2]  #remove 'start' tags
        for current_back_pointers in reverse_pointers:
            w = current_back_pointers[tuple(predicted_tags[:2])]
            predicted_tags.insert(0, w)
        if predicted_tags[0] == 'START':
            predicted_tags = predicted_tags[1:]
        return predicted_tags

def hmm_train(sents):
    """
        sents: list of tagged sentences
        Returns: the q-counts and e-counts of the sentences' tags, total number of tokens in the sentences
    """

    print("Start training")
    total_tokens = 0
    # YOU MAY OVERWRITE THE TYPES FOR THE VARIABLES BELOW IN ANY WAY YOU SEE FIT

    ### YOUR CODE HERE
    e_tag_counts = {}
    e_word_tag_counts = {}

    for sent in sents:
        total_tokens += len(sent) + 1
        for word, tag in sent:
            if (word, tag) in e_word_tag_counts.keys():
                e_word_tag_counts[(word, tag)] += 1
            else:
                e_word_tag_counts[(word, tag)] = 1

            if tag in e_tag_counts.keys():
                e_tag_counts[tag] += 1
            else:
                e_tag_counts[tag] = 1

    q_tri_counts = {}
    q_bi_counts = {}
    q_uni_counts = {}

    for sent in sents:
        tags = ['START'] + [tag for _, tag in sent] + ['STOP']

        for tag in tags:
            if tag in q_uni_counts.keys():
                q_uni_counts[tag] += 1
            else:
                q_uni_counts[tag] = 1

        tags1 = ['START', 'START'] + [tag for _, tag in sent]
        tags2 = ['START'] + [tag for _, tag in sent] + ['STOP']
        assert len(tags1) == len(tags2)

        for tag1, tag2 in zip(tags1, tags2):
            if (tag1, tag2) in q_bi_counts.keys():
                q_bi_counts[(tag1, tag2)] += 1
            else:
                q_bi_counts[(tag1, tag2)] = 1

        tags1 = ['START', 'START'] + [tag for _, tag in sent[:-1]]
        tags2 = ['START'] + [tag for _, tag in sent]
        tags3 = [tag for _, tag in sent] + ['STOP']
        assert len(tags1) == len(tags2)
        assert len(tags1) == len(tags3)

        for tag1, tag2, tag3 in zip(tags, tags, tags):
            if (tag1, tag2, tag3) in q_tri_counts.keys():
                q_tri_counts[(tag1, tag2, tag3)] += 1
            else:
                q_tri_counts[(tag1, tag2, tag3)] = 1

    ### END YOUR CODE
    return total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts


def hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                e_word_tag_counts, e_tag_counts, lambda1, lambda2):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Returns: predicted tags for the sentence
    """
    predicted_tags = ["O"] * (len(sent))
    ### YOUR CODE HERE

    proba_fn = Probabilities(e_tag_counts,
                 e_word_tag_counts,
                 q_tri_counts,
                 q_bi_counts,
                 q_uni_counts,
                 total_tokens,
                 lambda1,
                 lambda2,
                 1-lambda1-lambda2)
    graph = ViterbiGraph(proba_fn)
    predicted_tags = graph.predict(sent)

    ### END YOUR CODE
    return predicted_tags

def hmm_eval(test_data, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm
    """
    print("Start evaluation")
    gold_tag_seqs = []
    pred_tag_seqs = []
    for sent in test_data:
        words, true_tags = zip(*sent)
        gold_tag_seqs.append(true_tags)

        ### YOUR CODE HERE
        words = [word for word, tag in sent]
        hmm_tags = hmm_viterbi(words,
                               total_tokens,
                               q_tri_counts,
                               q_bi_counts,
                               q_uni_counts,
                               e_word_tag_counts,
                               e_tag_counts,
                               0.5, 0.5)
        pred_tag_seqs.append(tuple(hmm_tags))
        ### END YOUR CODE

    return evaluate_ner(gold_tag_seqs, pred_tag_seqs)


if __name__ == "__main__":
    start_time = time.time()
    train_sents = read_conll_ner_file("data/train.conll")
    dev_sents = read_conll_ner_file("data/dev.conll")
    vocab = compute_vocab_count(train_sents)

    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts = hmm_train(train_sents)

    hmm_eval(dev_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
             e_word_tag_counts, e_tag_counts)

    train_dev_end_time = time.time()
    print("Train and dev evaluation elapsed: " + str(train_dev_end_time - start_time) + " seconds")
