import os
import time
from data import *
from collections import defaultdict, Counter
import math

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

def e(word, tag, e_tag_counts, e_word_tag_counts):
    if (word, tag) in e_word_tag_counts.keys():
        return e_word_tag_counts[(word, tag)] / float(e_tag_counts[tag])
    else:
        return 0

def q_proba(tag1, tag2, tag3, q_tri_counts, q_bi_counts, q_uni_counts, total_tokens, lambda1, lambda2, lambda3):
    proba = unigram_prob(tag1, q_uni_counts, total_tokens, lambda1)
    proba += bigram_prob(tag1, tag2, q_bi_counts, q_uni_counts, lambda2)
    proba += trigram_prob(tag1, tag2, tag3, q_tri_counts, q_bi_counts, lambda3)
    return proba

def trigram_prob(tag1, tag2, tag3, q_tri_counts, q_bi_counts, factor):
    assert 0 <= factor < 1
    if (tag1, tag2, tag3) in q_tri_counts.keys():
        return factor * q_tri_counts[(tag1, tag2, tag3)]/q_bi_counts[(tag1, tag2)]
    else:
        return 0

def bigram_prob(tag1, tag2, q_bi_counts, q_uni_counts, factor):
    assert 0 <= factor < 1
    if (tag1, tag2) in q_bi_counts.keys():
        return factor * q_bi_counts[(tag1, tag2)]/q_uni_counts[tag1]
    else:
        return 0

def unigram_prob(tag1, q_uni_counts, total_tokens, factor):
    assert 0 <= factor < 1
    if tag1 in q_uni_counts.keys():
        return factor * q_uni_counts[tag1]/total_tokens
    else:
        return 0

# class Layer():
#     def __init__(self):
#         self.layer_dict = defaultdict(lambda: -math.inf)
#
#     def set_proba(self, tag1, tag2, log_proba):
#         self.layer_dict[(tag1, tag2)] = log_proba
#
#     def get_proba(self, tag1, tag2):
#         return self.layer_dict[(tag1, tag2)]
#
# class ViterbiGraph():
#     def __init__(self, lambda1, lambda2, lambda3):

def hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                e_word_tag_counts, e_tag_counts, lambda1, lambda2):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Returns: predicted tags for the sentence
    """
    predicted_tags = ["O"] * (len(sent))
    ### YOUR CODE HERE

    lambda3 = 1 - lambda1 - lambda2
    all_tags = e_tag_counts.keys()
    layer = {('START', 'START'): 0}
    back_pointers = []
    sizes = []
    for word in sent:
        new_layer = defaultdict(lambda: -math.inf)
        current_back_pointers = {}

        # Iterate over all possible tags
        for v in all_tags:
            if e(word, v, e_tag_counts, e_word_tag_counts) == 0:
                # move to the next tag
                continue
            for (w, u), prev_log_prob in layer.items():
                log_prob_wuv = (prev_log_prob +
                                math.log(q_proba(w, u, v, q_tri_counts, q_bi_counts, q_uni_counts, total_tokens, lambda1, lambda2, lambda3))
                                + math.log(e(word, v, e_tag_counts, e_word_tag_counts)))
                if log_prob_wuv > new_layer[(u, v)]:
                    new_layer[(u, v)] = log_prob_wuv
                    current_back_pointers[(u, v)] = w
        back_pointers.append(current_back_pointers)
        layer = {key: value for key, value in new_layer.items() if value > -math.inf}
        sizes.append(len(layer))

    f = lambda u_v_prob: u_v_prob[1] + math.log(q_proba(u_v_prob[0][0], u_v_prob[0][1], 'STOP', q_tri_counts, q_bi_counts, q_uni_counts, total_tokens, lambda1, lambda2, lambda3))
    two_final_tags = max(layer.items(), key=f)[0]
    predicted_tags = list(two_final_tags)
    for current_back_pointers in back_pointers[::-1][:-2]:
        w = current_back_pointers[tuple(predicted_tags[:2])]
        predicted_tags.insert(0, w)
    if predicted_tags[0] == 'START':
        predicted_tags = predicted_tags[1:]

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
        hmm_tags = hmm_viterbi(words, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,
                                   e_tag_counts,
                                   0.6, 0.3)
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
