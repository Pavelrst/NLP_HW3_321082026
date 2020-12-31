import os
from data import *
from collections import defaultdict

def most_frequent_train(train_data):
    """
    Gets training data that includes tagged sentences.
    Returns a dictionary that maps every word in the training set to its most frequent tag.
    The dictionary should have a default value.
    """
    ### YOUR CODE HERE
    all_possible_tags = []
    tag_count_dict = {}
    for sentence in train_data:
        for word, _ in sentence:
            tag_count_dict[word] = []

    for sentence in train_data:
        for word, tag in sentence:
            if not tag in all_possible_tags:
                all_possible_tags.append(tag)
            tag_count_dict[word].append(tag)

    print(all_possible_tags)

    def most_frequent(List):
        return max(set(List), key=List.count)

    for key in tag_count_dict.keys():
        most_common_tag = most_frequent(tag_count_dict[key])
        tag_count_dict[key] = most_common_tag

    return tag_count_dict
    ### END YOUR CODE

def most_frequent_eval(test_set, pred_tags):
    """
    Gets test data and tag prediction map.
    Returns an evaluation of the accuracy of the most frequent tagger.
    """
    gold_tag_seqs = []
    pred_tag_seqs = []
    for sent in test_set:
        words, true_tags = zip(*sent)
        gold_tag_seqs.append(true_tags)

        ### YOUR CODE HERE
        temp_list = []
        for word in words:
            temp_list.append(pred_tags[word])
        pred_tag_seqs.append(tuple(temp_list))
        ### END YOUR CODE

    return evaluate_ner(gold_tag_seqs, pred_tag_seqs)

if __name__ == "__main__":
    train_sents = read_conll_ner_file("data/train.conll")
    dev_sents = read_conll_ner_file("data/dev.conll")
    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    model = most_frequent_train(train_sents)
    most_frequent_eval(dev_sents, model)

