from data import *
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
import time
import os
import numpy as np
from collections import defaultdict
import pickle

MODEL_FN = 'models/memm.pickle'

def build_extra_decoding_arguments(train_sents):
    """
    Receives: all sentences from training set
    Returns: all extra arguments which your decoding procedures requires
    """
    extra_decoding_arguments = {}
    ### YOUR CODE HERE
    # raise NotImplementedError
    ### END YOUR CODE

    return extra_decoding_arguments


def extract_features_base(curr_word, next_word, nextnext_word, prev_word, prevprev_word, prev_tag, prevprev_tag):
    """
        Receives: a word's local information
        Returns: The word's features.
    """
    features = {
        'word': curr_word,
        'prev_word': prev_word,
        'prev_word_bigram': prevprev_word + '__' + prev_word,
        'next_word': next_word,
        'nextnext_word': nextnext_word,
        'prev_tag': prev_tag,
        'prev_tag_bigram': prevprev_tag + '__' + prev_tag
    }
    # import pdb; pdb.set_trace()
    return features

def extract_features(sentence, i):
    curr_word = sentence[i][0]
    prev_token = sentence[i - 1] if i > 0 else ('<st>', '*')
    prevprev_token = sentence[i - 2] if i > 1 else ('<st>', '*')
    next_token = sentence[i + 1] if i < (len(sentence) - 1) else ('</s>', 'STOP')
    nextnext_token = sentence[i + 2] if i < (len(sentence)) - 2 else ('</s>', 'STOP')
    return extract_features_base(curr_word, next_token[0], nextnext_token[0], prev_token[0], prevprev_token[0], prev_token[1], prevprev_token[1])

def vectorize_features(vec, features):
    """
        Receives: feature dictionary
        Returns: feature vector

        Note: use this function only if you chose to use the sklearn solver!
        This function prepares the feature vector for the sklearn solver,
        use it for tags prediction.
    """
    example = [features]
    return vec.transform(example)

def create_examples(sents, tag_to_idx_dict):
    examples = []
    labels = []
    num_of_sents = 0
    for sent in sents:
        num_of_sents += 1
        for i in range(len(sent)):
            features = extract_features(sent, i)
            examples.append(features)
            labels.append(tag_to_idx_dict[sent[i][1]])

    return examples, labels


def memm_greedy(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments):
    """
        Receives: a sentence to tag and the parameters learned by memm
        Returns: predicted tags for the sentence
    """

    decoded_sent = [(word, '') for word, pos in sent]
    # ^ we'll add NER tags to the decoded sentence as they are decoded

    for i in range(len(decoded_sent)):
        features = extract_features(decoded_sent, i)
        feature_matrix = vec.transform(features)
        # ^ 1 X N_FEATURES sparse matrix
        
        pred = logreg.predict(feature_matrix)[0]
        pred_tag = index_to_tag_dict[pred]
        
        # set NER tag in decoded sentence (will be used in subsequent steps)
        word, _ = decoded_sent[i]
        decoded_sent[i] = (word, pred_tag)

        # import pdb; pdb.set_trace()

    return [tag for _, tag in decoded_sent]

def memm_viterbi(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments):
    """
        Receives: a sentence to tag and the parameters learned by memm
        Returns: predicted tags for the sentence
    """
    
    # decoded_sent = [(word, '') for word, tag in sent]
    
    sent_no_tags = [(word, '') for word, tag in sent]

    N_TAGS = max(index_to_tag_dict)
    assert index_to_tag_dict[N_TAGS] == '*'
    # maximum index is dummy tag, which we won't use

    best_paths = [(0, sent_no_tags) for index in range(N_TAGS)]

    for i in range(len(sent)):

        features = [
            extract_features(decoded_sent, i)
            for _, decoded_sent in best_paths
        ]
        feature_matrix = vec.transform(features)
        # ^ N_TAGS X N_FEATURES sparse matrix

        probs = logreg.predict_proba(feature_matrix)
        logits = np.log(probs)
        # ^ N_TAGS X N_TAGS matrix of logits
        # first dimension: index in best_paths
        # second dimension: new tag added

        existing_scores = np.array([score for score, _ in best_paths])
        new_scores = existing_scores[:, None] + logits
        # note: new_scores[i, j] = existing_scores[i] + logits[i, j]

        indices_to_extend = np.argmax(new_scores, axis=0)
        # ^ for each new possible tag, gives index of partially decoded sentence to use

        # update best_paths with new predictions"
        new_best_paths = []
        for tag_index in range(N_TAGS):
            tag = index_to_tag_dict[tag_index]
            index_to_extend = indices_to_extend[tag_index]
            
            old_score, old_sent = best_paths[index_to_extend]
            new_score = old_score + logits[index_to_extend, tag_index]
            new_sent = old_sent.copy()
            new_sent[i] = (old_sent[i][0], tag)

            new_best_paths.append((new_score, new_sent))
        
        best_paths = new_best_paths
    
    # pick highest-scoring tagged sentence
    decoded_sent = sorted(best_paths)[-1][1]
    # ^ by default sorts by first element of tuples (score)

    return [tag for _, tag in decoded_sent]

def memm_eval(test_data, logreg, vec, index_to_tag_dict, extra_decoding_arguments):
    """
    Receives: test data set and the parameters learned by memm
    Returns an evaluation of the accuracy of Viterbi & greedy memm
    """
    acc_viterbi, acc_greedy = 0.0, 0.0
    eval_start_timer = time.time()

    correct_greedy_preds = 0
    correct_viterbi_preds = 0
    total_words_count = 0

    gold_tag_seqs = []
    greedy_pred_tag_seqs = []
    viterbi_pred_tag_seqs = []

    for i, sent in enumerate(test_data):

        if i % 100 == 0:
            print(i, '/', len(test_data), 'sentences evaluated')

        words, true_tags = zip(*sent)
        gold_tag_seqs.append(true_tags)

        pred_tags_greedy = memm_greedy(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments)
        pred_tags_viterbi = memm_viterbi(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments)

        greedy_pred_tag_seqs.append(pred_tags_greedy)
        viterbi_pred_tag_seqs.append(pred_tags_viterbi)
        correct_greedy_preds += sum(tag1 == tag2 for tag1, tag2 in zip(true_tags, pred_tags_greedy))
        correct_viterbi_preds += sum(tag1 == tag2 for tag1, tag2 in zip(true_tags, pred_tags_viterbi))
        total_words_count += len(words)

    acc_greedy = correct_greedy_preds / total_words_count
    acc_viterbi = correct_viterbi_preds / total_words_count

    print('Greedy accuracy:', acc_greedy)
    print('Viterbi accuracy:', acc_viterbi)

    greedy_evaluation = evaluate_ner(gold_tag_seqs, greedy_pred_tag_seqs)
    viterbi_evaluation = evaluate_ner(gold_tag_seqs, viterbi_pred_tag_seqs)

    return greedy_evaluation, viterbi_evaluation

def build_tag_to_idx_dict(train_sentences):
    curr_tag_index = 0
    tag_to_idx_dict = {}
    for train_sent in train_sentences:
        for token in train_sent:
            tag = token[1]
            if tag not in tag_to_idx_dict:
                tag_to_idx_dict[tag] = curr_tag_index
                curr_tag_index += 1

    tag_to_idx_dict['*'] = curr_tag_index
    return tag_to_idx_dict


if __name__ == "__main__":
    full_flow_start = time.time()
    train_sents = read_conll_ner_file("data/train.conll")
    dev_sents = read_conll_ner_file("data/dev.conll")

    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    extra_decoding_arguments = build_extra_decoding_arguments(train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)
    tag_to_idx_dict = build_tag_to_idx_dict(train_sents)
    index_to_tag_dict = invert_dict(tag_to_idx_dict)

    vec = DictVectorizer()
    print("Create train examples")
    train_examples, train_labels = create_examples(train_sents, tag_to_idx_dict)


    num_train_examples = len(train_examples)
    print("#example: " + str(num_train_examples))
    print("Done")

    print("Create dev examples")
    dev_examples, dev_labels = create_examples(dev_sents, tag_to_idx_dict)
    num_dev_examples = len(dev_examples)
    print("#example: " + str(num_dev_examples))
    print("Done")

    all_examples = train_examples
    all_examples.extend(dev_examples)

    print("Vectorize examples")
    all_examples_vectorized = vec.fit_transform(all_examples)
    train_examples_vectorized = all_examples_vectorized[:num_train_examples]
    dev_examples_vectorized = all_examples_vectorized[num_train_examples:]
    print("Done")

    if os.path.isfile(MODEL_FN):
        print('Model file exists, loading:', MODEL_FN)
        with open(MODEL_FN, 'rb') as f:
            logreg = pickle.load(f)
        print('Model loaded')
    else:
        logreg = linear_model.LogisticRegression(
            multi_class='multinomial', max_iter=128, solver='lbfgs', C=100000, verbose=1)
        print("Fitting...")
        start = time.time()
        logreg.fit(train_examples_vectorized, train_labels)
        end = time.time()
        print("End training, elapsed " + str(end - start) + " seconds")

        print('Saving trained model to file:', MODEL_FN)
        with open(MODEL_FN, 'wb') as f:
            pickle.dump(logreg, f)
        print('Model saved')
    # End of log linear model training

    # Evaluation code - do not make any changes
    start = time.time()
    print("Start evaluation on dev set")

    memm_eval(dev_sents, logreg, vec, index_to_tag_dict, extra_decoding_arguments)
    
    end = time.time()

    print("Evaluation on dev set elapsed: " + str(end - start) + " seconds")
