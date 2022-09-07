import logging
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from model import ElectraForSequenceDisfluency_real, ElectraForSequenceDisfluency_sing
import csv, os
from transformers import ElectraConfig, BertTokenizer, AdamW
import random, time

class InputExample(object):
    '''data class'''
    def __init__(self, guid, text_a, text_b=None, label=None, disf_label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.disf_label = disf_label

class InputFeatures(object):
    """feature class"""
    def __init__(self, input_ids, input_mask, segment_ids=None, label_id=None, label_disf_id=None, label_sing_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.label_disf_id = label_disf_id
        self.label_sing_id = label_sing_id

class DataProcessor(object):
    """Base class"""
    def get_train_examples(self, data_dir):
        raise NotImplementedError()
    def get_dev_examples(self, data_dir):
        raise NotImplementedError()
    def get_labels(self):
        raise NotImplementedError()
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class DisfluencyProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""
    def get_origin_examples(self, data_dir, file_name):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, file_name)))

    def get_true_examples(self, data_dir, file_name):
        return self._create_true_examples(self._read_tsv(os.path.join(data_dir, file_name)))
    
    def get_single_from_pair_examples(self, data_dir, file_name):
        return self._create_single_examples(self._read_tsv(os.path.join(data_dir, file_name)))
    
    def get_pair_examples(self, data_dir, file_name):
        return self._create_pair_examples(self._read_tsv(os.path.join(data_dir, file_name)))
    
    def get_pair_examples_fake(self, data_dir, file_name):
        return self._create_pair_examples_fake(self._read_tsv(os.path.join(data_dir, file_name)))
    
    
    def get_false_examples(self, data_dir, file_name):
        return self._create_false_examples(self._read_tsv(os.path.join(data_dir, file_name)))

    def get_labels(self):
        """See base class."""
        return ["add_0", "add_1", "del_0", "del_1"]
    
    def get_labels_judge(self):
        """See base class."""
        return ["error_0", "error_1"]

    def get_labels_disf(self):
        """See base class."""
        return ["O", "D"]

    def get_sing_labels(self):
        """See base class."""
        return ["false", "true"]

    def _create_examples(self, lines):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]
            text_a = line[1]
            text_b = line[2]
            try:
                label = float(line[5])
            except:
                label = -1
            disf_label = line[4]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, disf_label=disf_label))
        return examples
    
    def _create_pair_examples(self, lines):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]
            text_a = line[1]
            label = line[3]
            if label == 'NONE':
                label = 'true'
            disf_label = line[4]
            if line[2] == 'NONE':
                text_b = []
                list_a = text_a.split(' ')
                tag = disf_label.split(' ')
                for k in range(len(list_a)):
                    if tag[k] == 'O':
                        text_b.append(list_a[k])
                if len(text_b)==0:
                    continue
                text_b = ' '.join(text_b)
            else:
                text_b = line[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, disf_label=disf_label))
        return examples

    def _create_pair_examples_fake(self, lines):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]
            text_a = line[1]
            label = line[3]
            if label == 'NONE':
                label = 'true'
            disf_label = line[4]
            if line[2] == 'NONE':
                text_b = text_a
            else:
                text_b = line[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, disf_label=disf_label))
        return examples
    
    def _create_single_examples(self, lines):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]
            text_a = line[2]
            text_b = 'NONE'
            label = line[3]
            disf_label = 'NONE'
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, disf_label=disf_label))
        return examples
    
    def _create_true_examples(self, lines):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]
            text_a = line[1]
            text_b = line[2]
            try:
                label = float(line[5])
            except:
                label = 'true'
            disf_label = line[4]
            
            text_a_fix = list()
            label_disf_id_fix = list()
            for i, j in zip(text_a.split(" "), disf_label.split(" ")):
                if j == "D":
                    continue
                text_a_fix.append(i)
                label_disf_id_fix.append(j)
            
            if len(label_disf_id_fix) == 0:
                continue

            examples.append(
                InputExample(guid=guid, text_a=" ".join(text_a_fix), text_b=text_b, label=label, disf_label=" ".join(label_disf_id_fix)))
        return examples

    def _create_false_examples(self, lines):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]
            text_a = line[1]
            text_b = line[2]
            try:
                label = float(line[5])
            except:
                label = 'true'
            disf_label = line[4]
            
            text_a_fix = list()
            label_disf_id_fix = list()
            for i, j in zip(text_a.split(" "), disf_label.split(" ")):
                if j == "D":
                    continue
                text_a_fix.append(i)
                label_disf_id_fix.append(j)
            
            if len(label_disf_id_fix) == 0:
                continue

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, disf_label=disf_label))
        return examples

def label_to_map(label, label_map):
    label = label.strip().split(" ")
    out_label = []
    for el in label:
        out_label.append(label_map[el])
    return out_label

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        logging.info('over seq size')
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def random_word_no_prob(text, label, label_map, tokenizer):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :param label: labels such as ["D","O","O","D"]
    :param label_map: labels such as [0,1,,0]
    :param sel_prob: the prob to caluate the loss for each token
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    text = text.strip().split(" ")
    orig_to_map_label = []
    orig_to_map_token = []
    assert len(text) == len(label_map)
    for i in range(0, len(text)):
        orig_token = text[i]
        # orig_label = label[i]
        orig_label_map = label_map[i]
        tokens = tokenizer.tokenize(orig_token)
        orig_to_map_token.extend(tokens)
        orig_to_map_label.append(orig_label_map)
        for j in range(1, len(tokens)):
            orig_to_map_label.append(-1)
    assert len(orig_to_map_label) == len(orig_to_map_token)
    return orig_to_map_token, orig_to_map_label

def random_word_no_prob_unlabel(text, tokenizer):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :param label: labels such as ["D","O","O","D"]
    :param label_map: labels such as [0,1,,0]
    :param sel_prob: the prob to caluate the loss for each token
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    text = text.strip().split(" ")
    orig_to_map_token = []
    orig_to_map_label = []
    for i in range(0, len(text)):
        orig_token = text[i]
        tokens = tokenizer.tokenize(orig_token)
        orig_to_map_token.extend(tokens)
        orig_to_map_label.append(1)
        for j in range(1, len(tokens)):
            orig_to_map_label.append(-1)
    assert len(orig_to_map_label) == len(orig_to_map_token)
    return orig_to_map_token, orig_to_map_label


def convert_examples_to_features(examples, label_list, label_list_tagging, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}
    label_tagging_map = {label: i for i, label in enumerate(label_list_tagging)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = None
        tokens_b = None
        if example.text_b != "NONE":
            tokens_a = tokenizer.tokenize(example.text_a)
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            
        else:
            label_disf_id = label_to_map(example.disf_label, label_tagging_map)
            tokens_a, disf_label = random_word_no_prob(example.text_a, example.disf_label.strip().split(" "),
                                                       label_disf_id, tokenizer)
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
                disf_label = disf_label[:(max_seq_length - 2)]


        if tokens_b:
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)
            label_id = label_map[example.label]
            disf_label_id = [-1] * len(tokens)
        else:
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            label_id = example.label
            disf_label_id = ([-1] + disf_label + [-1])

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        padding_disf = [-1] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        disf_label_id += padding_disf

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(disf_label_id) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          label_disf_id=disf_label_id))
    return features

def accuracy_tagging(eval_examples, predict_result_tagging, gold_result_tagging, input_mask_tagging, output_name):
    output_file = open(output_name, "w")
    example_id = -1
    assert len(predict_result_tagging) == len(gold_result_tagging)
    assert len(predict_result_tagging) == len(input_mask_tagging)
    gold_number = 0
    predict_number = 0
    correct_number = 0
    for i in range(0, len(predict_result_tagging)):
        predict_results = predict_result_tagging[i]
        gold_results = gold_result_tagging[i]
        input_masks = input_mask_tagging[i]
        assert len(predict_results) == len(gold_results)
        assert len(predict_results) == len(input_masks)
        for j in range(0, len(gold_results)):
            example_id += 1
            text_a = eval_examples[example_id].text_a.strip().split(" ")
            length = input_masks[j].count(1)
            # print (eval_examples[example_id].text_a)
            # print (length)
            gold_result_tmp = gold_results[j][0:length]
            predict_result_tmp = predict_results[j][0:length]
            gold_result_tmp = gold_result_tmp[1:len(gold_result_tmp) - 1]
            predict_result_tmp = predict_result_tmp[1:len(predict_result_tmp) - 1]
            assert len(gold_result_tmp) == len(predict_result_tmp)
            gold_result = []
            predict_result = []
            for k in range(0, len(gold_result_tmp)):
                if gold_result_tmp[k] != -1:
                    gold_result.append(gold_result_tmp[k])
                    predict_result.append(predict_result_tmp[k])
            assert len(text_a) == len(gold_result)

            output_tokens = []
            for l in range(0, len(text_a)):
                gold_label = "D" if gold_result[l] == 1 else "O"
                predict_label = "D" if predict_result[l] == 1 else "O"
                word = text_a[l]
                output_tokens.append(word + "#" + gold_label + "#" + predict_label)
            output_file.write(" ".join(output_tokens) + "\n")

            gold_number += gold_result.count(1)
            predict_number += predict_result.count(1)
            sum_result = list(map(lambda x: x[0] + x[1], zip(gold_result, predict_result)))
            correct_number += sum_result.count(2)
    # print (gold_result)
    #         print (predict_result)
    # print (gold_number)
    # print (predict_number)
    # print (correct_number)
    output_file.close()
    try:
        p_score = correct_number * 1.0 / predict_number
        r_score = correct_number * 1.0 / gold_number
        f_score = 2.0 * p_score * r_score / (p_score + r_score)
    except:
        p_score = 0
        r_score = 0
        f_score = 0
    return p_score, r_score, f_score

def convert_examples_to_features_unlabel(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):

        tokens_a, disf_label = random_word_no_prob_unlabel(example.text_a, tokenizer)
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]
            disf_label = disf_label[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        disf_label_id = ([-1] + disf_label + [-1])
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        padding_disf = [-1] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        disf_label_id += padding_disf

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(disf_label_id) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_disf_id=disf_label_id,
                          label_id=None))
    return features


def unlabel_tagging(eval_examples, predict_result_tagging, gold_result_tagging, input_mask_tagging, output_name):
    
    output_file = open(output_name, "w")
    example_id = -1
    assert len(predict_result_tagging) == len(input_mask_tagging)
    id = 1
    for i in range(0, len(predict_result_tagging)):
        predict_results = predict_result_tagging[i]
        gold_results = gold_result_tagging[i]
        input_masks = input_mask_tagging[i]
        assert len(predict_results) == len(input_masks)
        for j in range(0, len(predict_results)):
            example_id += 1
            text_a = eval_examples[example_id].text_a.strip().split(" ")
            length = input_masks[j].count(1)
            gold_result_tmp = gold_results[j][0:length]
            gold_result_tmp = gold_result_tmp[1:len(gold_result_tmp) - 1]
            
            output_set = []
            for topK in range(len(predict_results[j])):
                predict_result_tmp = predict_results[j][topK][0:length]
                predict_result_tmp = predict_result_tmp[1:len(predict_result_tmp) - 1]
                gold_result = []
                predict_result = []
                for k in range(0, len(gold_result_tmp)):
                    if gold_result_tmp[k] != -1:
                        gold_result.append(gold_result_tmp[k])
                        predict_result.append(predict_result_tmp[k])

                assert len(text_a) == len(predict_result)

                output_tokens = []
                for l in range(0, len(text_a)):
                    predict_label = "D" if predict_result[l] == 1 else "O"
                    output_tokens.append(predict_label)
            
                if output_tokens in output_set or not 'O' in output_tokens:
                    pass
                else:
                    output_set.append(output_tokens)
                    output_file.write(str(id)+"\t"+" ".join(text_a)+"\tNONE\tNONE\t"+" ".join(output_tokens) + "\n")
            id += 1
    output_file.close()

def random_word(text1, label, label_map, tokenizer, sel_prob):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :param label: labels such as ["D","O","O","D"]
    :param label_map: labels such as [0,1,,0]
    :param sel_prob: the prob to caluate the loss for each token
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    text = text1.strip().split(" ")
    orig_to_map_label = []
    orig_to_map_token = []
    if len(text) != len(label_map):
        print("text:{}".format(text))
        print("len text:{}".format(len(text)))
        print("label_map:{}".format(label_map))
        print("len label_map:{}".format(len(label_map)))
    assert len(text) == len(label_map)
    assert len(text) == len(label)

    for i in range(0, len(text)):
        orig_token = text[i]
        orig_label = label[i]
        orig_label_map = label_map[i]
        tokens = tokenizer.tokenize(orig_token)
        orig_to_map_token.extend(tokens)

        prob = random.random()
        if orig_label == "D":
            if prob < sel_prob:
                orig_to_map_label.append(orig_label_map)
            else:
                orig_to_map_label.append(-1)
        else:
            if prob < sel_prob / 5.0:
                orig_to_map_label.append(orig_label_map)
            else:
                orig_to_map_label.append(-1)

        for j in range(1, len(tokens)):
            orig_to_map_label.append(-1)
    
    assert len(orig_to_map_label) == len(orig_to_map_token)

    return orig_to_map_token, orig_to_map_label

def convert_examples_to_features_judge(examples, label_list, label_list_tagging, label_sing_list, max_seq_length, tokenizer, train_type='eval'):

    label_map = {label: i for i, label in enumerate(label_list)}
    label_tagging_map = {label: i for i, label in enumerate(label_list_tagging)}
    label_sing_map = {label : i for i, label in enumerate(label_sing_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = None
        tokens_b = None
        sing_label = None
        disf_label = None
        if example.text_b != "NONE":
            tokens_a = tokenizer.tokenize(example.text_a)
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            # _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        elif example.text_b == "NONE" and example.disf_label == "NONE":
            tokens_a = tokenizer.tokenize(example.text_a)
            sing_label = example.label
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
        else:
            # label_disf_id = label_tagging_map[example.disf_label]
            try:
                label_disf_id = label_to_map(example.disf_label, label_tagging_map)
            except:
                exit(0)
            if train_type == "train":
                tokens_a, disf_label = random_word(example.text_a, example.disf_label.strip().split(" "),
                                                   label_disf_id, tokenizer, 0.5)
            else:

                tokens_a, disf_label = random_word_no_prob(example.text_a, example.disf_label.strip().split(" "),
                                                           label_disf_id, tokenizer)
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
                disf_label = disf_label[:(max_seq_length - 2)]

        if tokens_b:
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)
            label_id = label_map[example.label]
            disf_label_id = [-1] * len(tokens)
            sing_label_id = -1
        elif example.text_b == "NONE" and example.disf_label == "NONE":
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            label_id = -1
            disf_label_id = [-1] * len(tokens)
            sing_label_id = label_sing_map[example.label]
        else:
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            label_id = -1
            disf_label_id = ([-1] + disf_label + [-1])
            sing_label_id = label_sing_map[example.label]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        padding_disf = [-1] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        disf_label_id += padding_disf

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(disf_label_id) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          label_disf_id=disf_label_id,
                          label_sing_id=sing_label_id))
    return features