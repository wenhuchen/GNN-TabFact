# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from collections import OrderedDict
import argparse
import csv
import logging
import os
import random
import sys
import io
import json

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from models import *
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertConfig, BertTokenizer, XLNetConfig, XLNetTokenizer
from tensorboardX import SummaryWriter
from pprint import pprint
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """
    A single set of features of data.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask=None, token_type_ids=None, label_ids=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_ids = label_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            idx = 0
            for line in reader:
                idx += 1
                # if idx > 100: break
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train_baseline_examples.json")), "train")
        # return self._create_examples(self._read_tsv(os.path.join(data_dir, 'train.tsv')), 'train')

    def get_dev_examples(self, data_dir, dataset="val"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "{}_baseline_examples.json".format(dataset))), dataset)
        # return self._create_examples(self._read_tsv(os.path.join(data_dir, dataset + '.tsv')), dataset)

    def get_labels(self):
        """See base class."""
        # return ['0', '1']
        return [0, 1]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            text_a = line[2] + ' . ' + line[1] + ' .'
            text_b = line[3]
            label = line[4]
            examples.append(InputExample(guid=i, text_a=text_a, text_b=text_b, label=label))
        return examples
        """
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                # column_types = [int(x) for x in line[2].split()]
                #column_types = line[2].split()
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
        """


def convert_examples_to_features(
    examples,
    label_list,
    max_length,
    tokenizer,
    output_mode=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    pos_buf = []
    neg_buf = []
    logger.info("convert_examples_to_features ...")

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))

        inputs = tokenizer.encode_plus(example.text_b, example.text_a, add_special_tokens=True, max_length=max_length,)

        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )

        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label_ids=label
            )
        )

    return features


"""
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode, fact_place = None, balance = False, verbose = False):
    assert fact_place is not None
    label_map={label: i for i, label in enumerate(label_list)}
    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
    features=[]
    pos_buf=[]
    neg_buf=[]
    logger.info("convert_examples_to_features ...")
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a=tokenizer.tokenize(example.text_a)

        tokens_b=None
        if example.text_b:
            tokens_b=tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a=tokens_a[:(max_seq_length - 2)]

        # NOTE: fact is tokens_b and is now in front
        if fact_place == "first":
            tokens=["[CLS]"] + tokens_b + ["[SEP]"]
            segment_ids=[0] * (len(tokens_b) + 2)

            assert len(tokens) == len(segment_ids)

            tokens += tokens_a + ["[SEP]"]
            segment_ids += [1] * (len(tokens_a) + 1)
        else:
            tokens=["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids=[0] * (len(tokens_a) + 2)

            assert len(tokens) == len(segment_ids)

            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids=tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask=[1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding=[0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id=label_map[example.label]
        elif output_mode == "regression":
            label_id=float(example.label)
        else:
            raise KeyError(output_mode)

        if verbose and ex_index < 5:
            logger.info("*** Example ***")
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        if balance:
            if label_id == 1:
                pos_buf.append(InputFeatures(input_ids=input_ids,
                                             input_mask=input_mask,
                                             segment_ids=segment_ids,
                                             label_id=label_id))
            else:
                neg_buf.append(InputFeatures(input_ids=input_ids,
                                             input_mask=input_mask,
                                             segment_ids=segment_ids,
                                             label_id=label_id))

            if len(pos_buf) > 0 and len(neg_buf) > 0:
                features.append(pos_buf.pop(0))
                features.append(neg_buf.pop(0))
        else:
            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))

    return features
"""


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    return {"acc": acc}


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "qqp":
        return acc_and_f1(preds, labels)
    else:
        raise KeyError(task_name)


def main():
    logger.info("Running %s" % ' '.join(sys.argv))

    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--scan",
                        default="horizontal",
                        choices=["vertical", "horizontal"],
                        type=str,
                        help="The direction of linearizing table cells.")
    parser.add_argument("--data_dir",
                        default="data/",
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir",
                        default="models/",
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_dir",
                        type=str,
                        help="The output directory where the model checkpoints will be loaded during evaluation")
    parser.add_argument('--load_step',
                        type=int,
                        default=0,
                        help="The checkpoint step to be loaded")
    parser.add_argument('--dim', default=768, type=int, help="whether to train or test the model")
    parser.add_argument('--head', default=4, type=int, help="whether to train or test the model")
    parser.add_argument("--test_set",
                        default="val",
                        choices=["val", "test", "simple_test", "complex_test", "small_test"],
                        help="Which test set is used for evaluation",
                        type=str)
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--balance",
                        action='store_true',
                        help="balance between + and - samples for training.")
    # Other parameters
    """
    parser.add_argument("--bert_model",
                        default="bert-base-uncased",
                        type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    """
    parser.add_argument("--model",
                        default="xlnet-base-cased",
                        type=str,
                        help="Bert pre-trained model selected in the list: xlnet-base-cased, xlnet-large-cased")
    parser.add_argument("--task_name",
                        default="QQP",
                        type=str,
                        help="The name of the task to train.")
    parser.add_argument('--period',
                        type=int,
                        default=500)
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--fact",
                        default="first",
                        choices=["first", "second"],
                        type=str,
                        help="Whether to put fact in front.")
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=20.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()
    pprint(vars(args))
    sys.stdout.flush()

    processors = {
        "qqp": QqpProcessor,
    }

    output_modes = {
        "qqp": "classification",
    }

    device = torch.device("cuda")

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    logger.info("Datasets are loaded from {}\n Outputs will be saved to {}".format(args.data_dir, args.output_dir))

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #    raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    args.output_dir = os.path.join(args.output_dir, args.model)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    writer = SummaryWriter(os.path.join(args.output_dir, 'events'))

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]

    label_list = processor.get_labels()
    num_labels = len(label_list)

    if 'xlnet' in args.model:
        tokenizer = XLNetTokenizer.from_pretrained(args.model, do_lower_case=True)
    else:
        tokenizer = BertTokenizer.from_pretrained(args.model, do_lower_case=True)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

    model = Baseline(args.dim, args.head, args.model, num_labels)
    model.to(device)

    # Prepare optimizer
    if args.do_train:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8, correct_bias=False)
        # scheduler = get_linear_schedule_with_warmup(
        #    optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
        warm_up_steps = num_train_optimization_steps * 0.1
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warm_up_steps, num_training_steps=num_train_optimization_steps
        )

    global_step = 0
    tr_loss = 0
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode,
            pad_on_left=bool('xlnet' in args.model), pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if 'xlnet' in args.model else 0)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.attention_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.token_type_ids for f in train_features], dtype=torch.long)
        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_ids for f in train_features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_ids for f in train_features], dtype=torch.float)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        train_sampler = RandomSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            logger.info("Training epoch {} ...".format(epoch))
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, token_type_ids, label_ids = batch

                inputs = {'input_ids': input_ids, 'attention_mask': input_mask, 'token_type_ids': token_type_ids}
                logits = model('cell', **inputs)

                if (step + 1) % 50 == 0:
                    preds = torch.argmax(logits, -1).cpu().data.numpy()
                    #print("pred: {}; groundtruth: {}".format(preds, label_ids.cpu().data.numpy()))

                if output_mode == "classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                elif output_mode == "regression":
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), label_ids.view(-1))

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                writer.add_scalar('train/loss', loss, global_step)
                tr_loss += loss.item()

                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    total_norm = 0.0
                    for n, p in model.named_parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2

                    total_norm = total_norm ** (1. / 2)
                    preds = torch.argmax(logits, -1) == label_ids
                    acc = torch.sum(preds).float() / preds.size(0)
                    writer.add_scalar('train/gradient_norm', total_norm, global_step)

                    optimizer.step()
                    scheduler.step()

                    # optimizer.zero_grad()
                    model.zero_grad()
                    global_step += 1

                if (step + 1) % args.period == 0:
                    # If we save using the predefined names, we can load using `from_pretrained`
                    torch.save(model.state_dict(), '{}/model.pt'.format(args.output_dir))

                    model.eval()
                    torch.set_grad_enabled(False)  # turn off gradient tracking
                    evaluate(args, model, device, processor, label_list, num_labels, tokenizer, output_mode, tr_loss,
                             global_step, task_name, tbwriter=writer, save_dir=args.output_dir)
                    model.train()  # turn on train mode
                    torch.set_grad_enabled(True)  # start gradient tracking
                    tr_loss = 0

    if args.do_eval:
        if args.load_dir:
            model.load_state_dict(torch.load('{}/model.pt'.format(args.output_dir)))
            print("loading model from the given folder")

        model.eval()
        torch.set_grad_enabled(False)  # turn off gradient tracking
        evaluate(args, model, device, processor, label_list, num_labels, tokenizer, output_mode, tr_loss,
                 global_step, task_name, tbwriter=writer, save_dir=args.output_dir)
        logger.info("Finished evaluating the model")


def evaluate(args, model, device, processor, label_list, num_labels, tokenizer, output_mode, tr_loss, global_step,
             task_name, tbwriter=None, save_dir=None, load_step=0):

    if args.do_eval:
        eval_examples = processor.get_dev_examples(args.data_dir, dataset=args.test_set)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, output_mode,
            pad_on_left=bool('xlnet' in args.model), pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if 'xlnet' in args.model else 0)

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.attention_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.token_type_ids for f in eval_features], dtype=torch.long)
        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_ids for f in eval_features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_ids for f in eval_features], dtype=torch.float)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        batch_idx = 0
        eval_loss = 0
        nb_eval_steps = 0
        preds = []
        temp = []

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            token_type_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            inputs = {'input_ids': input_ids, 'attention_mask': input_mask, 'token_type_ids': token_type_ids}

            with torch.no_grad():
                logits = model('cell', **inputs)

            # create eval loss and other metric required by the task
            if output_mode == "classification":
                loss_fct = CrossEntropyLoss()
                tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            elif output_mode == "regression":
                loss_fct = MSELoss()
                tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)

            labels = label_ids.detach().cpu().numpy().tolist()
            batch_idx += 1

        eval_loss = eval_loss / nb_eval_steps
        preds = preds[0]
        if output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(preds)

        result = compute_metrics(task_name, preds, all_label_ids.numpy())
        loss = tr_loss / args.period if args.do_train and global_step > 0 else None

        log_step = global_step if args.do_train and global_step > 0 else load_step
        result['eval_loss'] = eval_loss
        result['global_step'] = log_step
        result['loss'] = loss

        output_eval_metrics = os.path.join(save_dir, "eval_metrics.txt")
        with open(output_eval_metrics, "a") as writer:
            logger.info("***** Eval results {}*****".format(args.test_set))
            writer.write("***** Eval results {}*****\n".format(args.test_set))
            for key in sorted(result.keys()):
                if result[key] is not None and tbwriter is not None:
                    tbwriter.add_scalar('{}/{}'.format(args.test_set, key), result[key], log_step)
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    main()
