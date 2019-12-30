import json
from models import *
import os
import pandas
from transformers import BertTokenizer
import torch
import torch.optim as optim
import scipy.stats as ss
import argparse
from itertools import chain
import sys
import random
import gc
from transformers import AdamW, WarmupLinearSchedule

device = torch.device('cuda')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', default=768, type=int, help="whether to train or test the model")
    parser.add_argument('--head', default=4, type=int, help="whether to train or test the model")
    parser.add_argument('--epochs', default=20, type=int, help="whether to train or test the model")
    parser.add_argument('--split', default=256, type=int, help="whether to train or test the model")
    parser.add_argument('--max_len', default=30, type=int, help="whether to train or test the model")
    parser.add_argument('--do_train', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_test', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--lr_default', type=float, default=5e-5, help="whether to train or test the model")
    parser.add_argument('--load_from', default='', type=str, help="whether to train or test the model")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_opt()

    model_type = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_type)

    model = TableEncoder(args.dim, args.head, model_type)
    model.to(device)

    def detect_number(string):
        string = str(string)
        if '-' not in string:
            for _ in string.split(' '):
                try:
                    __ = float(_)
                    return __
                except:
                    return string
        else:
            return string

    def paralell_table(table):
        # Parallel text encoding
        texts = []
        for i in range(len(table)):
            entry = table.iloc[i]
            for j, col in enumerate(cols):
                text = 'in row {}, {} is {} .'.format(i + 1, col, entry[col])
                tmp = tokenizer.encode(text)
                tmp = [tokenizer.cls_token_id] + tmp
                if len(tmp) > args.max_len:
                    tmp = tmp[:args.max_len]
                texts.append(tmp)
        max_len = max([len(_) for _ in texts])
        for i in range(len(texts)):
            texts[i] = texts[i] + [tokenizer.pad_token_id] * (max_len - len(texts[i]))
        return texts

    def distribute_encode(texts):
        # Using BERT to encode the cells
        inp = torch.tensor(texts).to(device)
        encoded = []
        splits = len(inp) // args.split + 1
        for i in range(splits):
            if i * args.split < len(inp):
                encoded.append(model('cell', inp[i * args.split: (i + 1) * args.split])[0][:, 0])
        encoded = torch.cat(encoded, 0)
        representation = encoded.view(len(table), len(cols), -1)
        return representation

    def filter_num_cols(table, cols):
        outputs = []
        idxs = []
        col_mask = torch.FloatTensor(len(cols)).zero_()

        for i, col in enumerate(cols):
            tmp = []
            for _ in table[col]:
                tmp.append(detect_number(_))

            if len([_ for _ in tmp if isinstance(_, float)]) >= len(tmp) - 1:
                for j in range(len(tmp)):
                    if not isinstance(tmp[j], float):
                        tmp[j] = 0.
                outputs.append(ss.rankdata(tmp))
                idxs.append(i)
                col_mask[i] = 1
        return outputs, idxs, col_mask

    if args.do_train:
        with open('data/train_examples.json') as f:
            examples = json.load(f)

        optimizer = AdamW(model.parameters(), lr=args.lr_default, eps=1e-8)
        t_total = len(examples) * args.epochs
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=1000, t_total=t_total)
        cross_entropy = torch.nn.CrossEntropyLoss()

        files = list(examples.keys())

        for epoch_ in range(args.epochs):
            random.shuffle(files)
            for f_idx, f in enumerate(files):
                table = pandas.read_csv('all_csv/{}'.format(f), '#')
                cols = table.columns

                model.zero_grad()
                optimizer.zero_grad()

                texts = paralell_table(table)
                representation = distribute_encode(texts)

                # Using row-wise attention
                representation = model('row', representation, None)
                outputs, idxs, col_mask = filter_num_cols(table, cols)

                """
                if len(idxs) > 0:
                    number_rank = torch.FloatTensor(outputs)

                    graph_mask = number_rank.unsqueeze(1).expand(-1, number_rank.size(-1), -1) \
                        < number_rank.unsqueeze(-1).expand(-1, -1, number_rank.size(-1))
                    graph = graph_mask.long().to(device)

                    d_nodes = representation[:, idxs].transpose(1, 0)

                    output_nodes = model('col', d_nodes, graph)

                    scattered_d_nodes = torch.zeros_like(representation)

                    scattered_d_nodes[:, idxs] = output_nodes.transpose(1, 0)

                    col_mask = col_mask.to(device)

                    representation = representation * (1 - col_mask[None, :, None]) + scattered_d_nodes
                """
                statements = examples[f][0]
                sub_cols = examples[f][1]
                labels = examples[f][2]
                title = examples[f][3]

                inps = []
                for stat in statements:
                    inps.append(tokenizer.encode('[CLS] ' + title + ' [SEP] ' + stat))

                max_len = max([len(_) for _ in inps])

                for i in range(len(inps)):
                    inps[i] = inps[i] + [tokenizer.pad_token_id] * (max_len - len(inps[i]))

                inps = torch.LongTensor(inps).to(device)
                labels = torch.LongTensor(labels).to(device)

                stat_representation = model('cell', inps)[0][:, 0]

                # representation = representation.view(-1, args.dim)  # .repeat(stat_representation.shape[0], 1, 1)
                max_col = max([len(_) for _ in sub_cols])

                scattered_representation = torch.FloatTensor(
                    len(statements), len(table), max_col, representation.shape[-1]).zero_().to(device)

                for i in range(len(statements)):
                    scattered_representation[i][:, :len(sub_cols[i])] = representation[:, sub_cols[i]]

                representation = scattered_representation.view(len(statements), -1, representation.shape[-1])

                logits = model('fusion', stat_representation.unsqueeze(1), representation, None)

                loss = cross_entropy(logits.view(-1, 2), labels)
                preds = torch.argmax(logits, -1)

                if (f_idx + 1) % 50 == 0:
                    correct_or_not = (preds.view(-1) == labels.view(-1))
                    acc = correct_or_not.sum().item() / correct_or_not.size(0)

                    # Compute the accuracy and labels
                    preds = preds.squeeze().cpu().data.numpy()
                    print("{}/{} loss function = {} predictions: {} accuracy: {}".
                          format(f_idx, len(files), loss.item(), preds, acc))

                loss.backward()

                optimizer.step()
                scheduler.step()

            torch.save(model.state_dict(), 'models/model_ep{}.pt'.format(epoch_))

    if args.do_test:
        model.load_state_dict(torch.load(args.load_from))
        model.eval()

        with open('data/test_examples.json') as f:
            examples = json.load(f)
        files = list(examples.keys())

        with torch.no_grad():
            correct, total = 0, 0
            for f_idx, f in enumerate(files):
                table = pandas.read_csv('all_csv/{}'.format(f), '#')
                cols = table.columns

                texts = paralell_table(table)
                representation = distribute_encode(texts)

                # Using row-wise attention
                representation = model('row', representation, None)
                outputs, idxs, col_mask = filter_num_cols(table, cols)

                if len(idxs) > 0:
                    number_rank = torch.FloatTensor(outputs)

                    graph_mask = number_rank.unsqueeze(1).expand(-1, number_rank.size(-1), -1) \
                        < number_rank.unsqueeze(-1).expand(-1, -1, number_rank.size(-1))
                    graph = graph_mask.long().to(device)

                    d_nodes = representation[:, idxs].transpose(1, 0)

                    output_nodes = model('col', d_nodes, graph)

                    scattered_d_nodes = torch.zeros_like(representation)

                    scattered_d_nodes[:, idxs] = output_nodes.transpose(1, 0)

                    col_mask = col_mask.to(device)

                    representation = representation * (1 - col_mask[None, :, None]) + scattered_d_nodes

                statements = examples[f][0]
                title = examples[f][2]
                inps = []
                for stat in statements:
                    inps.append(tokenizer.encode('[CLS] ' + title + ' [SEP] ' + stat))

                max_len = max([len(_) for _ in inps])

                for i in range(len(inps)):
                    inps[i] = inps[i] + [tokenizer.pad_token_id] * (max_len - len(inps[i]))

                inps = torch.LongTensor(inps).to(device)
                labels = torch.LongTensor(examples[f][1]).to(device)

                stat_representation = model('cell', inps)[0][:, 0]
                representation = representation.view(-1, args.dim).repeat(stat_representation.shape[0], 1, 1)

                logits = model('fusion', stat_representation.unsqueeze(1), representation, None)

                preds = torch.argmax(logits, -1)

                correct_or_not = (preds.view(-1) == labels.view(-1)).float()

                correct += correct_or_not.sum().item()
                total += correct_or_not.size(0)

                sys.stdout.write("prediction accuracy: {} \r".format(correct / total))
