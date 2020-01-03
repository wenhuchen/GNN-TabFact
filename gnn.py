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
from transformers import AdamW, get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

device = torch.device('cuda')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', default=768, type=int, help="whether to train or test the model")
    parser.add_argument('--head', default=4, type=int, help="whether to train or test the model")
    parser.add_argument('--epochs', default=5, type=int, help="whether to train or test the model")
    parser.add_argument('--split', default=256, type=int, help="whether to train or test the model")
    parser.add_argument('--max_len', default=30, type=int, help="whether to train or test the model")
    parser.add_argument('--do_train', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_test', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--simple', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--complex', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--fp16', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--lr_default', type=float, default=2e-5, help="whether to train or test the model")
    parser.add_argument('--load_from', default='', type=str, help="whether to train or test the model")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--batch_size", default=16, type=int, help="Max gradient norm.")
    parser.add_argument('--model', default='bert-base-multilingual-uncased', type=str, help='model to use')
    parser.add_argument('--output_dir', default='models/baseline', type=str, help='model to use')
    parser.add_argument('--encoding', default='concat', type=str,
                        help='the type of table encoder; choose from concat|row|cell')
    parser.add_argument('--max_length', default=512, type=int, help='model to use')
    parser.add_argument('--max_batch_size', default=12, type=int, help='model to use')
    parser.add_argument('--id', default=1, type=int, help='model to use')
    parser.add_argument('--attention', default='cross', type=str,
                        help='the attention used for interaction between statement and table')
    args = parser.parse_args()

    return args


def forward_pass(table, example, model):
    cols = table.columns

    statements = example[0]
    sub_cols = example[1]
    labels = example[2]
    title = example[3]

    idxs = list(range(0, len(statements)))
    random.shuffle(idxs)

    selected_idxs = idxs[:args.max_batch_size]
    statements = [statements[_] for _ in selected_idxs]
    sub_cols = [sub_cols[_] for _ in selected_idxs]
    labels = [labels[_] for _ in selected_idxs]

    tab_len = len(table)
    batch_size = len(statements)
    if args.encoding == 'gnn':
        texts = []
        segs = []
        masks = []
        mapping = {}
        cur_index = 0
        for sub_col, stat in zip(sub_cols, statements):
            table_inp = []
            lengths = []

            stat_inp = tokenizer.tokenize('[CLS] ' + stat + ' [SEP]')
            tit_inp = tokenizer.tokenize('title is : ' + title + ' .')
            mapping[(cur_index, -1, -1)] = (0, len(stat_inp))

            prev_position = position = len(stat_inp) + len(tit_inp)
            for i in range(len(table)):
                tmp = tokenizer.tokenize('row {} is : '.format(i + 1))
                table_inp.extend(tmp)
                position += len(tmp)

                entry = table.iloc[i]
                for j, col in enumerate(sub_col):
                    tmp = tokenizer.tokenize('{} is {} , '.format(cols[col], entry[col]))
                    mapping[(cur_index, i, j)] = (position, position + len(tmp))
                    table_inp.extend(tmp)
                    position += len(tmp)

                lengths.append(position - prev_position)
                prev_position = position

            # Tokens
            tokens = stat_inp + tit_inp + table_inp
            tokens = tokens[:args.max_length]
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            texts.append(token_ids)

            # Segment Ids
            seg = [0] * len(stat_inp) + [1] * (len(tit_inp) + len(table_inp))
            seg = seg[:args.max_length]
            segs.append(seg)

            # Masks
            mask = torch.zeros(len(token_ids), len(token_ids))
            start = 0
            mask[start:start + len(stat_inp), :] = 1
            start += len(stat_inp)

            mask[start:start + len(tit_inp), start:start + len(tit_inp)] = 1

            start += len(tit_inp)
            for l in lengths:
                mask[start:start + l, :len(stat_inp) + len(tit_inp)] = 1
                mask[start:start + l, start:start + l] = 1
                start += l
            masks.append(mask)
            cur_index += 1

        max_len = max([len(_) for _ in texts])
        for i in range(len(texts)):
            # Padding the mask
            tmp = torch.zeros(max_len, max_len)
            tmp[:masks[i].shape[0], :masks[i].shape[1]] = masks[i]
            masks[i] = tmp.unsqueeze(0)

            # Padding the Segmentation
            segs[i] = segs[i] + [0] * (max_len - len(segs[i]))
            texts[i] = texts[i] + [tokenizer.pad_token_id] * (max_len - len(texts[i]))

        # Transform into tensor vectors
        inps = torch.tensor(texts).to(device)
        seg_inps = torch.tensor(segs).to(device)
        mask_inps = torch.cat(masks, 0).to(device)

        inputs = {'input_ids': inps, 'attention_mask': mask_inps, 'token_type_ids': seg_inps}
        representation = model('row', **inputs)[0]

        max_len_col = max([len(_) for _ in sub_cols])
        max_len_stat = max([mapping[(_, -1, -1)][1] for _ in range(batch_size)])
        stat_representation = torch.zeros(batch_size, max_len_stat, representation.shape[-1])
        graph_representation = torch.zeros(batch_size, tab_len, max_len_col, representation.shape[-1])

        table_masks = []
        stat_masks = []
        for i in range(batch_size):
            col_len = len(sub_cols[i])
            mask = []
            for j in range(tab_len):
                for k in range(col_len):
                    start, end = mapping[(i, j, k)]
                    if start < representation.shape[1]:
                        tmp = representation[i, start:end]
                        tmp = torch.mean(tmp, 0)
                        graph_representation[i][j][k] = tmp
                        mask.append(1)
                    else:
                        mask.append(0)
            table_masks.append(mask + [0] * (max_len_col * tab_len - len(mask)))

            start, end = mapping[(i, -1, -1)]
            stat_representation[i, start:end] = representation[i, start:end]
            stat_masks.append([1] * end + [0] * (max_len_stat - end))

        stat_representation = stat_representation.to(device)
        # Transpose to make column first, which becomes B x col x T x dim
        graph_representation = graph_representation.transpose(
            1, 2).contiguous().view(-1, tab_len, graph_representation.shape[-1])

        pre_trained = torch.load('masks/{}.bin'.format(f))
        greater_masks = pre_trained['g']
        smaller_masks = pre_trained['s']
        m_matrix = pre_trained['m']  # model('emb', x=pre_trained['m'].long().to(device))
        c_matrix = pre_trained['c']  # model('emb', x=pre_trained['c'].long().to(device))

        graph_masks_greater = torch.zeros(batch_size, max_len_col, tab_len, tab_len)
        graph_masks_smaller = torch.zeros(batch_size, max_len_col, tab_len, tab_len)
        graph_representation_m = torch.zeros(batch_size, tab_len, max_len_col)
        graph_representation_c = torch.zeros(batch_size, tab_len, max_len_col)
        for i in range(batch_size):
            for j in range(max_len_col):
                if j < len(sub_cols[i]):
                    graph_masks_greater[i][j] = greater_masks[sub_cols[i][j]]
                    graph_masks_smaller[i][j] = smaller_masks[sub_cols[i][j]]
                    graph_representation_m[i, :, j] = m_matrix[:, sub_cols[i][j]]
                    graph_representation_c[i, :, j] = c_matrix[:, sub_cols[i][j]]

        inputs = {'d_node': graph_representation.to(device),
                  'greater_graph': graph_masks_greater.view(-1, tab_len, tab_len).to(device),
                  'smaller_graph': graph_masks_smaller.view(-1, tab_len, tab_len).to(device)}
        graph_representation = model('gnn', **inputs)
        graph_representation = graph_representation.transpose(1, 2).contiguous().view(
            len(statements), -1, graph_representation.shape[-1]).to(device)

        graph_representation_m = model('emb', x=graph_representation_m.view(batch_size, -1).long().to(device))
        graph_representation_c = model('emb', x=graph_representation_c.view(batch_size, -1).long().to(device))

        graph_representation = graph_representation + graph_representation_m + graph_representation_c
        if args.attention == 'self':
            x_masks = torch.cat([torch.tensor(stat_masks), torch.tensor(table_masks)], 1).to(device)
            representation = torch.cat([stat_representation, graph_representation], 1)
            inputs = {'x': representation.to(device), 'x_mask': (1 - x_masks).unsqueeze(1).unsqueeze(2).byte()}
            logits = model('sa', **inputs)
        elif args.attention == 'cross':
            inputs = {'x': stat_representation, 'x_mask': torch.tensor(stat_masks).to(device),
                      'y': graph_representation, 'y_mask': torch.tensor(table_masks).to(device)}
            logits = model('sa', **inputs)
    elif args.encoding == 'gnn_ind':
        stats = []
        texts = []
        segs = []
        masks = []
        mapping = {}
        cur_index = 0
        for sub_col, stat in zip(sub_cols, statements):
            table_inp = []
            lengths = []

            stats.append(tokenizer.encode('[CLS] ' + stat))

            tit_inp = tokenizer.tokenize('title is : ' + title + ' .')
            prev_position = position = len(tit_inp)
            for i in range(len(table)):
                tmp = tokenizer.tokenize('row {} is : '.format(i + 1))
                table_inp.extend(tmp)
                position += len(tmp)

                entry = table.iloc[i]
                for j, col in enumerate(sub_col):
                    tmp = tokenizer.tokenize('{} is {} , '.format(cols[col], entry[col]))
                    mapping[(cur_index, i, j)] = (position, position + len(tmp))
                    table_inp.extend(tmp)
                    position += len(tmp)

                lengths.append(position - prev_position)
                prev_position = position

            # Tokens
            tokens = tit_inp + table_inp
            tokens = tokens[:args.max_length]
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            texts.append(token_ids)

            # Segment Ids
            seg = [0] * len(tit_inp) + [1] * len(table_inp)
            seg = seg[:args.max_length]
            segs.append(seg)

            # Masks
            mask = torch.zeros(len(token_ids), len(token_ids))
            start = 0
            mask[start:start + len(tit_inp), start:start + len(tit_inp)] = 1

            start += len(tit_inp)
            for l in lengths:
                mask[start:start + l, :len(tit_inp)] = 1
                mask[start:start + l, start:start + l] = 1
                start += l
            masks.append(mask)
            cur_index += 1

        # For the statements
        max_len = max([len(_) for _ in stats])
        stat_masks = []
        for i in range(len(stats)):
            # Padding the mask
            stat_masks.append([1] * len(stats[i]) + [0] * (max_len - len(stats[i])))
            stats[i] = stats[i] + [tokenizer.pad_token_id] * (max_len - len(stats[i]))

        # Transform into tensor vectors
        stat_masks = torch.tensor(stat_masks)
        stat_representation = model('row', input_ids=torch.tensor(stats).to(device),
                                    attention_mask=(1 - stat_masks).byte().to(device))[0]

        # For the table
        max_len = max([len(_) for _ in texts])
        for i in range(len(texts)):
            # Padding the mask
            tmp = torch.zeros(max_len, max_len)
            tmp[:masks[i].shape[0], :masks[i].shape[1]] = masks[i]
            masks[i] = tmp.unsqueeze(0)
            # Padding the Segmentation
            segs[i] = segs[i] + [0] * (max_len - len(segs[i]))
            texts[i] = texts[i] + [tokenizer.pad_token_id] * (max_len - len(texts[i]))

        # Transform into tensor vectors
        representation = model('row', input_ids=torch.tensor(texts).to(device),
                               attention_mask=torch.cat(masks, 0).to(device),
                               token_type_ids=torch.tensor(segs).to(device))[0]

        max_len_col = max([len(_) for _ in sub_cols])
        graph_representation = torch.zeros(batch_size, tab_len, max_len_col, representation.shape[-1])
        table_masks = []
        for i in range(batch_size):
            col_len = len(sub_cols[i])
            mask = []
            for j in range(tab_len):
                for k in range(col_len):
                    start, end = mapping[(i, j, k)]
                    if start < representation.shape[1]:
                        tmp = representation[i, start:end]
                        tmp = torch.mean(tmp, 0)
                        graph_representation[i][j][k] = tmp
                        mask.append(1)
                    else:
                        mask.append(0)
            table_masks.append(mask + [0] * (max_len_col * tab_len - len(mask)))

        table_masks = torch.tensor(table_masks)
        # Transpose to make column first, which becomes B x col x T x dim
        graph_representation = graph_representation.transpose(
            1, 2).contiguous().view(-1, tab_len, graph_representation.shape[-1])

        pre_trained = torch.load('masks/{}.bin'.format(f))
        greater_masks = pre_trained['g']
        smaller_masks = pre_trained['s']
        m_matrix = pre_trained['m']
        c_matrix = pre_trained['c']

        graph_masks_greater = torch.zeros(batch_size, max_len_col, tab_len, tab_len)
        graph_masks_smaller = torch.zeros(batch_size, max_len_col, tab_len, tab_len)
        graph_representation_m = torch.zeros(batch_size, tab_len, max_len_col)
        graph_representation_c = torch.zeros(batch_size, tab_len, max_len_col)
        for i in range(batch_size):
            for j in range(max_len_col):
                if j < len(sub_cols[i]):
                    graph_masks_greater[i][j] = greater_masks[sub_cols[i][j]]
                    graph_masks_smaller[i][j] = smaller_masks[sub_cols[i][j]]
                    graph_representation_m[i, :, j] = m_matrix[:, sub_cols[i][j]]
                    graph_representation_c[i, :, j] = c_matrix[:, sub_cols[i][j]]

        inputs = {'d_node': graph_representation.to(device),
                  'greater_graph': graph_masks_greater.view(-1, tab_len, tab_len).to(device),
                  'smaller_graph': graph_masks_smaller.view(-1, tab_len, tab_len).to(device)}
        graph_representation = model('gnn', **inputs)
        graph_representation = graph_representation.transpose(1, 2).contiguous().view(
            len(statements), -1, graph_representation.shape[-1]).to(device)

        graph_representation_m = model('emb', x=graph_representation_m.view(batch_size, -1).long().to(device))
        graph_representation_c = model('emb', x=graph_representation_c.view(batch_size, -1).long().to(device))

        graph_representation = graph_representation + graph_representation_m + graph_representation_c
        if args.attention == 'self':
            x_masks = torch.cat([stat_masks, table_masks], 1).to(device)
            representation = torch.cat([stat_representation, graph_representation], 1)
            inputs = {'x': representation.to(device), 'x_mask': (1 - x_masks).unsqueeze(1).unsqueeze(2).byte()}
            logits = model('sa', **inputs)
        elif args.attention == 'cross':
            inputs = {'x': stat_representation, 'x_mask': stat_masks.to(device),
                      'y': graph_representation, 'y_mask': table_masks.to(device)}
            logits = model('sa', **inputs)
    else:
        raise NotImplementedError

    labels = torch.LongTensor(labels).to(device)

    return logits, labels


if __name__ == "__main__":
    args = parse_opt()

    tokenizer = BertTokenizer.from_pretrained(args.model)

    model = GNN(args.dim, args.head, args.model, 2, attention=args.attention)
    model.to(device)

    if args.do_train:
        # Create the folder for the saving the intermediate files
        args.output_dir = '{}_{}'.format(args.output_dir, args.id)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        with open('data/train_examples.json') as f:
            examples = json.load(f)
        files = list(examples.keys())

        writer = SummaryWriter(os.path.join(args.output_dir, 'events'))

        with open(os.path.join(args.output_dir, 'config.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        optimizer = AdamW(model.parameters(), lr=args.lr_default, eps=1e-8)
        t_total = len(examples) * args.epochs

        warm_up_steps = 0.1 * t_total
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warm_up_steps, num_training_steps=t_total
        )

        if args.fp16:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

        cross_entropy = torch.nn.CrossEntropyLoss()

        global_step = 0
        for epoch_ in range(args.epochs):
            random.shuffle(files)
            model.zero_grad()
            optimizer.zero_grad()
            print("starting the training of {}th epoch".format(epoch_))

            local_step = 0
            total_steps = len(files)

            for f in tqdm(files, desc="Iteration"):
                # for f in files:
                table = pandas.read_csv('all_csv/{}'.format(f), '#')
                table = table.head(40)

                logits, labels = forward_pass(table, examples[f], model)

                loss = cross_entropy(logits.view(-1, 2), labels)
                writer.add_scalar('train/loss', loss, global_step)

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                if (local_step + 1) % args.gradient_accumulation_steps == 0:
                    #torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)

                    total_norm = 0.0
                    for n, p in model.named_parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2

                    total_norm = total_norm ** (1. / 2)
                    writer.add_scalar('train/gradient_norm', total_norm, global_step)

                    learning_rate_scalar = scheduler.get_lr()[0]
                    writer.add_scalar('train/lr', learning_rate_scalar, global_step)

                    preds = (torch.argmax(logits, -1) == labels)
                    acc = torch.sum(preds).float() / preds.size(0)

                    optimizer.step()
                    scheduler.step()

                    model.zero_grad()

                    global_step += 1
                    local_step += 1

                if (local_step + 1) % 2000 == 0:
                    with open('data/test_examples.json') as f:
                        test_examples = json.load(f)

                    model.eval()
                    with torch.no_grad():
                        correct, total = 0, 0
                        for f in tqdm(test_examples.keys(), 'Evaluation'):
                            table = pandas.read_csv('all_csv/{}'.format(f), '#')
                            table = table.head(40)

                            logits, labels = forward_pass(table, test_examples[f], model)

                            preds = torch.argmax(logits, -1)

                            correct_or_not = (preds == labels)

                            correct += (correct_or_not).sum().item()
                            total += correct_or_not.shape[0]

                    acc = correct / total
                    print('evaluation results (accuracy) = {}'.format(acc))
                    writer.add_scalar('val/acc', acc, global_step)

                    model.train()

            torch.save(model.state_dict(), '{}/model_ep{}.pt'.format(args.output_dir, epoch_))

    if args.do_test:
        model.load_state_dict(torch.load(args.load_from))
        model.eval()

        if args.simple:
            with open('data/simple_test_examples.json') as f:
                examples = json.load(f)
        if args.complex:
            with open('data/complex_test_examples.json') as f:
                examples = json.load(f)
        if args.complex:
            with open('data/test_examples.json') as f:
                examples = json.load(f)

        files = list(examples.keys())

        with torch.no_grad():
            correct, total = 0, 0
            for f in tqdm(files, "Evaluation"):
                table = pandas.read_csv('all_csv/{}'.format(f), '#')
                table = table.head(40)

                logits, labels = forward_pass(table, examples[f], model)

                preds = torch.argmax(logits, -1)

                correct_or_not = (preds == labels)

                correct += (correct_or_not).sum().item()
                total += correct_or_not.shape[0]

                acc = correct / total
                #sys.stdout.write("finished {}/{}, the accuracy is {} \r".format(i, len(files), acc))

            print("the final accuracy is {}".format(acc))
