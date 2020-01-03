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

    if args.encoding == 'concat':
        texts = []
        segs = []
        for sub_col, stat in zip(sub_cols, statements):
            text = ''
            for i in range(len(table)):
                text += 'row {} is : '.format(i + 1)
                entry = table.iloc[i]
                for col in sub_col:
                    text += '{} is {} , '.format(cols[col], entry[col])
                if i < len(table) - 1:
                    text = text[:-2] + ' . '
                else:
                    text = text[:-2]

            stat_inp = tokenizer.tokenize(stat)
            tit_inp = tokenizer.tokenize(title)
            table_inp = tokenizer.tokenize(text)

            tokens = ['[CLS]'] + stat_inp + ['[SEP]'] + tit_inp + table_inp
            tokens = tokens[:args.max_length]
            seg = [0] * (len(stat_inp) + 2) + [1] * (len(tit_inp) + len(table_inp))
            seg = seg[:args.max_length]

            segs.append(seg)
            texts.append(tokenizer.convert_tokens_to_ids(tokens))

        masks = []
        max_len = max([len(_) for _ in texts])
        for i in range(len(texts)):
            masks.append([1] * len(texts[i]) + [0] * (max_len - len(texts[i])))
            segs[i] = segs[i] + [0] * (max_len - len(segs[i]))
            texts[i] = texts[i] + [tokenizer.pad_token_id] * (max_len - len(texts[i]))

        inps = torch.tensor(texts).to(device)
        seg_inps = torch.tensor(segs).to(device)
        mask_inps = torch.tensor(masks).to(device)

        inputs = {'input_ids': inps, 'attention_mask': mask_inps, 'token_type_ids': seg_inps}
        logits = model('cell', **inputs)

    elif args.encoding == 'row':
        texts = []
        stats = []
        for sub_col, stat in zip(sub_cols, statements):
            for i in range(len(table)):
                text = 'row {} is : '.format(i + 1)
                entry = table.iloc[i]
                for col in sub_col:
                    text += '{} is {} , '.format(cols[col], entry[col])
                text = text[:-2] + ' .'

                text = tokenizer.tokenize(text)[:40]
                text = tokenizer.convert_tokens_to_ids(text)
                texts.append(text)

            stats.append(tokenizer.encode('[CLS] ' + stat + ' [SEP] ' + title))

        masks = []
        max_len = max([len(_) for _ in texts])
        for i in range(len(texts)):
            masks.append([1] * len(texts[i]) + [0] * (max_len - len(texts[i])))
            texts[i] = texts[i] + [tokenizer.pad_token_id] * (max_len - len(texts[i]))

        stat_masks = []
        max_len_stat = max([len(_) for _ in stats])
        for i in range(len(stats)):
            stat_masks.append([1] * len(stats[i]) + [0] * (max_len_stat - len(stats[i])))
            stats[i] = stats[i] + [tokenizer.pad_token_id] * (max_len_stat - len(stats[i]))

        inps = torch.tensor(texts).to(device)
        mask_inps = torch.tensor(masks).to(device)
        stats = torch.tensor(stats).to(device)
        stat_mask_inps = torch.tensor(stat_masks).to(device)

        inputs = {'input_ids': inps, 'attention_mask': mask_inps, 'token_type_ids': None}
        row_representations = model('row', **inputs)[0]

        row_representations = row_representations.view(len(statements), -1, row_representations.shape[-1])

        inputs = {'input_ids': stats, 'attention_mask': stat_mask_inps, 'token_type_ids': None}
        stat_representations = model('row', **inputs)[0]

        representations = torch.cat([stat_representations, row_representations], 1)
        row_mask_inps = torch.ones(row_representations.shape[0], row_representations.shape[1]).long().to(device)
        # mask_inps.view(stat_mask_inps.shape[0], -1)

        #segs = torch.cat([torch.zeros_like(stat_mask_inps), torch.ones_like(row_mask_inps)], 1)
        #mask = torch.cat([stat_mask_inps, row_mask_inps], 1)

        inputs = {'x': representations, 'x_mask': (1 - mask).unsqueeze(1).unsqueeze(2).type(torch.bool)}

        logits = model('sa', segs, **inputs)

    elif args.encoding == 'concat_row':
        texts = []
        segs = []
        separations = []
        for sub_col, stat in zip(sub_cols, statements):
            text = ''
            for i in range(len(table)):
                text += 'row {} is : '.format(i + 1)
                entry = table.iloc[i]
                for col in sub_col:
                    text += '{} is {} , '.format(cols[col], entry[col])
                if i < len(table) - 1:
                    text = text[:-2] + ' . '
                else:
                    text = text[:-2]

            stat_inp = tokenizer.tokenize(stat)
            tit_inp = tokenizer.tokenize(title)
            table_inp = tokenizer.tokenize(text)
            tokens = ['[CLS]'] + stat_inp + ['[SEP]'] + tit_inp + table_inp
            texts.append(tokenizer.convert_tokens_to_ids(tokens))

            separations.append(tokens.index('[SEP]'))
            tokens = tokens[:args.max_length]
            seg = [0] * (len(stat_inp) + 2) + [1] * (len(tit_inp) + len(table_inp))
            seg = seg[:args.max_length]
            segs.append(seg)

        masks = []
        lengths = [len(_) for _ in texts]
        max_len = max(lengths)
        for i in range(len(texts)):
            masks.append([1] * len(texts[i]) + [0] * (max_len - len(texts[i])))
            segs[i] = segs[i] + [0] * (max_len - len(segs[i]))
            texts[i] = texts[i] + [tokenizer.pad_token_id] * (max_len - len(texts[i]))

        inps = torch.tensor(texts).to(device)
        seg_inps = torch.tensor(segs).to(device)
        mask_inps = torch.tensor(masks).to(device)

        inputs = {'input_ids': inps, 'attention_mask': mask_inps, 'token_type_ids': seg_inps}
        representations = model('row', **inputs)[0]

        bi_mask = []
        for i in range(len(separations)):
            tmp_mask_1 = torch.tensor([0] * (separations[i] + 1) + [1] * (lengths[i] - (separations[i] + 1)) +
                                      [0] * (max_len - lengths[i]))
            tmp_mask_2 = 1 - tmp_mask_1
            tmp_mask_1 = torch.cat([tmp_mask_1.unsqueeze(0)] * (separations[i] + 1), 0)
            tmp_mask_2 = torch.cat([tmp_mask_2.unsqueeze(0)] * (max_len - (separations[i] + 1)), 0)
            mask = torch.cat([tmp_mask_1, tmp_mask_2], 0)
            bi_mask.append(mask.unsqueeze(0))

        bi_mask = torch.cat(bi_mask, 0).unsqueeze(1).to(device)
        logits = model('sa', x=representations, x_mask=(1 - bi_mask).type(torch.bool))

    elif args.encoding == 'concat_row_sparse':
        texts = []
        segs = []
        masks = []
        for sub_col, stat in zip(sub_cols, statements):
            table_inp = []
            lengths = []
            for i in range(len(table)):
                text = 'row {} is : '.format(i + 1)
                entry = table.iloc[i]
                for col in sub_col:
                    text += '{} is {} , '.format(cols[col], entry[col])
                if i < len(table) - 1:
                    text = text[:-2] + ' . '
                else:
                    text = text[:-2]
                tokens = tokenizer.tokenize(text)
                lengths.append(len(tokens))
                table_inp.extend(tokens)

            stat_inp = tokenizer.tokenize(stat)
            tit_inp = tokenizer.tokenize(title)

            # Tokens
            tokens = ['[CLS]'] + stat_inp + ['[SEP]'] + tit_inp + table_inp
            tokens = tokens[:args.max_length]
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            texts.append(token_ids)

            # Segment Ids
            seg = [0] * (len(stat_inp) + 2) + [1] * (len(tit_inp) + len(table_inp))
            seg = seg[:args.max_length]
            segs.append(seg)

            # Masks
            mask = torch.zeros(len(token_ids), len(token_ids))
            start = 0
            mask[start:start + len(stat_inp) + 2, :] = 1
            start += len(stat_inp) + 2

            mask[start:start + len(tit_inp), :start + len(tit_inp)] = 1

            start += len(tit_inp)
            for l in lengths:
                mask[start:start + l, :len(stat_inp) + len(tit_inp)] = 1
                mask[start:start + l, start:start + l] = 1
                start += l
            masks.append(mask)

        max_len = max([len(_) for _ in texts])
        for i in range(len(texts)):
            #masks.append([1] * len(texts[i]) + [0] * (max_len - len(texts[i])))
            tmp = torch.zeros(max_len, max_len)
            tmp[:masks[i].shape[0], :masks[i].shape[1]] = masks[i]
            masks[i] = tmp.unsqueeze(0)

            segs[i] = segs[i] + [0] * (max_len - len(segs[i]))
            texts[i] = texts[i] + [tokenizer.pad_token_id] * (max_len - len(texts[i]))

        inps = torch.tensor(texts).to(device)
        seg_inps = torch.tensor(segs).to(device)
        mask_inps = torch.cat(masks, 0).to(device)

        inputs = {'input_ids': inps, 'attention_mask': mask_inps, 'token_type_ids': seg_inps}
        logits = model('cell', **inputs)

    elif args.encoding == 'cell':
        texts = []
        stats = []
        positions = []
        for sub_col, stat in zip(sub_cols, statements):
            for i in range(len(table)):
                text = 'row {} is : '.format(i + 1)
                entry = table.iloc[i]
                for col in sub_col:
                    text += '[SEP] {} is {} [SEP]'.format(cols[col], entry[col])

                text = tokenizer.tokenize(text)[:40]
                text = tokenizer.convert_tokens_to_ids(text)

                pos = []
                for j, t in enumerate(text):
                    if j == len(text) - 1 and t != tokenizer.sep_token_id:
                        t = tokenizer.sep_token_id

                    if t == tokenizer.sep_token_id:
                        if len(pos) == 0:
                            pos.append((0, j))
                        else:
                            pos.append((pos[-1][1], j))

                texts.append(text)
                positions.append(pos)

            stats.append(tokenizer.encode('[CLS] ' + stat))

        masks = []
        max_len = max([len(_) for _ in texts])
        for i in range(len(texts)):
            masks.append([1] * len(texts[i]) + [0] * (max_len - len(texts[i])))
            texts[i] = texts[i] + [tokenizer.pad_token_id] * (max_len - len(texts[i]))

        stat_masks = []
        max_len_stat = max([len(_) for _ in stats])
        for i in range(len(stats)):
            stat_masks.append([1] * len(stats[i]) + [0] * (max_len_stat - len(stats[i])))
            stats[i] = stats[i] + [tokenizer.pad_token_id] * (max_len_stat - len(stats[i]))

        inps = torch.tensor(texts).to(device)
        mask_inps = torch.tensor(masks).to(device)
        stats = torch.tensor(stats).to(device)
        stat_mask_inps = torch.tensor(stat_masks).to(device)

        inputs = {'input_ids': inps, 'attention_mask': mask_inps, 'token_type_ids': None}
        row_representations = model('row', **inputs)[0]

        tmps = []
        for i in range(0, row_representations.shape[0], len(table)):
            tmp = []
            for idx in range(i, i + len(table)):
                for pos in positions[idx]:
                    _ = row_representations[idx][pos[0]:pos[1] + 1]
                    _ = torch.mean(_, 0)
                    tmp.append(_.unsqueeze(0))
            tmps.append(tmp)

        max_len = max([len(_) for _ in tmps])
        cell_representation = torch.zeros(len(statements), max_len, row_representations.shape[-1]).to(device)
        cell_mask_inps = torch.zeros(len(statements), max_len).to(device).long()
        for i, tmp in enumerate(tmps):
            cell_representation[i][:len(tmp)] = torch.cat(tmp, 0)
            cell_mask_inps[i][:len(tmp)] = 1

        inputs = {'input_ids': stats, 'attention_mask': stat_mask_inps, 'token_type_ids': None}
        stat_representations = model('row', **inputs)[0]

        representations = torch.cat([stat_representations, cell_representation], 1)
        #row_mask_inps = torch.ones(row_representations.shape[0], row_representations.shape[1]).long().to(device)

        segs = torch.cat([torch.zeros_like(stat_mask_inps), torch.ones_like(cell_mask_inps)], 1)
        mask = torch.cat([stat_mask_inps, cell_mask_inps], 1)

        inputs = {'x': representations, 'x_mask': (1 - mask).unsqueeze(1).unsqueeze(2).type(torch.bool)}

        logits = model('sa', segs, **inputs)
    else:
        raise NotImplementedError

    labels = torch.LongTensor(labels).to(device)

    return logits, labels


if __name__ == "__main__":
    args = parse_opt()

    tokenizer = BertTokenizer.from_pretrained(args.model)

    # if args.encoding in ['row', 'cell', 'concat_row']:
    #    model = Baseline2(args.dim, args.head, args.model, 2)
    # else:
    #    model = Baseline1(args.dim, args.head, args.model, 2)
    model = Baseline2(args.dim, args.head, args.model, 2)
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

        with open('data/test_examples.json') as f:
            examples = json.load(f)
        files = list(examples.keys())

        with torch.no_grad():
            correct, total = 0, 0
            for i, f in enumerate(files):
                table = pandas.read_csv('all_csv/{}'.format(f), '#')
                table = table.head(40)

                logits, labels = forward_pass(table, examples[f], model)

                preds = torch.argmax(logits, -1)

                correct_or_not = (preds == labels)

                correct += (correct_or_not).sum().item()
                total += correct_or_not.shape[0]

                acc = correct / total
                sys.stdout.write("finished {}/{}, the accuracy is {} \r".format(i, len(files), acc))
