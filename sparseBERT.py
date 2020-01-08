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

SEPARATIONS = [0, 6, 5, 7, 12]


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', default=768, type=int, help="whether to train or test the model")
    parser.add_argument('--head', default=4, type=int, help="whether to train or test the model")
    parser.add_argument('--layers', default=3, type=int, help="whether to train or test the model")
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
    parser.add_argument('--with_title', default=False, action='store_true', help='model to use')
    parser.add_argument('--free_attention', default=False, action='store_true', help='model to use')
    parser.add_argument('--max_length', default=512, type=int, help='model to use')
    parser.add_argument('--max_batch_size', default=18, type=int, help='model to use')
    parser.add_argument('--id', default=1, type=int, help='model to use')
    args = parser.parse_args()

    return args


def paralell_table(table, cols):
    # Parallel text encoding
    texts = []
    for i in range(len(table)):
        entry = table.iloc[i]
        for j, col in enumerate(table.columns):
            text = '[CLS] in row {}, {} is {} .'.format(i + 1, col, entry[col])
            tmp = tokenizer.encode(text)
            if len(tmp) > args.max_len:
                tmp = tmp[:args.max_len]
            texts.append(tmp)

    max_len = max([len(_) for _ in texts])
    masks = []
    token_types = []

    for i in range(len(texts)):
        masks.append([1] * len(texts[i]) + [0] * (max_len - len(texts[i])))
        texts[i] = texts[i] + [tokenizer.pad_token_id] * (max_len - len(texts[i]))
        token_types.append([1] * max_len)

    return torch.tensor(texts), torch.tensor(masks), torch.tensor(token_types)


def distribute_encode(texts, masks, token_types, model, table, cols):
    # Using BERT to encode the cells
    encoded = []
    splits = len(texts) // args.split + 1
    for _ in range(splits):
        if _ * args.split < len(texts):
            text_split = texts[_ * args.split: (_ + 1) * args.split]
            mask_split = masks[_ * args.split: (_ + 1) * args.split]
            token_type_split = token_types[_ * args.split: (_ + 1) * args.split]
            # Inputs to the model
            inputs = {'input_ids': text_split.to(device), 'attention_mask': mask_split.unsqueeze(1).to(device),
                      'token_type_ids': token_type_split.to(device), 'i': SEPARATIONS[0], 'j': SEPARATIONS[1]}
            tmp = model('intermediate', **inputs)[:, 0]  # .view(-1, args.hidden_dim)
            encoded.append(tmp)

    encoded = torch.cat(encoded, 0)
    representation = encoded.view(len(table), -1, args.dim)
    return representation


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

    texts, masks, token_types = paralell_table(table, cols)
    cell_representation = distribute_encode(texts, masks, token_types, model, table, cols)
    cell_representation = model('calibrate', x=cell_representation)

    hidden_dim = cell_representation.shape[-1]
    max_col = max([len(_) for _ in sub_cols])
    row_representation = torch.zeros(batch_size, tab_len, max_col, cell_representation.shape[-1])
    row_mask = torch.zeros(batch_size, tab_len, max_col)
    idx = 0

    texts = []
    for sub_col, statement in zip(sub_cols, statements):
        for j, col in enumerate(sub_col):
            row_representation[idx, :, j, :] = cell_representation[:, col, :]
            row_mask[idx, :, j] = 1
        if args.with_title:
            texts.append(tokenizer.encode('[CLS] ' + statement + ' [SEP] ' + title + ' [SEP] '))
        else:
            texts.append(tokenizer.encode('[CLS] ' + statement + ' [SEP] '))
        idx += 1

    max_len = max([len(_) for _ in texts])
    masks = []
    token_types = []
    for i in range(batch_size):
        masks.append([1] * len(texts[i]) + [0] * (max_len - len(texts[i])))
        texts[i] = texts[i] + [tokenizer.pad_token_id] * (max_len - len(texts[i]))
        token_types.append([0] * max_len)

    texts = torch.tensor(texts).to(device)
    masks = torch.FloatTensor(masks).to(device)
    token_types = torch.tensor(token_types).to(device)
    stat_representation = model('intermediate', i=SEPARATIONS[0], j=SEPARATIONS[1], input_ids=texts,
                                attention_mask=masks.unsqueeze(1), token_type_ids=token_types)

    # Ablation Study
    table_representation = row_representation.view(batch_size, -1, hidden_dim).to(device)
    representation = torch.cat([stat_representation, table_representation], 1)
    full_mask = torch.cat([masks, row_mask.view(batch_size, -1).to(device)], 1).unsqueeze(1)
    """
    row_representation = row_representation.view(batch_size * tab_len, max_col, hidden_dim).to(device)
    row_mask = row_mask.view(batch_size * tab_len, max_col).to(device)
    row_representation = model('intermediate', i=SEPARATIONS[1], j=SEPARATIONS[2], attention_mask=row_mask.unsqueeze(
        1), hidden_states=row_representation)

    column_representation = row_representation.view(
        batch_size, tab_len, max_col, hidden_dim).transpose(1, 2).contiguous().view(-1, tab_len, hidden_dim)
    column_mask = row_mask.view(batch_size, tab_len, max_col).transpose(1, 2).contiguous().view(-1, tab_len)
    column_representation = model('intermediate', i=SEPARATIONS[2], j=SEPARATIONS[3], attention_mask=column_mask.unsqueeze(
        1), hidden_states=column_representation)

    table_representation = column_representation.view(
        batch_size, max_col, tab_len, hidden_dim).transpose(1, 2).contiguous().view(batch_size, -1, hidden_dim)

    representation = torch.cat([stat_representation, table_representation.view(batch_size, -1, hidden_dim)], 1)

    if args.free_attention:
        # Free attention to interact between all words
        full_mask = torch.cat([masks, row_mask.view(batch_size, -1)], 1).unsqueeze(1)
    else:
        # Form the upper mask
        upper_mask = torch.cat([torch.zeros(batch_size, max_len).to(device), row_mask.view(batch_size, -1)], 1)
        upper_mask = upper_mask.unsqueeze(1).repeat(1, max_len, 1)
        # Form the lower mask
        lower_mask = torch.cat([masks, torch.zeros(batch_size, tab_len * max_col).to(device)], 1)
        lower_mask = lower_mask.unsqueeze(1).repeat(1, tab_len * max_col, 1)
        # Concatenate the two masks
        full_mask = torch.cat([upper_mask, lower_mask], 1)
    """
    logits = model('final', i=SEPARATIONS[1], j=SEPARATIONS[4], attention_mask=full_mask, hidden_states=representation)

    labels = torch.LongTensor(labels).to(device)

    return logits, labels


def forward_pass_only_stat(table, example, model):
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

    texts = []
    idx = 0
    for sub_col, statement in zip(sub_cols, statements):
        if args.with_title:
            texts.append(tokenizer.encode('[CLS] ' + statement + ' [SEP] ' + title + ' [SEP] '))
        else:
            texts.append(tokenizer.encode('[CLS] ' + statement + ' [SEP] '))
        idx += 1

    max_len = max([len(_) for _ in texts])
    masks = []
    token_types = []
    for i in range(batch_size):
        masks.append([1] * len(texts[i]) + [0] * (max_len - len(texts[i])))
        texts[i] = texts[i] + [tokenizer.pad_token_id] * (max_len - len(texts[i]))
        token_types.append([0] * max_len)

    texts = torch.tensor(texts).to(device)
    masks = torch.FloatTensor(masks).to(device)
    token_types = torch.tensor(token_types).to(device)
    logits = model('final', i=SEPARATIONS[0], j=SEPARATIONS[-1], input_ids=texts,
                   attention_mask=masks.unsqueeze(1), token_type_ids=token_types)
    labels = torch.LongTensor(labels).to(device)

    return logits, labels


if __name__ == "__main__":
    args = parse_opt()

    config = BertConfig.from_pretrained(args.model, cache_dir='tmp/')
    tokenizer = BertTokenizer.from_pretrained(args.model, cache_dir='tmp/')

    model = TabularBert(args.dim, args.head, args.model, config, 2)
    model.to(device)

    if args.do_train:
        # Create the folder for the saving the intermediate files
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
                table = pandas.read_csv('all_csv/{}'.format(f), '#')
                table = table.head(40)

                logits, labels = forward_pass_only_stat(table, examples[f], model)

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
