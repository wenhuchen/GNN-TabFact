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
    parser.add_argument('--epochs', default=200, type=int, help="whether to train or test the model")
    parser.add_argument('--split', default=256, type=int, help="whether to train or test the model")
    parser.add_argument('--max_len', default=30, type=int, help="whether to train or test the model")
    parser.add_argument('--do_train', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_test', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_tabfact', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--fp16', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--lr_default', type=float, default=5e-6, help="whether to train or test the model")
    parser.add_argument('--load_from', default='', type=str, help="whether to train or test the model")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--batch_size", default=16, type=int, help="Max gradient norm.")
    parser.add_argument('--model', default='bert-base-multilingual-uncased', type=str, help='model to use')
    parser.add_argument('--output_dir', default='models/number', type=str, help='model to use')
    parser.add_argument('--encoding', default='concat', type=str,
                        help='the type of table encoder; choose from concat|row|cell')
    parser.add_argument('--max_num', default=500, type=int, help='model to use')
    parser.add_argument('--max_length', default=512, type=int, help='model to use')
    parser.add_argument('--max_batch_size', default=64, type=int, help='model to use')
    parser.add_argument('--id', default=1, type=int, help='model to use')

    args = parser.parse_args()

    return args


dictionary = {str(_): _ for _ in range(2000)}
i_dictionary = {v: k for k, v in dictionary.items()}


def forward_pass(examples, model):
    # cols = table.columns
    texts = []
    labels = []
    for e in examples:
        if isinstance(e, dict):
            texts.append(tokenizer.encode('[CLS] ' + e['sent']))
            labels.append(dictionary[e['answer']])
        else:
            texts.append(tokenizer.encode(e))
            labels.append(0)

    masks = []
    max_len = max([len(_) for _ in texts])
    for i in range(len(texts)):
        masks.append([1] * len(texts[i]) + [0] * (max_len - len(texts[i])))
        texts[i] = texts[i] + [tokenizer.pad_token_id] * (max_len - len(texts[i]))

    texts = torch.tensor(texts).to(device)
    labels = torch.tensor(labels).to(device)
    masks = torch.tensor(masks).to(device)

    logits = model('cell', input_ids=texts, attention_mask=masks)

    return logits, labels


def gen_dataset():
    people = ['dave', 'jesus', 'mary', 'nancy', 'sampson', 'cuba', 'sam']
    examples = []
    visited = set()
    for i in range(300000):
        e = {}
        a = random.choice(people)
        b = random.choice(people)

        c = random.randint(0, 500)
        d = random.randint(0, 500)

        if (c, d) not in visited:
            ans = c + d
            e['sent'] = 'how many apples do they have in total ? [SEP] {} gets {} apples, while {} gets {} apples'.format(
                a, c, b, d)
            e['answer'] = str(ans)
            examples.append(e)
            visited.add((c, d))

    return examples


if __name__ == "__main__":
    args = parse_opt()

    config = BertConfig.from_pretrained(args.model, cache_dir='tmp/')
    tokenizer = BertTokenizer.from_pretrained(args.model, cache_dir='tmp/')

    model = Baseline1(args.dim, args.model, config, len(dictionary))
    model.to(device)

    if args.do_train:
        # Create the folder for the saving the intermediate files
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        if args.load_from != '':
            model.load_state_dict(torch.load(args.load_from))

        examples = gen_dataset()

        # Split the train/test set
        train_examples = examples[:int(len(examples) * 0.9)]
        test_examples = examples[int(len(examples) * 0.9):]

        writer = SummaryWriter(os.path.join(args.output_dir, 'events'))

        with open(os.path.join(args.output_dir, 'config.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        optimizer = AdamW(model.parameters(), lr=args.lr_default, eps=1e-8)
        t_total = len(train_examples) * args.epochs // args.max_batch_size

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
            random.shuffle(train_examples)
            model.zero_grad()
            optimizer.zero_grad()
            print("starting the training of {}th epoch".format(epoch_))

            local_step = 0

            for cur_idx in tqdm(range(0, len(train_examples), args.max_batch_size), desc="Iteration"):
                cur_examples = [train_examples[_]
                                for _ in range(cur_idx, min(cur_idx + args.max_batch_size, len(train_examples)))]

                if len(cur_examples) == 0:
                    break

                logits, labels = forward_pass(cur_examples, model)

                loss = cross_entropy(logits.view(-1, len(dictionary)), labels)
                writer.add_scalar('train/loss', loss, global_step)

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                if (local_step + 1) % args.gradient_accumulation_steps == 0:
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

                if (global_step + 1) % 2000 == 0:
                    model.eval()
                    with torch.no_grad():
                        correct, total = 0, 0
                        for cur_idx in tqdm(range(0, len(test_examples), args.max_batch_size), desc="Evaluation"):
                            cur_examples = [test_examples[_]
                                            for _ in range(cur_idx, min(cur_idx + args.max_batch_size, len(test_examples)))]
                            logits, labels = forward_pass(cur_examples, model)

                            preds = torch.argmax(logits, -1)

                            correct_or_not = (preds == labels)

                            correct += (correct_or_not).sum().item()
                            total += correct_or_not.shape[0]

                    acc = correct / total
                    print('evaluation results (accuracy) = {}'.format(acc))
                    writer.add_scalar('val/acc', acc, global_step)

                    torch.save(model.state_dict(), '{}/model_step{}.pt'.format(args.output_dir, global_step))
                    model.train()
