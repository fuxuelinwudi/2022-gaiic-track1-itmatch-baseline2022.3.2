# -*- coding: utf-8 -*-

import gc
import json
import time
import warnings
from argparse import ArgumentParser
from src.utils.classifier_utils import *


def read_data(args, tokenizer):
    train_inputs, dev_inputs = defaultdict(list), defaultdict(list)
    with open(args.train_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line_id, line in enumerate(lines):
            data = json.loads(line)
            text, label_list, feature = data['text'], data['label'], data['feature']
            label_list = [int(i) for i in label_list]
            build_bert_inputs(train_inputs, text, label_list, feature, tokenizer)

    with open(args.dev_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line_id, line in enumerate(lines):
            data = json.loads(line)
            text, label_list, feature = data['text'], data['label'], data['feature']
            label_list = [int(i) for i in label_list]
            build_bert_inputs(dev_inputs, text, label_list, feature, tokenizer)

    train_cache_pkl_path = os.path.join(args.data_cache_path, 'train.pkl')
    dev_cache_pkl_path = os.path.join(args.data_cache_path, 'dev.pkl')

    save_pickle(train_inputs, train_cache_pkl_path)
    save_pickle(dev_inputs, dev_cache_pkl_path)


def train(args):
    tokenizer, model = build_model_and_tokenizer(args)

    if not os.path.exists(os.path.join(args.data_cache_path, 'train.pkl')):
        read_data(args, tokenizer)

    train_dataloader, dev_dataloader = load_data(args, tokenizer)

    total_steps = args.num_epochs * len(train_dataloader)

    optimizer, scheduler = build_optimizer(args, model, total_steps)

    global_steps, total_loss, cur_avg_loss, best_acc = 0, 0., 0., 0.

    print("\n >> Start training ... ... ")
    for epoch in range(1, args.num_epochs + 1):

        train_iterator = tqdm(train_dataloader, desc=f'Epoch : {epoch}', total=len(train_dataloader))

        model.train()

        for batch in train_iterator:

            model.zero_grad()

            batch_cuda = batch2cuda(args, batch)
            loss = model(**batch_cuda)[0]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if args.use_fgm:
                model.zero_grad()
                fgm = FGM(args, model)
                fgm.attack()
                adv_loss = model(**batch_cuda)[0]
                adv_loss.backward()
                fgm.restore()

            if args.use_pgd:
                model.zero_grad()
                pgd = PGD(args, model)
                pgd.backup_grad()
                for t in range(args.adv_k):
                    pgd.attack(is_first_attack=(t == 0))
                    if t != args.adv_k - 1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                    adv_loss = model(**batch_cuda)[0]
                    adv_loss.backward()
                pgd.restore()

            total_loss += loss.item()
            cur_avg_loss += loss.item()

            optimizer.step()
            scheduler.step()

            if args.use_ema:
                if args.ema_start:
                    ema.update()

            optimizer.zero_grad()

            if (global_steps + 1) % args.logging_steps == 0:

                epoch_avg_loss = cur_avg_loss / args.logging_steps
                global_avg_loss = total_loss / (global_steps + 1)

                print(f"\n>> epoch - {epoch},  global steps - {global_steps + 1}, "
                      f"epoch avg loss - {epoch_avg_loss:.4f}, global avg loss - {global_avg_loss:.4f}.")

                if args.use_ema:
                    if global_steps >= args.ema_start_step and not args.ema_start:
                        print('\n>>> EMA starting ...')
                        args.ema_start = True
                        ema = EMA(model.module if hasattr(model, 'module') else model, decay=0.95)

                if args.do_eval:

                    model.eval()

                    if args.use_ema:
                        if args.ema_start:
                            ema.apply_shadow()

                    print("\n >> Start evaluating ... ... ")

                    metric = evaluation(args, dev_dataloader, model)

                    acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10, acc11, acc12, acc13, avg_val_loss = \
                        metric['acc1'], metric['acc2'], metric['acc3'], metric['acc4'], metric['acc5'], \
                        metric['acc6'], metric['acc7'], metric['acc8'], metric['acc9'], metric['acc10'], \
                        metric['acc11'], metric['acc12'], metric['acc13'], metric['avg_val_loss']

                    all_acc = acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7 + \
                              acc8 + acc9 + acc10 + acc11 + acc12 + acc13
                    all_acc = round(all_acc, 4)

                    if all_acc > best_acc:
                        best_acc = all_acc
                        save_model(model, tokenizer, args.output_path)

                        print(f"\n >> Best saved, acc is {all_acc} !")

                    if args.use_ema:
                        if args.ema_start:
                            ema.restore()

                    model.train()
                    cur_avg_loss = 0.

            global_steps += 1
            train_iterator.set_postfix_str(f'running training loss: {loss.item():.4f}')

    if args.use_ema:
        ema.apply_shadow()

    if not args.do_eval:
        save_model(args, model, tokenizer)

    data = time.asctime(time.localtime(time.time())).split(' ')
    now_time = data[-1] + '-' + data[-5] + '-' + data[-3] + '-' + \
               data[-2].split(':')[0] + '-' + data[-2].split(':')[1] + '-' + data[-2].split(':')[2]
    os.makedirs(os.path.join(args.output_path, f'acc-{best_acc}-{now_time}'), exist_ok=True)

    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()

    print('\n >> Finish training .')


def main():
    parser = ArgumentParser()

    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--output_path', type=str,
                        default=f'../user_data/output_model')
    parser.add_argument('--train_path', type=str,
                        default=f'../raw_data/train_fine_sample.json')
    parser.add_argument('--dev_path', type=str,
                        default=f'../raw_data/dev_fine_sample.json')
    parser.add_argument('--data_cache_path', type=str,
                        default=f'../user_data/process_data/pkl')

    parser.add_argument('--model_name_or_path', type=str,
                        default=f'../user_data/pretrain_model/bert-base-chinese')

    parser.add_argument('--do_lower_case', type=bool, default=True)
    parser.add_argument('--do_eval', type=bool, default=True)

    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_seq_len', type=int, default=32)

    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--other_learning_rate', type=float, default=1e-4)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)

    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    parser.add_argument('--use_fgm', type=bool, default=False)
    parser.add_argument('--use_pgd', type=bool, default=False)
    parser.add_argument('--use_ema', type=bool, default=False)
    parser.add_argument('--use_lookahead', type=bool, default=True)

    parser.add_argument('--adv_k', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--emb_name', type=str, default='word_embeddings.')

    parser.add_argument('--ema_start', type=bool, default=False)
    parser.add_argument('--ema_start_step', type=int, default=25 * 8)

    parser.add_argument('--logging_steps', type=int, default=25)

    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--device', type=str, default='cuda')

    warnings.filterwarnings('ignore')
    args = parser.parse_args()

    path_list = [args.output_path, args.data_cache_path]
    for i in path_list:
        os.makedirs(i, exist_ok=True)

    seed_everything(args.seed)

    train(args)


if __name__ == '__main__':
    main()
