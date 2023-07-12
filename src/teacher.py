import logging

import math
from torch.nn import parameter
from data_producer import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import argparse
import torch, random
import numpy as np
import logging
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
import os
import torch.nn.functional as F


class TeacherTrainer():
    def __init__(self, args) -> None:
        # torch
        self.device, self.n_gpu = self._set_device(args.local_rank, args.no_cuda)
        logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(self.device, self.n_gpu, bool(args.local_rank != -1), args.fp16))
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        self._set_seed(args.seed, self.n_gpu)
        # data
        self.processor = DisfluencyProcessor()
        self.label_list = self.processor.get_labels()
        self.num_labels = len(self.label_list)
        self.label_disf_list = self.processor.get_labels_disf()
        self.num_labels_tagging = len(self.label_disf_list)
        # model
        pretrained = str(args.model_name_or_path)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained, do_lower_case=args.do_lower_case)
        self.model = self._get_model(args.use_new_model, pretrained, self.num_labels, self.num_labels_tagging, args.task_name).to(self.device)
        # train_set
        if args.do_train:
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
        
        # self.writer_local = SummaryWriter(os.path.join(args.log_dir, str(args.unlabel_size)))
        self.writer_total = SummaryWriter(args.log_dir)
    
    @staticmethod
    def warmup_linear(x, warmup=0.002):
        if x < warmup:
            return x / warmup
        return 1.0 - x

    
    def train(self, args):
        train_examples = self.processor.get_origin_examples(args.data_dir, 'train.tsv')
        soft_examples = self.processor.get_origin_examples(args.data_dir, 'soft.tsv')
        num_train_steps = int(len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
        train_features = convert_examples_to_features(train_examples, self.label_list, self.label_disf_list, args.max_seq_length, self.tokenizer)
        soft_features = convert_examples_to_features_soft(soft_examples, self.label_list, self.label_disf_list, args.max_seq_length, self.tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)
        all_label_disf_ids = torch.tensor([f.label_disf_id for f in train_features], dtype=torch.long)
        all_label_soft_ids = torch.tensor([f.label_disf_id for f in soft_features], dtype=torch.float)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_label_disf_ids, all_label_soft_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        # training
        global_step = 0
        epoch_size = 0
        prev_best_dev_f1 = -1.0
        prev_best_test_f1 = -1.0
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            logger.info("***** Running Training on TrainSet of epoch %d *****", epoch_size)
            logger.info("  Num examples = %d", len(train_examples))
            logger.info("  Batch size = %d", args.train_batch_size)
            self.model.train()
            epoch_size += 1
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, label_disf_ids, label_soft_ids = batch
                _, logits_tagging = self.model(input_ids=input_ids, attention_mask=input_mask)
                if args.judge_score:
                    judge_score = torch.zeros(logits_tagging.size(), requires_grad=False).to(self.device)
                    for s in range(len(label_ids)):
                        judge_score[s] = label_ids[s]
                    logits_tagging = logits_tagging * judge_score
                
                elif args.judge_score_e:
                    judge_score = torch.zeros(logits_tagging.size(), requires_grad=False).to(self.device)
                    for s in range(len(label_ids)):
                        judge_score[s] = label_ids[s]
                    logits_tagging = logits_tagging * torch.exp(judge_score)

                elif args.teacher_score:
                    judge_score = torch.zeros(logits_tagging.size(), requires_grad=False).to(self.device)
                    for s in range(len(label_ids)):
                        judge_score[s] = label_ids[s]
                    # logits_tagging = logits_tagging * torch.exp(judge_score)

                    teacher_score = torch.zeros(logits_tagging.size(), requires_grad=False).to(self.device)
                    for s in range(label_soft_ids.size(0)):
                        for d in range(label_soft_ids.size(1)):
                            teacher_score[s][d] = label_soft_ids[s][d]
                    # logits_tagging = logits_tagging * (teacher_score)

                    logits_tagging = logits_tagging * (judge_score  + teacher_score) / 2

                
                
                loss = self.loss_fct(logits_tagging.view(-1, self.num_labels_tagging), label_disf_ids.view(-1))

                if self.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                # self.writer_local.add_scalar('batch_loss', loss.item(), global_step)
                
                loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * self.warmup_linear(global_step / num_train_steps, args.warmup_proportion)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1
            logger.info("  epoch loss = %f", tr_loss/nb_tr_steps)
            # self.writer_local.add_scalar('epoch_loss', tr_loss/nb_tr_steps, epoch_size)

            if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
                eval_examples = self.processor.get_origin_examples(args.data_dir, 'dev.tsv')
                eval_features = convert_examples_to_features(eval_examples, self.label_list, self.label_disf_list, args.max_seq_length, self.tokenizer)
                logger.info("***** Running evaluation on dev of epoch %d *****", epoch_size)
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.train_batch_size)
                all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
                all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
                all_label_disf_ids = torch.tensor([f.label_disf_id for f in eval_features], dtype=torch.long)
                eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                          all_label_disf_ids)
                # Run prediction for full data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.train_batch_size)

                self.model.eval()
                eval_loss, eval_accuracy = 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0

                # predict_result_pair = []
                predict_result_tagging = []
                # gold_result_pair = []
                gold_result_tagging = []
                input_mask_tagging = []

                for input_ids, input_mask, segment_ids, label_ids, label_disf_ids in tqdm(eval_dataloader,
                                                                                          desc="Evaluating"):
                    input_ids = input_ids.to(self.device)
                    input_mask = input_mask.to(self.device)
                    segment_ids = segment_ids.to(self.device)
                    label_ids = label_ids.to(self.device)
                    label_disf_ids = label_disf_ids.to(self.device)

                    with torch.no_grad():
                        tmp_eval_loss = self.model(input_ids=input_ids,
                                              token_type_ids=segment_ids,
                                              attention_mask=input_mask,
                                              labels_tagging=label_disf_ids)
                        logits_pair, logits_tagging = self.model(input_ids=input_ids, token_type_ids=segment_ids,
                                                            attention_mask=input_mask)

                    logits_pair = logits_pair.detach().cpu().numpy()
                    logits_tagging = logits_tagging.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    label_disf_ids = label_disf_ids.to('cpu').numpy()
                    input_mask = input_mask.to('cpu').numpy()

                    predict_result_tagging.append(np.argmax(logits_tagging, axis=-1).tolist())
                    gold_result_tagging.append(label_disf_ids.tolist())
                    input_mask_tagging.append(input_mask.tolist())

                    eval_loss += tmp_eval_loss.mean().item()
                    nb_eval_examples += input_ids.size(0)
                    nb_eval_steps += 1
                eval_loss = eval_loss / nb_eval_steps

                p_score, r_score, f_score = accuracy_tagging(eval_examples, predict_result_tagging,
                                                                 gold_result_tagging, input_mask_tagging,
                                                                 os.path.join(args.output_dir,
                                                                              "dev_results.txt.epoch" + str(
                                                                                  epoch_size)))
                result = {'eval_loss': eval_loss,
                              'dev p_score': p_score,
                              'dev r_score': r_score,
                              'dev f_score': f_score}
                logger.info('***** prev best results %f*****', prev_best_dev_f1)
                logger.info('***** now f results %f*****', f_score)

                if f_score > prev_best_dev_f1:
                    
                    model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model it-self
                    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")

                    output_eval_file = os.path.join(args.output_dir, "best_dev.epoch" + str(epoch_size))
                    with open(output_eval_file, "w") as writer:
                        writer.write("best")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    prev_best_dev_f1 = f_score
                    logger.info('save the model, %s', output_model_file)

                output_eval_file = os.path.join(args.output_dir, "dev_eval_results.txt.epoch" + str(epoch_size))
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Dev Eval results %d*****", epoch_size)
                    for key in sorted(result.keys()):
                        logger.info("  %s = %s", key, str(result[key]))
                        writer.write("%s = %s\n" % (key, str(result[key])))


            if args.do_test and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
                eval_examples = self.processor.get_origin_examples(args.data_dir, 'test.tsv')
                eval_features = convert_examples_to_features(eval_examples, self.label_list, self.label_disf_list, args.max_seq_length, self.tokenizer)
                logger.info("***** Running evaluation on test %d*****", epoch_size)
                logger.info("  Test Num examples = %d", len(eval_examples))
                logger.info("  Test Batch size = %d", args.train_batch_size)
                all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
                all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
                all_label_disf_ids = torch.tensor([f.label_disf_id for f in eval_features], dtype=torch.long)
                eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                          all_label_disf_ids)
                # Run prediction for full data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.train_batch_size)

                self.model.eval()
                eval_loss, eval_accuracy = 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0

                # predict_result_pair = []
                predict_result_tagging = []
                # gold_result_pair = []
                gold_result_tagging = []
                input_mask_tagging = []

                for input_ids, input_mask, segment_ids, label_ids, label_disf_ids in tqdm(eval_dataloader,
                                                                                          desc="Evaluating"):
                    input_ids = input_ids.to(self.device)
                    input_mask = input_mask.to(self.device)
                    segment_ids = segment_ids.to(self.device)
                    label_ids = label_ids.to(self.device)
                    label_disf_ids = label_disf_ids.to(self.device)

                    with torch.no_grad():
                        tmp_eval_loss = self.model(input_ids=input_ids,
                                              token_type_ids=segment_ids,
                                              attention_mask=input_mask,
                                              labels_tagging=label_disf_ids)
                        logits_pair, logits_tagging = self.model(input_ids=input_ids, token_type_ids=segment_ids,
                                                            attention_mask=input_mask)

                    logits_pair = logits_pair.detach().cpu().numpy()
                    logits_tagging = logits_tagging.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    label_disf_ids = label_disf_ids.to('cpu').numpy()
                    input_mask = input_mask.to('cpu').numpy()


                    predict_result_tagging.append(np.argmax(logits_tagging, axis=-1).tolist())
                    gold_result_tagging.append(label_disf_ids.tolist())
                    input_mask_tagging.append(input_mask.tolist())

                    eval_loss += tmp_eval_loss.mean().item()
                    nb_eval_examples += input_ids.size(0)
                    nb_eval_steps += 1
                eval_loss = eval_loss / nb_eval_steps

                p_score, r_score, f_score = accuracy_tagging(eval_examples, predict_result_tagging,
                                                                 gold_result_tagging, input_mask_tagging,
                                                                 os.path.join(args.output_dir,
                                                                              "test_results.txt.epoch" + str(
                                                                                  epoch_size)))
                result = {'test_loss': eval_loss,
                              'test p_score': p_score,
                              'test r_score': r_score,
                              'test f_score': f_score}
                output_eval_file = os.path.join(args.output_dir, "test_eval_results.txt.epoch" + str(epoch_size))
                if f_score > prev_best_test_f1:
                    prev_best_test_f1 = f_score
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Test Eval results epoch%d*****", epoch_size)
                    for key in sorted(result.keys()):
                        logger.info("  %s = %s", key, str(result[key]))
                        writer.write("%s = %s\n" % (key, str(result[key])))
        
        self.writer_total.add_scalar('dev_f1', prev_best_dev_f1, args.unlabel_size)
        self.writer_total.add_scalar('test_f1', prev_best_test_f1, args.unlabel_size)
        # self.writer_local.close()
        self.writer_total.close()
        



    def tag(self, args):
        unlabel_examples = self.processor.get_origin_examples(args.data_dir, 'unlabel.tsv')
        random.shuffle(unlabel_examples)
        unlabel_examples = unlabel_examples[:args.unlabel_size]
        with open(os.path.join(args.output_dir, "unlabel_gold.txt"), 'w') as f:
            for index, line in enumerate(unlabel_examples):
                f.write(str(
                    index) + "\t" + line.text_a + "\t" + line.text_b + "\t" + str(line.label) + "\t" + line.disf_label + "\n")
        unlabel_features = convert_examples_to_features_unlabel(
            unlabel_examples, args.max_seq_length, self.tokenizer, args.sel_prob, "unlabel")
        logger.info("***** Running evaluation on dev of epoch *****")
        logger.info("  Num examples = %d", len(unlabel_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in unlabel_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in unlabel_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in unlabel_features], dtype=torch.long)
        all_label_disf_ids = torch.tensor([f.label_disf_id for f in unlabel_features],
                                          dtype=torch.long)
        unlabel_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                     all_label_disf_ids)
        unlabel_sampler = SequentialSampler(unlabel_data)
        unlabel_dataloader = DataLoader(unlabel_data, sampler=unlabel_sampler,
                                        batch_size=args.train_batch_size)

        self.model.eval()

        predict_result_tagging = []
        predict_result_value = []
        gold_result_tagging = []
        input_mask_tagging = []

        for input_ids, input_mask, segment_ids, label_disf_ids in tqdm(unlabel_dataloader,
                                                                       desc="Evaluating"):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            label_disf_ids = label_disf_ids.to(self.device)

            with torch.no_grad():
                logits_pair, logits_tagging = self.model(input_ids=input_ids, token_type_ids=segment_ids,
                                                    attention_mask=input_mask)

            logits_tagging = logits_tagging.detach().cpu().numpy()
            input_mask = input_mask.to('cpu').numpy()
            label_disf_ids = label_disf_ids.to('cpu').numpy()

            gold_result_tagging.append(label_disf_ids.tolist())
            input_mask_tagging.append(input_mask.tolist())
            if args.K == 1:
                predict_result_tagging.append(torch.unsqueeze(torch.tensor(np.argmax(logits_tagging, axis=-1)), dim=1))
                predict_result_value.append(np.max(logits_tagging, axis=-1))
            else:
                rate = args.unlabel_size / 200000
                k_thed = np.linspace(0.5-rate, 0.5+rate, args.K) * 2 - 1
                k_tagging = []
                logits_tagging = F.softmax(torch.tensor(logits_tagging), dim=-1)
                for th in k_thed:
                    temp = logits_tagging.clone()
                    # th = torch.rand(temp[..., 1].size())*2-1
                    temp[..., 1] -= th
                    k_tagging.append(np.argmax(temp, axis=-1).tolist())
                k_tagging = np.transpose(k_tagging, [1, 0, 2])
                predict_result_tagging.append(k_tagging)
        
        temp = args.temp *  - math.log(args.unlabel_size / 100000 / 2, 10)
        
        self.writer_total.add_scalar('temp', temp, args.unlabel_size)

        all_d_set = unlabel_tagging(unlabel_examples, predict_result_tagging, gold_result_tagging, input_mask_tagging, os.path.join(args.output_dir, "pseudo.tsv"))
        unlabel_soft(unlabel_examples, predict_result_value, gold_result_tagging, input_mask_tagging, os.path.join(args.output_dir, "soft.tsv"), temp, all_d_set, do_temp=False if args.no_temp else True)

    def _set_seed(self, seed, n_gpu):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)
    
    def _set_device(self, local_rank, no_cuda):
        if local_rank == -1 or no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
            n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            n_gpu = 1
            torch.distributed.init_process_group(backend='nccl')
        return device, n_gpu

    def _get_model(self, use_model, pretrained, num_labels, num_labels_tagging, task_name):
        if use_model:
            new_model_file = os.path.join(args.pretrain_model_dir, args.pretrain_model_name)
            logger.info("use pretrain model {}".format(new_model_file))
            state = torch.load(new_model_file)
            config = ElectraConfig.from_pretrained(
                pretrained,
                num_labels=num_labels,
                finetuning_task=task_name)
            if "state_dict" in state:
                state = state['state_dict']
            model = ElectraForSequenceDisfluency_real.from_pretrained(
                pretrained,
                config=config,
                state_dict=state,
                num_labels=num_labels, num_labels_tagging=num_labels_tagging)
        else:
            logger.info("train new model ")
            config = ElectraConfig.from_pretrained(
                pretrained,
                num_labels=num_labels,
                finetuning_task=task_name,
            )
            model = ElectraForSequenceDisfluency_real.from_pretrained(
                pretrained,
                config=config,
                num_labels=num_labels, num_labels_tagging=num_labels_tagging
            )
        return model



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # required
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--log_dir', required=True, type=str)
    parser.add_argument("--bert_model", default=None, type=str, required=True)
    parser.add_argument("--pretrain_model_dir", default=None, type=str, required=True)
    parser.add_argument("--pretrain_model_name", default=None, type=str, required=True)
    ## Others
    parser.add_argument("--max_seq_length", default=128, type=int)
    
    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument("--do_train", action='store_true')
    group1.add_argument("--do_unlabel", action='store_true')

    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--do_eval_format", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--no_temp", action='store_true')
    parser.add_argument("--use_new_model", action='store_true')
    parser.add_argument("--do_lower_case", action='store_true')
    
    parser.add_argument("--task_name", default=None, type=str, required=True)
    parser.add_argument("--train_batch_size", default=64, type=int)
    parser.add_argument("--eval_batch_size", default=128, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--sel_prob", default=0.5, type=float)
    parser.add_argument("--num_train_epochs", default=5, type=int)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("--judge_score", action='store_true')
    parser.add_argument("--judge_score_e", action='store_true')
    parser.add_argument("--teacher_score", action='store_true')
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--seed', type=int, default=226)
    parser.add_argument('--unlabel_size', type=int, default=500)
    
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--loss_scale', type=float, default=0)

    parser.add_argument("--model_name_or_path", default=None, type=str, required=True)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("-K", default=1, type=int)
    parser.add_argument("--k_thred", default=0.5, type=float)
    parser.add_argument("--temp", default=1, type=float)

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    teacher = TeacherTrainer(args)
    if args.do_train:
        teacher.train(args)
    else:
        teacher.tag(args)
