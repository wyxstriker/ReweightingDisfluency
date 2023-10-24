import logging
from modeling import *
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
import torch.nn.functional as F


class JudgeTrainer():
    def __init__(self, args) -> None:
        # torch
        self.device, self.n_gpu = self._set_device(args.local_rank, args.no_cuda)
        logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(self.device, self.n_gpu, bool(args.local_rank != -1), args.fp16))
        os.makedirs(args.output_dir, exist_ok=True)
        self._set_seed(args.seed, self.n_gpu)
        # data
        self.processor = DisfluencyProcessor()
        self.label_list = self.processor.get_labels_judge()
        self.num_labels = len(self.label_list)
        self.label_disf_list = self.processor.get_labels_disf()
        self.num_labels_tagging = len(self.label_disf_list)
        self.label_sing_list = self.processor.get_sing_labels()
        self.num_sing_labels = len(self.label_sing_list)
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
        self.writer = SummaryWriter(args.log_dir)
    
    @staticmethod
    def warmup_linear(x, warmup=0.002):
        if x < warmup:
            return x / warmup
        return 1.0 - x

    
    def train(self, args):
        train_examples = self.processor.get_examples(args.data_dir, 'train.tsv')
        num_train_steps = int(len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
        train_features = convert_examples_to_features_judge(train_examples, self.label_sing_list, self.label_disf_list, self.label_sing_list, args.max_seq_length, self.tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)
        all_label_disf_ids = torch.tensor([f.label_disf_id for f in train_features], dtype=torch.long)
        all_label_sing_ids = torch.tensor([f.label_sing_id for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_label_disf_ids, all_label_sing_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
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
            total = 0
            correct = 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, label_disf_ids, label_sing_ids = batch

                logits_pair, logits_tagging, logits_sing = self.model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
                
                loss = self.loss_fct(logits_sing.view(-1, 2), label_sing_ids.view(-1))

                if self.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                
                
                loss.backward()

                _, pretrained = torch.max(logits_sing.data, -1)
                total = pretrained.size(0)
                correct = (pretrained == label_sing_ids).sum().item()

                self.writer.add_scalar('batch_acc', correct/total, global_step)

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * self.warmup_linear(global_step / num_train_steps, args.warmup_proportion)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1
            
            logger.info("  epoch loss = %f", tr_loss/nb_tr_steps)
            self.writer.add_scalar('epoch_loss', tr_loss/nb_tr_steps, epoch_size)

            if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
                eval_examples = self.processor.get_examples(args.data_dir, 'dev.tsv')
                eval_features = convert_examples_to_features_judge(eval_examples, self.label_sing_list, self.label_disf_list, self.label_sing_list, args.max_seq_length, self.tokenizer)
                logger.info("***** Running evaluation on dev of epoch %d *****", epoch_size)
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.train_batch_size)
                all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
                all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
                all_label_disf_ids = torch.tensor([f.label_disf_id for f in eval_features], dtype=torch.long)
                all_label_sing_ids = torch.tensor([f.label_sing_id for f in eval_features], dtype=torch.long)

                eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                          all_label_disf_ids, all_label_sing_ids)
                # Run prediction for full data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.train_batch_size)

                self.model.eval()

                total = 0
                correct = 0

                for input_ids, input_mask, segment_ids, label_ids, label_disf_ids, label_sing_ids in tqdm(eval_dataloader,
                                                                                          desc="Evaluating"):
                    input_ids = input_ids.to(self.device)
                    input_mask = input_mask.to(self.device)
                    segment_ids = segment_ids.to(self.device)
                    label_ids = label_ids.to(self.device)
                    label_disf_ids = label_disf_ids.to(self.device)
                    label_sing_ids = label_sing_ids.to(self.device)

                    with torch.no_grad():
                        logits_pair, logits_tagging, logits_sing = self.model(input_ids=input_ids,
                                              token_type_ids=segment_ids,
                                              attention_mask=input_mask)
                        
                        _, pretrained = torch.max(logits_sing.data, -1)
                        total += pretrained.size(0)
                        correct += (pretrained == label_sing_ids).sum().item()

                
                logger.info("***** Eval results *****")
                acc = correct/total
                logger.info("  %s = %f", 'acc', acc)
                result = {'acc': acc}
                self.writer.add_scalar('dev_acc', acc, epoch_size)
                
                logger.info('***** prev best results %f*****', prev_best_dev_f1)
                logger.info('***** now f results %f*****', acc)
                if acc > prev_best_dev_f1:
                    model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model it-self
                    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")

                    output_eval_file = os.path.join(args.output_dir, "best_dev.epoch" + str(epoch_size))
                    with open(output_eval_file, "w") as writer:
                        writer.write("best")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    prev_best_dev_f1 = acc
                    logger.info('save the model, %s', output_model_file)

                output_eval_file = os.path.join(args.output_dir, "dev_eval_results.txt.epoch" + str(epoch_size))
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Dev Eval results %d*****", epoch_size)
                    for key in sorted(result.keys()):
                        logger.info("  %s = %s", key, str(result[key]))
                        writer.write("%s = %s\n" % (key, str(result[key])))


            if args.do_test and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
                eval_examples = self.processor.get_examples(args.data_dir, 'test.tsv')
                eval_features = convert_examples_to_features_judge(eval_examples, self.label_sing_list, self.label_disf_list, self.label_sing_list, args.max_seq_length, self.tokenizer)
                logger.info("***** Running evaluation on test %d*****", epoch_size)
                logger.info("  Test Num examples = %d", len(eval_examples))
                logger.info("  Test Batch size = %d", args.train_batch_size)
                all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
                all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
                all_label_disf_ids = torch.tensor([f.label_disf_id for f in eval_features], dtype=torch.long)
                all_label_sing_ids = torch.tensor([f.label_sing_id for f in eval_features], dtype=torch.long)

                eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                          all_label_disf_ids, all_label_sing_ids)
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
                total = 0
                correct = 0
                for input_ids, input_mask, segment_ids, label_ids, label_disf_ids, label_sing_ids in tqdm(eval_dataloader,
                                                                                          desc="Evaluating"):
                    input_ids = input_ids.to(self.device)
                    input_mask = input_mask.to(self.device)
                    segment_ids = segment_ids.to(self.device)
                    label_ids = label_ids.to(self.device)
                    label_disf_ids = label_disf_ids.to(self.device)
                    label_sing_ids = label_sing_ids.to(self.device)

                    with torch.no_grad():
                        logits_pair, logits_tagging, logits_sing = self.model(input_ids=input_ids, token_type_ids=segment_ids
                                                                 , attention_mask=input_mask)
                        _, pretrained = torch.max(logits_sing.data, -1)
                        total += pretrained.size(0)
                        correct += (pretrained == label_sing_ids).sum().item()


                logger.info("***** test results *****")
                acc = correct/total
                logger.info("  %s = %f", 'acc', acc)
                result = {'acc': acc}
                self.writer.add_scalar('test_acc', acc, epoch_size)
                
                output_eval_file = os.path.join(args.output_dir, "test_eval_results.txt.epoch" + str(epoch_size))
                if acc > prev_best_test_f1:
                    prev_best_test_f1 = acc
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Test Eval results epoch%d*****", epoch_size)
                    for key in sorted(result.keys()):
                        logger.info("  %s = %s", key, str(result[key]))
                        writer.write("%s = %s\n" % (key, str(result[key])))
        
        self.writer.close()

        



    def judge(self, args):
        eval_examples = self.processor.get_pair_examples(args.data_dir, 'pseudo.tsv')
        eval_origin_examples = self.processor.get_false_examples(args.data_dir, 'pseudo.tsv')
        eval_features = convert_examples_to_features_judge(eval_examples, self.label_sing_list, self.label_disf_list, self.label_sing_list, args.max_seq_length, self.tokenizer)
        # gpt_features = convert_examples_to_gpt_features(eval_examples, args.max_seq_length, tokenizerGPT2)
        logger.info("***** Running evaluation on dev of epoch *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_label_disf_ids = torch.tensor([f.label_disf_id for f in eval_features], dtype=torch.long)
        all_label_sing_ids = torch.tensor([f.label_sing_id for f in eval_features], dtype=torch.long)

        # all_gpt_input_ids = torch.tensor([f.input_ids for f in gpt_features], dtype=torch.long)
        # all_gpt_input_mask = torch.tensor([f.input_mask for f in gpt_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                  all_label_disf_ids, all_label_sing_ids)
        # gpt_data = TensorDataset(all_gpt_input_ids, all_gpt_input_mask)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.train_batch_size)

        # gpt_sampler = SequentialSampler(gpt_data)
        # gpt_dataloader = DataLoader(gpt_data, sampler=gpt_sampler, batch_size=args.train_batch_size)

        self.model.eval()

        logits_total = list()
        input_ids_total = list()

        for input_ids, input_mask, segment_ids, label_ids, label_disf_ids, label_sing_ids in tqdm(eval_dataloader,
                                                                                                  desc="Evaluating"):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            label_ids = label_ids.to(self.device)
            label_disf_ids = label_disf_ids.to(self.device)
            label_sing_ids = label_sing_ids.to(self.device)

            with torch.no_grad():
                # tmp_eval_loss = model(input_ids=input_ids,
                #                       token_type_ids=segment_ids,
                #                       attention_mask=input_mask,
                #                       labels_tagging=label_disf_ids)
                # &&&& 增加了sing的logits
                logits_pair, logits_tagging, logits_sing = self.model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

            # &&&&
            logits_sing = F.softmax(logits_sing, dim=-1)
            logits_sing = logits_sing.detach().cpu().numpy()
            logits_total.extend(logits_sing)
            input_ids_total.extend(input_ids.detach().cpu().numpy())

        # with open(os.path.join(args.data_dir, "train.tsv"), 'w', encoding='utf8') as fw:
        #     select_id = -1
        #     select_score = -1
        #     select_content = ''
        #     for i in range(len(logits_total)):
        #         # if logits_total[i][1] > args.thre:
        #         # fw.write(str(eval_origin_examples[i].guid)+'\t'+eval_origin_examples[i].text_a.strip()+'\tNONE\tNONE\t'+eval_origin_examples[i].disf_label.strip()+'\t'+str(logits_total[i][1].item())+"\n")
        #         now_id = eval_origin_examples[i].guid
        #         now_score = logits_total[i][1].item()
        #         if now_id != select_id:
        #             if select_content != '':
        #                 fw.write(select_content+'\n')
        #             select_id = now_id
        #             select_score = now_score
        #             select_content = (str(eval_origin_examples[i].guid)+'\t'+eval_origin_examples[i].text_a.strip()+'\tNONE\tNONE\t'+eval_origin_examples[i].disf_label.strip()+'\t'+str(logits_total[i][1].item()))
        #         elif now_score > select_score:
        #             select_score = now_score
        #             select_content = (str(eval_origin_examples[i].guid)+'\t'+eval_origin_examples[i].text_a.strip()+'\tNONE\tNONE\t'+eval_origin_examples[i].disf_label.strip()+'\t'+str(logits_total[i][1].item()))
        #     if select_content != '':
        #         fw.write(select_content+'\n')

        with open(os.path.join(args.data_dir, "train.tsv"), 'w', encoding='utf8') as fw:
            for i in range(len(logits_total)):
                if logits_total[i][1] > args.thre:
                # fw.write(t[i].strip()+'\t'+str(logits_total[i][1])+'\t'+str(gpt_total[i])+'\t'+str(gpt_n[i].item())+"\n")
                    fw.write(str(eval_origin_examples[i].guid)+'\t'+eval_origin_examples[i].text_a.strip()+'\tNONE\tNONE\t'+eval_origin_examples[i].disf_label.strip()+'\t'+str(logits_total[i][1].item())+"\n")
                        
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
                finetuning_task=task_name,
            )
            if "state_dict" in state:
                state = state['state_dict']
            model = ElectraForSequenceDisfluency_sing.from_pretrained(
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
            model = ElectraForSequenceDisfluency_sing.from_pretrained(
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
    # parser.add_argument('--log_dir', required=True, type=str)
    # parser.add_argument("--bert_model", default=None, type=str, required=True)
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
    parser.add_argument("--do_tagging", action='store_true')
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
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--seed', type=int, default=226)
    parser.add_argument('--unlabel_size', type=int, default=500)
    
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--loss_scale', type=float, default=0)

    parser.add_argument("--model_name_or_path", default=None, type=str, required=True)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--thre", default=0.5, type=float)

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    judge = JudgeTrainer(args)

    if args.do_train:
        judge.train(args)
    else:
        judge.judge(args)
