from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import gc
import math
import random
import json
#import faiss
import pickle
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler,TensorDataset)
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
# from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter
from model import RetrievalModel,TeacherModel,MVD
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from utils.data_utils import master_process,convert_examples_to_features, processors,output_modes
from utils.sampler import MultiDomainSampler
from eval import to_eval
from collections import OrderedDict
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()

    # Task parameters
    parser.add_argument("--kb", default="zeshel",type=str,choices= ["zeshel","wikipedia"], 
                        required=True,help="Knowledge base.")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train.")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input train/vaild/test data dir. Should contain the .jsonl files (or other data files) for the task.")
    parser.add_argument("--entity_data_dir", default=None, type=str, required=True,
                        help="The input entity data dir. Should contain the .jsonl files (or other data files) for the task.")
    parser.add_argument("--output_dir", default="./output", type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cache_dir",default="cache",
                        type=str,help="Where do you want to store embeddings")
    parser.add_argument("--do_train",action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",action='store_true',
                        help="Whether to run eval on the dev/test set.")
    parser.add_argument("--do_predict",action='store_true',
                        help="Whether to do inference.")
    parser.add_argument("--predict_file", default="train",type=str,
                        choices= ["train","valid","test"], help="File for inference.")
    # Model parameters
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--pretrain_retriever", default=None, type=str,
                        help="Pre-trained dual-encoder retriever. ")
    parser.add_argument("--pretrain_teacher", default=None, type=str,
                        help="Pre-trained cross-encoder teacher. ")
    # Data parameters
    parser.add_argument("--max_seq_length",default=128,type=int,
                        help="The maximum total input mention.")
    parser.add_argument("--cand_num",default=16,type=int,
                        help="The num of candidate entity per sample.")
    parser.add_argument("--infer_view_type",default="global-local",type=str,
                        choices= ["local","global","global-local"],help="global-local entity representations.")
    parser.add_argument("--train_view_type",default="local",type=str,
                        choices= ["local","global"],help="global-local entity representations during training.")
    parser.add_argument("--max_ent_length",default=128,type=int,
                        help="The maximum total input entity length.")
    parser.add_argument("--max_view_length",default=40,type=int,
                        help="The maximum length of each view within entity.")
    parser.add_argument("--max_view_num",default=10,type=int,
                        help="The number of view within entity for training.")
    # Train parameters
    parser.add_argument("--num_train_epochs",default=3.0,type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--train_batch_size",default=32,type=int,
                        help="Total batch size for training.")
    parser.add_argument('--gradient_accumulation_steps',type=int,default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate",default=5e-5,type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--warmup_proportion",default=0.1,type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                        "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--weight_decay',type=float,default=0.01,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    # Eval parameters
    parser.add_argument("--eval_per_epoch", default=10, type=float,
                        help="How many times it evaluates on dev set per epoch")
    parser.add_argument("--eval_batch_size",default=8,type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--top_k",default=100,type=int,
                        help="Top_k candiates entity retrieval")
    parser.add_argument("--faiss",action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--thread_num",default=8,type=int,
                        help="faiss thread_num.")
    parser.add_argument("--best_checkpoint_metric", default="top64_hits", type=str,
                        help="Metric used to choose the best checkpoint. ")
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=10000,
                        help="random seed for initialization")
    parser.add_argument("--load_cache",action='store_true',
                        help="Whether to load entity embeddings from cache.")
    parser.add_argument("--do_lower_case",action='store_true',
                        help="Set this flag if you are using an uncased model.")
    
    parser.add_argument("--no_cuda",action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",type=int,default=-1,
                        help="local_rank for distributed training on gpus")
    
    parser.add_argument('--fp16',action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--server_ip', type=str, default='',
                        help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='',
                        help="Can be used for distant debugging.")
    args = parser.parse_args()

    args.output_dir = args.output_dir.replace(
        '[PT_OUTPUT_DIR]', os.getenv('PT_OUTPUT_DIR', ''))
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(
        args.output_dir, 'opt.json'), 'w'), sort_keys=True, indent=2)

    return args

def load_model(args):
    men_bert = BertModel.from_pretrained(args.bert_model)
    ent_bert = BertModel.from_pretrained(args.bert_model)
    tech_bert = BertModel.from_pretrained(args.bert_model)
    retriever = RetrievalModel(men_bert,ent_bert)
    teacher = TeacherModel(tech_bert)
    if args.pretrain_retriever:
        retriever.load_state_dict(torch.load(args.pretrain_retriever,map_location='cpu'),strict=False)
    if args.pretrain_teacher:
        teacher.load_state_dict(torch.load(args.pretrain_teacher,map_location='cpu'),strict=False)
    if args.task_name == 'mvd':
        model = MVD(retriever=retriever,teacher=teacher)
        config = retriever.mention_encoder.config
    if args.task_name == 'retriever':
        model = retriever
        config = retriever.mention_encoder.config
    elif args.task_name == 'teacher':
        model = teacher
        config = teacher.rank_encoder.config
    return model,config

def main():
    args = get_args()
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        n_gpu = 1
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    is_roberta = "roberta" in args.bert_model

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if args.seed > 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    else:
        random_seed = random.randint(0, 10000)
        logger.info("Set random seed as: {}".format(random_seed))
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    if not args.do_train and not args.do_eval and not args.do_predict:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #   raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if master_process(args): 
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        if not os.path.exists(args.cache_dir):
            os.makedirs(args.cache_dir)

    task_name = args.task_name.lower()
    cand_num = args.cand_num
    view_type = args.train_view_type
    kb = args.task_nakbme.lower()

    if task_name not in output_modes:
        raise ValueError("Task not found: %s" % task_name)

    processor = processors[kb](task_name,cand_num,view_type)
    output_mode = output_modes[task_name]
    if master_process(args): 
        logger.info("***** Running Config *****")
        logger.info("  Knowledge Base = {}".format(kb))
        logger.info("  Task = {}".format(task_name))

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    train_examples = None
    num_train_optimization_steps = None

    if args.do_train:
        # entity_examples,global_examples = processor.get_entity_examples(args.entity_data_dir)
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    model,config = load_model(args)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.local_rank],output_device=args.local_rank,find_unused_parameters=True)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if args.do_train:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        weight_decay = args.weight_decay
        logger.info("Set weight_decay as %.4f" % weight_decay)
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, weight_decay=weight_decay, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=args.warmup_proportion * num_train_optimization_steps,
                                                    num_training_steps=num_train_optimization_steps)

    global_step = 0
    dev_eval_features = None
    if master_process(args):
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "log"), exist_ok=True)
        os.makedirs("log", exist_ok=True)
        #summary_writer = SummaryWriter("log")
#         summary_writer = SummaryWriter(os.path.join(args.output_dir, "log"))

    if args.do_train:
        dev_eval_examples = processor.get_dev_examples(args.data_dir)
        dev_eval_features = convert_examples_to_features(args,
            dev_eval_examples,tokenizer,
            output_mode=output_mode,
            do_train=False
        )

        #Prepare train data
        train_features = convert_examples_to_features(args,
        train_examples,tokenizer,
        output_mode=output_mode
        )

        if task_name == "retriever":
            all_mention_ids = torch.tensor(
                [f.mention_ids for f in train_features], dtype=torch.long)
            all_entity_ids = torch.from_numpy(
                np.asarray([f.entity_ids for f in train_features]).astype(np.int64))
            all_domain = [f.domain for f in train_features]
            train_data = TensorDataset(all_mention_ids, all_entity_ids)

        elif task_name == "teacher":
            all_segment_ids = torch.from_numpy(
                np.asarray([f.segment_ids for f in train_features]).astype(np.int64))
            all_label = torch.tensor(
                [f.label for f in train_features], dtype=torch.long)
            all_domain = [f.domain for f in train_features]

            input_path = os.path.join(args.cache_dir,'all_input_ids.pt')
            if master_process(args):
                all_input_ids = torch.from_numpy(
                    np.asarray([f.input_ids for f in train_features]).astype(np.int64))
                torch.save(all_input_ids, input_path)
            
            dist.barrier()

            if not master_process(args):
                all_input_ids = torch.load(input_path)
            
            dist.barrier()

            if master_process(args):
                os.remove(input_path)

            train_data = TensorDataset(all_input_ids, all_segment_ids,all_label)

        elif task_name == "mvd":
            all_mention_ids = torch.tensor(
                [f.mention_ids for f in train_features], dtype=torch.long)
            all_segment_ids = torch.from_numpy(
                np.asarray([f.segment_ids for f in train_features]).astype(np.int64))
            all_label = torch.tensor(
                [f.label for f in train_features], dtype=torch.long)
            all_guid = torch.tensor(
                [f.guid for f in train_features], dtype=torch.long)
            all_entity_idx = torch.tensor(
                [f.entity_idx for f in train_features], dtype=torch.long)
            all_mention_tokens = [f.mention_tokens for f in train_features]
            all_domain = [f.domain for f in train_features]

            entity_path = os.path.join(args.cache_dir,'all_entity_ids.pt')
            input_path = os.path.join(args.cache_dir,'all_input_ids.pt')
            if master_process(args):
                all_entity_ids = torch.from_numpy(
                    np.asarray([f.entity_ids for f in train_features]).astype(np.int64))
                all_input_ids = torch.from_numpy(
                    np.asarray([f.input_ids for f in train_features]).astype(np.int64))
                torch.save(all_entity_ids, entity_path)
                torch.save(all_input_ids, input_path)
            
            dist.barrier()

            if not master_process(args):
                all_entity_ids = torch.load(entity_path)
                all_input_ids = torch.load(input_path)
            
            dist.barrier()

            if master_process(args):
                os.remove(entity_path)
                os.remove(input_path)

            train_data = TensorDataset(all_mention_ids, all_guid)
            
        if master_process(args): 
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", len(train_examples))
            logger.info("  Batch size = %d", args.train_batch_size)
            logger.info("  Number of candidates = %d", args.cand_num if output_mode == "rank" or task_name=='mvd' else args.train_batch_size)
            logger.info("  Embedding type = {}".format(args.train_view_type))
            logger.info("  Num steps = %d", num_train_optimization_steps)


        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            if args.kb == "zeshel":
                train_sampler = MultiDomainSampler(train_data,args.train_batch_size,all_domain)
            else:
                train_sampler = DistributedSampler(train_data)

        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        eval_step = max(1, len(train_features) //
                        args.train_batch_size // args.eval_per_epoch // args.gradient_accumulation_steps)
        
        del train_features
        gc.collect()

        if args.local_rank != -1:
            eval_step = eval_step // torch.distributed.get_world_size()
        if master_process(args):
            logger.info("  Eval steps = %d", eval_step)

        model.train()
        best_perf = None
        total_steps = 0
        for _ in range(int(args.num_train_epochs)):
            train_sampler.set_epoch(_)
            tr_loss = 0
            nb_tr_steps = 0
            if master_process(args):
                train_dataloader = tqdm(train_dataloader)
            for batch in train_dataloader:
                batch = tuple(t.to(device) for t in batch)

                if task_name == "retriever":
                    mention_ids,entity_ids = batch
                    loss,_ = model(mention_ids=mention_ids,entity_ids=entity_ids)

                elif task_name == "teacher":
                    input_ids,segment_ids,label = batch
                    loss,_ = model(input_ids=input_ids,segment_ids=segment_ids,label=label)

                elif task_name == "mvd":
                    mention_ids,guid = batch
                    # Update dynamic hard negatives
                    entity_ids = all_entity_ids[guid.detach().cpu().tolist()].to(device)
                    input_ids = all_input_ids[guid.detach().cpu().tolist()].to(device)
                    segment_ids = all_segment_ids[guid.detach().cpu().tolist()].to(device)
                    label = all_label[guid.detach().cpu().tolist()].to(device)
                    loss = model(mention_ids=mention_ids, entity_ids=entity_ids,input_ids=input_ids,segment_ids=segment_ids,label=label)

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                # nb_tr_examples += mention_ids.size(0)
                nb_tr_steps += 1
                total_steps += 1

                if total_steps % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * \
                            warmup_linear(
                                global_step/num_train_optimization_steps, args.warmup_proportion)
                        if args.lr_layerwise_decay == 1:
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = lr_this_step
                        else:
                            _lr = lr_this_step
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = _lr
                                _lr *= args.lr_layerwise_decay

                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    # optimizer.zero_grad()
                    global_step += 1


                    if global_step % eval_step == 0 or global_step == num_train_optimization_steps:

                        train_mention_features = tuple()
                        if args.task_name == 'mvd':
                            train_mention_features = (all_mention_ids,all_entity_idx,all_mention_tokens,all_domain)

                        model.eval()
                        result,train_entity_features = to_eval(args,logger,
                                model,dev_eval_features, device, tokenizer,
                                output_mode,global_step,
                                train_mention_features=train_mention_features)
                        
                        if args.task_name == 'mvd':    
                            all_entity_ids = train_entity_features[0]
                            all_input_ids = train_entity_features[1]
                            all_segment_ids = train_entity_features[2]
                            all_label = train_entity_features[3]


                        if master_process(args):
                            logger.info("***** Eval results *****")
                            logger.info(json.dumps(result, indent=2))

                        model.train()

                        if master_process(args):
                            # if best_perf is None or isinstance(result["eval_accuracy"], dict) or best_perf < result["eval_accuracy"]:
                            model_to_save = model.module if hasattr(
                                model, 'module') else model  # Only save the model it-self
                            if task_name == "mvd":
                                model_to_save = model_to_save.retriever
                            output_model_file = os.path.join(
                                args.output_dir, task_name + str(global_step) + ".bin")
                            torch.save(model_to_save.state_dict(),output_model_file)
                            output_config_file = os.path.join(args.output_dir, "config.json")
                            with open(output_config_file, 'w') as f:
                                f.write(config.to_json_string())
                            f.close()

                            if (best_perf is not None) and (args.best_checkpoint_metric in best_perf):
                                best_perf = result["eval_metrics"] if result["eval_metrics"][args.best_checkpoint_metric] > best_perf[
                                    args.best_checkpoint_metric] else best_perf
                            else:
                                best_perf = result["eval_metrics"]
            # if master_process(args):
            #     print("Retrieval Result = {}".format(str(r_num/total_num)))
        if master_process(args):
            logger.info("Best performance = %s" % json.dumps(best_perf, indent=2))
            if "global_step" in best_perf:
                onlyfiles = [f for f in os.listdir(args.output_dir) if
                             os.path.isfile(os.path.join(args.output_dir, f)) and 'pytorch_model.bin' in f]
                print(onlyfiles)
                for ckpt in onlyfiles:
                    ckpt_step = ckpt[len(args.task_name):-len("pytorch_model.bin")]
                    print("ckpt={}, ckpt_step={}, best_step={}".format(ckpt, ckpt_step, best_perf["global_step"]))
                    if int(ckpt_step) != int(best_perf["global_step"]):
                        os.remove(os.path.join(args.output_dir, ckpt))
                with open(os.path.join(args.output_dir, "best_perf.json"), 'w') as fp_bestperf:
                    json.dump(best_perf, fp_bestperf)
        # perf_writer.close()

    if args.do_eval:
        test_examples = processor.get_test_examples(args.data_dir)
        features = convert_examples_to_features(args,
        test_examples,
        tokenizer,
        output_mode,
        do_train=False
        )
        result, _ = to_eval(args,logger,model,features,device,tokenizer,output_mode)

        if master_process(args):
            logger.info("***** Evaluate results *****")
            logger.info(json.dumps(result, indent=2))
            model.eval()
            dev_perf_file = os.path.join(
                args.output_dir, "dev_{}_perf.json".format(task_name))
            with open(dev_perf_file, mode="w") as writer:
                writer.write(json.dumps(result, indent=2))

    if args.do_predict:
        pred_file_path = os.path.join(args.data_dir,args.predict_file+".jsonl")
        pred_examples = processor.get_predict_examples(pred_file_path)
        features = convert_examples_to_features(args,
        pred_examples,
        tokenizer,
        output_mode,
        do_train=False
        )
        result, _ = to_eval(args,logger,model,features,device,tokenizer,output_mode)

        if master_process(args):
            logger.info("***** Predict results *****")
            logger.info(json.dumps(result, indent=2))
            model.eval()
            dev_perf_file = os.path.join(
                args.output_dir, "dev_{}_perf.json".format(task_name))
            with open(dev_perf_file, mode="w") as writer:
                writer.write(json.dumps(result, indent=2))
        logger.info("***** Inference Finished *****")

if __name__ == "__main__":
    main()
