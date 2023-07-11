import logging
import os
import json
from tqdm import tqdm
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset,Dataset)
from utils.data_utils import (master_process,InputExample,convert_examples_to_features,WORLDS_set)
from generate_index import generate_new_embeddings
from typing import List,Optional, Tuple, Union,TypeVar, Optional, Iterator

def eval_rank(dataloader,model,device,global_step,all_domain):

    targets,preds = list(),list()
    pred_hit,domain_hit = 0,0
    for input_ids,segment_ids,label in tqdm(dataloader):  
        
        input_ids = input_ids.to(device)
        segment_ids = segment_ids.to(device)
        label = label.tolist()

        with torch.no_grad():
   
            logits = model(input_ids=input_ids,segment_ids=segment_ids,return_loss=False)
            logits = logits.max(dim=-1)[0]
            logits = torch.argmax(logits,dim=-1).detach().cpu().tolist()
            targets.extend(label)
            preds.extend(logits)

    target_domain,pred_domain = dict(),dict()

    for i in range(len(targets)):
        domain = all_domain[i]
        if target_domain.get(domain) is None:
            target_domain[domain] = 0
        if pred_domain.get(domain) is None:
            pred_domain[domain] = 0
        target_domain[domain] += 1
        if targets[i] == preds[i]:
            pred_hit += 1
            pred_domain[domain] += 1
            
    for domain in target_domain.keys():
        domain_hit += pred_domain[domain] / target_domain[domain]

    acc = pred_hit/len(targets)
    marco_acc = domain_hit/len(target_domain.keys())
            
    result = dict()
    result["accuracy"] = acc
    result["marco_accuracy"] = marco_acc
    
    result = {
    'eval_metrics': result,
    'global_step': global_step,
    }

    return result

def eval_retrieval(args,logger,dataloader,model,device,global_step,all_domain):
    
    mention_embeds,entity_idxs = list(),list()
    for mention_ids,entity_idx in dataloader:
        
        mention_ids = mention_ids.to(device)
        with torch.no_grad():
            mention_embed = model(mention_ids=mention_ids)
            mention_embed = mention_embed.detach().cpu().numpy()
            mention_embeds.append(mention_embed)
            entity_idxs.append(entity_idx)
            
    mention_embeds = np.concatenate(mention_embeds, axis=0)
    entity_idxs = np.concatenate(entity_idxs, axis=0)
    if args.do_eval:
        dataset = "test"
    elif args.do_predict:
        dataset = args.predict_file
    else:
        dataset = 'valid'
    topk_hits,cand_entities = generate_new_embeddings(args,logger,model,device,mention_embeds,entity_idxs,all_domain,dataset)

    result = {}
    result['global_step'] = global_step
    if master_process(args):
        result["top1_hits"] = topk_hits[0]
        result["top2_hits"] = topk_hits[1]
        result["top4_hits"] = topk_hits[3]
        result["top8_hits"] = topk_hits[7]
        result["top10_hits"] = topk_hits[9]
        result["top16_hits"] = topk_hits[15]
        result["top30_hits"] = topk_hits[29]
        result["top32_hits"] = topk_hits[31]
        result["top50_hits"] = topk_hits[49]
        result["top64_hits"] = topk_hits[63]
        result["top100_hits"] = topk_hits[99]
    result = {
    'eval_metrics': result,
    'global_step': global_step,
    }
    return result,cand_entities

def to_eval(args,logger,
    model,
    eval_features,
    device,tokenizer,
    output_mode:str,
    global_step:int=0,
    train_mention_features:Optional[Tuple]=None
    ):
    
    batch_size = args.eval_batch_size
    task_name = args.task_name
    output_dir = args.output_dir
    

    all_domain = [f.domain for f in eval_features]

    if master_process(args):
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_features))
        logger.info("  Evaluation type = {}".format(output_mode))
        logger.info("  Global steps = {}".format(global_step))
    
    if output_mode == "retrieval":
        all_mention_ids = torch.tensor(
            [f.mention_ids for f in eval_features], dtype=torch.long)
        all_entity_idx = torch.tensor(
            [f.entity_idx for f in eval_features], dtype=torch.long)
        
        eval_data = TensorDataset(all_mention_ids,all_entity_idx)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)
        result,cand_entities = eval_retrieval(args,logger,eval_dataloader,model,device,global_step,all_domain)

        if args.do_predict:
            if master_process(args):
                predict_file_path = os.path.join(args.data_dir,args.predict_file+"_cand.jsonl")
                entity_idxs = all_entity_idx.tolist()
                all_mention_tokens = [f.mention_tokens for f in eval_features]
                fout = open(predict_file_path, 'w')
                for mention_tokens,cand_idx,entity_idx,domain in zip(all_mention_tokens,cand_entities,entity_idxs,all_domain):
                    field = dict()
                    field['mention'] = mention_tokens
                    field['cand_idx'] = cand_idx
                    field['entity_idx'] = entity_idx
                    field['domain'] = domain
                    fout.write('%s\n' % json.dumps(field))
                fout.close()
        
    elif output_mode == "rank":
        all_input_ids = torch.from_numpy(
            np.asarray([f.input_ids for f in eval_features]).astype(np.int64))
        all_segment_ids = torch.from_numpy(
            np.asarray([f.segment_ids for f in eval_features]).astype(np.int64))
        all_label = torch.from_numpy(
            np.asarray([f.label for f in eval_features]).astype(np.int64))
    
        eval_data = TensorDataset(all_input_ids,all_segment_ids,all_label)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)
        result = eval_rank(eval_dataloader,model,device,global_step,all_domain)

    if master_process(args) and not args.do_predict:
        output_eval_file = os.path.join(output_dir, "eval_results.txt")
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results {} *****".format(global_step))
            writer.write("{0}\n".format(global_step))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    dist.barrier()

    train_entity_features = tuple()
    if task_name == "mvd":
        cand_num = args.cand_num
        train_mention_ids,train_entity_idx,train_mention_tokens,train_domain = train_mention_features
        if master_process(args):
            logger.info("***** Running hard negative mining *****")
            logger.info("  Num examples = %d", train_entity_idx.size(0))
            logger.info("  Num candidates = %d", cand_num)
            logger.info("  Global steps = {}".format(global_step))

        train_data = TensorDataset(train_mention_ids,train_entity_idx)
        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        train_embedding_ids,train_embeddings =[],[]
        for mention_ids,entity_idx in train_dataloader:

            mention_ids = mention_ids.to(device)
            with torch.no_grad():
                mention_emd = model(mention_ids=mention_ids)
                mention_emd = mention_emd.detach().cpu().numpy()
                train_embeddings.append(mention_emd)
                train_embedding_ids.append(entity_idx)

        train_embeddings = np.concatenate(train_embeddings, axis=0)
        train_embedding_ids = np.concatenate(train_embedding_ids, axis=0)
        train_topk_hits,train_scores = generate_new_embeddings(args,logger,model,device,train_embeddings,train_embedding_ids,train_domain,dataset='train')

        if not master_process(args):
            train_scores = []
            cand_path = os.path.join(args.cache_dir, "train_negatives.jsonl")
            with open(cand_path, 'rt') as f:
                for line in f:
                    line = line.rstrip()
                    item = json.loads(line)
                    train_scores.append(item['cand_idx'])

        train_examples = list()
        for i in range(len(train_scores)):
            train_label = train_scores[i][:cand_num]
            train_e_id = train_embedding_ids.tolist()[i]
            if int(train_e_id) not in train_label:
                    train_label[-1] = int(train_e_id)
            train_examples.append(InputExample(guid=i, mention=train_mention_tokens[i], domain=train_domain[i], cand_idx=train_label,entity_idx=train_e_id))

        train_features = convert_examples_to_features(args,train_examples,tokenizer,output_mode)
        train_segment_ids = torch.from_numpy(
                np.asarray([f.segment_ids for f in train_features]).astype(np.int64))
        train_label = torch.tensor(
                [f.label for f in train_features], dtype=torch.long)
        
        entity_path = os.path.join(args.cache_dir,'train_entity_ids.pt')
        input_path = os.path.join(args.cache_dir,'train_input_ids.pt')
        if master_process(args):
            train_entity_ids = torch.from_numpy(
                np.asarray([f.entity_ids for f in train_features]).astype(np.int64)) 
            train_input_ids = torch.from_numpy(
                np.asarray([f.input_ids for f in train_features]).astype(np.int64))
            torch.save(train_entity_ids, entity_path)
            torch.save(train_input_ids, input_path)
        dist.barrier()
        if not master_process(args):
            train_entity_ids = torch.load(entity_path)
            train_input_ids = torch.load(input_path)     
        dist.barrier()
        if master_process(args):
            os.remove(entity_path)
            os.remove(input_path)

        train_entity_features = (train_entity_ids,train_input_ids,train_segment_ids,train_label)
    

    return result,train_entity_features
