import math
import pickle
import json
import os
import gc
import random
from tqdm import tqdm
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader,Dataset
from torch.utils.data.distributed import DistributedSampler
#import faiss
from utils.data_utils import master_process,WORLDS_set,get_entity_examples

class Eval_Tool:
    @classmethod
    def MRR_n(cls, results_list, n):
        mrr_100_list = []
        for hits in results_list:
            score = 0
            for rank, item in enumerate(hits[:n]):
                if item:
                    score = 1.0 / (rank + 1.0)
                    break
            mrr_100_list.append(score)
        return sum(mrr_100_list) / len(mrr_100_list)

    @classmethod
    def MAP_n(cls, results_list, n):
        MAP_n_list = []
        for predict in results_list:
            ap = 0
            hit_num = 1
            for rank, item in enumerate(predict[:n]):
                if item:
                    ap += hit_num / (rank + 1.0)
                    hit_num += 1
            ap /= n
            MAP_n_list.append(ap)
        return sum(MAP_n_list) / len(MAP_n_list)

    @classmethod
    def DCG_n(cls, results_list, n):
        DCG_n_list = []
        for predict in results_list:
            DCG = 0
            for rank, item in enumerate(predict[:n]):
                if item:
                    DCG += 1 / math.log2(rank + 2)
            DCG_n_list.append(DCG)
        return sum(DCG_n_list) / len(DCG_n_list)

    @classmethod
    def nDCG_n(cls, results_list, n):
        nDCG_n_list = []
        for predict in results_list:
            nDCG = 0
            for rank, item in enumerate(predict[:n]):
                if item:
                    nDCG += 1 / math.log2(rank + 2)
            nDCG /= sum([math.log2(i + 2) for i in range(n)])
            nDCG_n_list.append(nDCG)
        return sum(nDCG_n_list) / len(nDCG_n_list)

    @classmethod
    def P_n(cls, results_list, n):
        p_n_list = []
        for predict in results_list:
            true_num = 0
            for rank, item in enumerate(predict[:n]):
                if item:
                    true_num += 1
            p = true_num / n
            p_n_list.append(p)
        return sum(p_n_list) / len(p_n_list)

    @classmethod
    def get_matrics(cls, results_list):
        p_list = [1, 5, 10, 20, 50]
        metrics = {'MRR_n': cls.MRR_n,
                   'MAP_n': cls.MAP_n,
                   'DCG_n': cls.DCG_n, 'nDCG_n': cls.nDCG_n, 'P_n': cls.P_n}
        result_dict = {}
        for metric_name, fuction in metrics.items():
            for p in p_list:
                temp_result = fuction(results_list, p)
                result_dict[metric_name + '@_' + str(p)] = temp_result
        return result_dict

class EntityDataset(Dataset):

    def __init__(self, entities,view_type="local"):
        self.len = len(entities)
        self.entities = entities
        self.view_type = view_type


    def __len__(self):
        'Denotes the total number of samples'
        return self.len

    def free(self):
        self.inputs = None

    def __getitem__(self, index,max_ent_length=512):
        
        entity_ids = self.entities[index][0]
        # global-view
        if self.view_type != "local":
            entity_ids = [101] + entity_ids[1:-2][:max_ent_length-2] + [102]
            entity_ids += [0] * (max_ent_length-len(entity_ids))
        entity_ids = torch.LongTensor(entity_ids)
        entity_idx = self.entities[index][1]
        res = [entity_ids,entity_idx]
        return res

def validate(closest_entities,entity_idxs,view2entity,top_k=100):
    
    mention_num = len(entity_idxs)
    top_k_hits  = [0] * top_k
    final_scores = list()
    pred_entity_idxs = list()
    for i in range(mention_num):
        hits = [False] * len(closest_entities[0])
        entity_idx = entity_idxs[i]
        pred_entity_idx = [eidx for eidx in closest_entities[i]]
        if view2entity is not None:
            pred_entity_idx = [view2entity[eidx] for eidx in closest_entities[i]]

        new_pred_entity_idx = list()
        for item in pred_entity_idx:
            if item not in new_pred_entity_idx and len(new_pred_entity_idx) < top_k:
                new_pred_entity_idx.append(item)
        pred_entity_idxs.append(new_pred_entity_idx)
        pred_entity_idx = new_pred_entity_idx

        if entity_idx in pred_entity_idx:
            h_index = pred_entity_idx.index(entity_idx)
            hits[h_index:] = [True for v in hits[h_index:]]
            top_k_hits[h_index:] = [v + 1 for v in top_k_hits[h_index:]]
        final_scores.append(hits)

    return top_k_hits,pred_entity_idxs

def embed_entities(args,model,device,dataset="valid"):
    
    domain_idx = dict()
    view2entity = dict()
    view_type = args.train_view_type if dataset == "train" else args.infer_view_type
    local_views,global_views = get_entity_examples(args.entity_data_dir,view_type)
    local_view_embeds,global_view_embeds = list(),list()

    if args.kb != "zeshel":
        WORLDS_set[dataset] = ['all']

    num = 0
    for domain in WORLDS_set[dataset]:
        view2entity[domain] = dict()
        view_idx,entity_idx = 0,0
        start_idx = num
        for local_ids,global_ids in zip(local_views[domain],global_views[domain]):

            if view_type == "local":
                entity_ids = local_ids
            elif view_type == "global":
                entity_ids = [global_ids]
            elif view_type == "global-local":
                entity_ids = [global_ids] + local_ids

            if dataset == "train":
                entity_ids = entity_ids[:args.max_view_num]

            view_num = len(entity_ids)
            for i in range(view_num):
                if i == 0 and view_type != "local":
                    global_view_embeds.append((entity_ids[i],num))
                else:
                    local_view_embeds.append((entity_ids[i],num))

                view2entity[domain][view_idx] = entity_idx

                num += 1
                view_idx += 1
            entity_idx += 1
        end_idx = num
        domain_idx[domain] = [start_idx,end_idx]
    
    dist.barrier()

    entity_idxs,entity_embeds = list(),list()
    dasatests = list()
    with torch.no_grad():
        if len(local_view_embeds) > 0:
            dasatests.append(EntityDataset(local_view_embeds,view_type="local"))
        if len(global_view_embeds) > 0:
            dasatests.append(EntityDataset(global_view_embeds,view_type="global"))
        
        for dataset in dasatests:
            infer_sampler = DistributedSampler(dataset)
            infer_dataloader = DataLoader(
                dataset, sampler=infer_sampler, batch_size=args.eval_batch_size)
            if master_process(args):
                infer_dataloader = tqdm(infer_dataloader)
            for batch in infer_dataloader:
                entity_ids,entity_idx = batch
                entity_ids = entity_ids.to(device)
                entity_emd = model(entity_ids=entity_ids)
                entity_emd = entity_emd.detach().cpu()

                entity_idxs.append(entity_idx)
                entity_embeds.append(entity_emd)


    entity_embeds = torch.cat(entity_embeds, dim=0).numpy()
    entity_idxs = torch.cat(entity_idxs, dim=0).numpy()

    return entity_idxs,entity_embeds,domain_idx,view2entity

def get_entity_embedding(args,logger,model,device,dataset="valid",load_cache=False):

    output_dir = os.path.join(args.cache_dir,"embeds")
    process_num = torch.distributed.get_world_size()
    os.makedirs(output_dir, exist_ok=True)
    entity_idxs,entity_embeds,domain_idx,view2entity = embed_entities(args,model,device,dataset=dataset)
    if not args.load_cache and not load_cache:
        pickle_path = os.path.join(output_dir,"{}_data_obj_{}.pb".format('entity_embedding',str(args.local_rank)))
        if pickle_path is not None and os.path.exists(pickle_path):
                os.remove(pickle_path)
        with open(pickle_path, 'wb') as handle:
            pickle.dump(entity_embeds, handle, protocol=4)

        pickle_path = os.path.join(output_dir,"{}_data_obj_{}.pb".format('entity_embedding_idx',str(args.local_rank)))
        with open(pickle_path, 'wb') as handle:
            pickle.dump(entity_idxs, handle, protocol=4)
        logger.info(f'Total entities processed {len(entity_idxs)*process_num}. Written to {pickle_path}.')
        dist.barrier()
    if not master_process(args):
        dist.barrier()

    entity_idxs,entity_embeds = list(),list()    
    if master_process(args):
        entity_embeds = []
        entity_idxs = []

        for i in range(process_num):
            pickle_path = os.path.join(output_dir,
                                                "{}_data_obj_{}.pb".format('entity_embedding',str(i)))
            with open(pickle_path, 'rb') as handle:
                entity_embed = pickle.load(handle)
                entity_embeds.append(entity_embed)

            pickle_path = os.path.join(output_dir,
                                            "{}_data_obj_{}.pb".format('entity_embedding_idx',str(i)))
            with open(pickle_path, 'rb') as handle:
                entity_idx = pickle.load(handle)
                entity_idxs.append(entity_idx)

        entity_embeds = np.concatenate(entity_embeds, axis=0)
        entity_idxs = np.concatenate(entity_idxs, axis=0)


        print('view num: ' + str(entity_embeds.shape[0]))
        dist.barrier()
    return entity_embeds,entity_idxs,domain_idx,view2entity

def generate_new_embeddings(args,
    logger,
    model,
    device,
    mention_embeds,
    entity_idxs,
    domains,
    dataset="valid",
    load_cache=False):

    top_k_hits = 0
    retrieval_labels = [None] * len(mention_embeds)
    if master_process(args):
        logger.info("***** inference of entities *****")
    entity_embedding, entity_embedding2id,worlds,view2entity = get_entity_embedding(args,logger,model,device,dataset=dataset,load_cache=load_cache)
    if not master_process(args):
        dist.barrier()
    if master_process(args):
        logger.info("***** Begin entity_embedding reorder *****")
        new_entity_embedding = entity_embedding.copy()
        for i in range(entity_embedding.shape[0]):
            new_entity_embedding[entity_embedding2id[i]] = entity_embedding[i]
        del entity_embedding,entity_embedding2id
        gc.collect()
        entity_embedding = new_entity_embedding

        logger.info("***** Done entities inference *****")
        dim = entity_embedding.shape[1]
        logger.info('entity embedding shape: ' + str(entity_embedding.shape))
        logger.info("***** Begin embedding build *****")
        men_embeds = dict()
        ent_idxs = dict()
        men_guids = dict()
        
        if args.kb != "zeshel":
            WORLDS_set[dataset] = ['all']
        for domain in WORLDS_set[dataset]:
            men_guids[domain] = list()
            men_embeds[domain] = list()
            ent_idxs[domain] = list()
        for i in range(mention_embeds.shape[0]):
            men_embeds[domains[i]].append([mention_embeds[i]])
            ent_idxs[domains[i]].append(entity_idxs[i])
            men_guids[domains[i]].append(i)
        for domain in WORLDS_set[dataset]:
            men_embeds[domain] = np.concatenate(men_embeds[domain], axis=0)

        logger.info("***** Begin ANN Index build *****")
        top_k = args.top_k * 10
        if dataset != "train":
            top_k = args.top_k * 30
        
        if args.faiss:
            import faiss
            faiss.omp_set_num_threads(args.thread_num)

        all_top_k_hits = [0] * args.top_k
        query_num = 0
        for src in WORLDS_set[dataset]:
            
            new_embedding = entity_embedding[worlds[src][0]:worlds[src][1]]
            if args.faiss:
                cpu_index = faiss.IndexFlatIP(dim)
                cpu_index.add(new_embedding.astype(np.float32))
                _, closest_entities = cpu_index.search(men_embeds[src].astype(np.float32),top_k)
            else:   
                infer_size = args.eval_batch_size
                closest_entities = list()
                cand_embedding = torch.Tensor(new_embedding.astype(np.float32)).to(device)
                for i in range(math.ceil(len(men_embeds[src])/infer_size)):
                    search_embedding = torch.Tensor(men_embeds[src][i*infer_size:(i+1)*infer_size].astype(np.float32)).to(device)
                    _, closest_entity = torch.matmul(search_embedding,cand_embedding.T).topk(top_k,dim=-1)
                    closest_entities.append(closest_entity.detach().cpu())
                closest_entities = torch.cat(closest_entities,dim=0).numpy()
            
            top_k_hits,cand_idxs = \
                validate(closest_entities,ent_idxs[src],view2entity[src],top_k=args.top_k)
            
            for guid,idx in zip(men_guids[src],cand_idxs):
                if dataset == 'train' and args.task_name == "mvd":
                   cand_idx = random.sample(idx[:args.top_k],args.cand_num)
                   retrieval_labels[guid] = cand_idx
                else:
                    retrieval_labels[guid] = idx

            query_num += len(closest_entities)
            all_top_k_hits = [v0+v1 for v0,v1 in zip(all_top_k_hits,top_k_hits)]
        logger.info("***** Done test validate *****")
        top_k_hits = [v/query_num for v in all_top_k_hits]
        if dataset == 'train':
            cand_path = os.path.join(args.cache_dir, "train_negatives.json")
        else:
            cand_path = os.path.join(args.cache_dir, "test_negatives.json")
        if cand_path is not None and os.path.exists(cand_path):
                os.remove(cand_path)
        with open(cand_path, 'w') as fin:
            for r_labels in retrieval_labels:
                r = dict()
                r['cand_idx'] = r_labels
                fin.write('%s\n' % json.dumps(r))
        fin.close()
        dist.barrier()
    return top_k_hits,retrieval_labels
