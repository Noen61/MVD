from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist
import math
from typing import List,Optional, Tuple, Union,TypeVar, Optional, Iterator
from re import template
import sys
import collections
import numpy as np
from io import open
import random
import json
import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score, precision_recall_curve, auc, roc_curve, confusion_matrix

logger = logging.getLogger(__name__)
WORLDS = [
    'american_football',
    'doctor_who',
    'fallout',
    'final_fantasy',
    'military',
    'pro_wrestling',
    'starwars',
    'world_of_warcraft',
    'coronation_street',
    'muppets',
    'ice_hockey',
    'elder_scrolls',
    'forgotten_realms',
    'lego',
    'star_trek',
    'yugioh'
]
WORLDS_set = {}
WORLDS_set['train'] = ['american_football', 'doctor_who', 'fallout', 'final_fantasy', 'military', 'pro_wrestling', 'starwars', 'world_of_warcraft']
WORLDS_set['valid'] = ['coronation_street','muppets','ice_hockey','elder_scrolls']
WORLDS_set['test'] = ['forgotten_realms', 'lego', 'star_trek', 'yugioh']


def master_process(args):
    return args.no_cuda or (args.local_rank == -1) or (torch.distributed.get_rank() == 0)

class InputExample(object):
  """Constructs a InputExample."""

  def __init__(
        self, 
        guid:int,
        mention:Union[str,List[int]],
        mention_left:Optional[str]=None, 
        mention_right:Optional[str]=None,
        cand_idx:List[int]=None,
        entity_idx:int=0,
        domain:str='all',
        ):

    self.guid = guid
    self.mention = mention
    self.mention_left = mention_left
    self.mention_right = mention_right
    self.cand_idx = cand_idx
    self.entity_idx = entity_idx
    self.domain = domain





class InputFeatures(object):
  """Constructs train/dev InputFeatures."""

  def __init__(
        self,
        mention_ids:Optional[List[int]]=None,
        entity_ids:Union[List[int],List[List[int]],List[List[List[int]]]]=None,
        input_ids:Union[List[List[int]],List[List[List[int]]]]=None,
        segment_ids:Union[List[List[int]],List[List[List[int]]]]=None,
        label:int=None,
        entity_idx:int = None,
        mention_tokens:List[str]=None,
        domain:str='all',
        guid:Optional[int]=None
        ):
    
    self.guid = guid
    self.mention_ids = mention_ids
    self.entity_ids = entity_ids
    self.input_ids = input_ids
    self.segment_ids = segment_ids
    self.label = label
    self.entity_idx = entity_idx
    self.mention_tokens = mention_tokens
    self.domain = domain




class DataProcessor(object):
  """Base class for data converters for Entity Linking datasets."""

  def __init__(self,task_name:str="retriever",cand_num:int=16,view_type:str="local"):
        self.wiki_id = dict()
        self.task_name = task_name
        self.cand_num = cand_num
        self.view_type = view_type
  def get_train_examples(self,data_dir):
        """See base class."""
        train_name = "train.jsonl" if self.task_name == "retriever" else "train_cand.jsonl"
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, train_name)))
        return self._create_examples(self._read_json(os.path.join(data_dir, train_name)))
  def get_dev_examples(self, data_dir):
        """See base class."""
        dev_name = "valid_cand.jsonl" if self.task_name == "teacher" else "valid.jsonl"
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, dev_name)))
        return self._create_examples(self._read_json(os.path.join(data_dir, dev_name)),do_train=False)
  def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "test.jsonl")))
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.jsonl")),do_train=False)
  def get_predict_examples(self, predict_file):
        """See base class."""
        logger.info("LOOKING AT {}".format(predict_file))
        return self._create_examples(self._read_json(predict_file),do_train=False)
  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
      # reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      reader = csv.reader((line.replace('\0', '') for line in f),delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      logger.info("Read %d lines from %s" % (len(lines), input_file))
      return lines

  @classmethod
  def _read_json(cls, input_file):
    """Reads a tab separated value file."""
    with open(input_file, "r") as f:
        lines = f.readlines()
    return lines



class WikipediaProcessor(DataProcessor):
    """Processor for the Wikipedia-based datasets."""
    def _create_examples(self,lines,do_train:bool=True):
        """Creates examples for the training and dev sets."""

        examples = list()
        for (i, line) in enumerate(lines):
          item = json.loads(line)
          if self.task_name == "retriever" or do_train == False:
            entity_idx = item['label_id']
            try:
              entity_idx = self.wiki_id[entity_idx]
            except:
              continue
            mention = item['mention']
            mention_left = item['context_left']
            mention_right = item['context_right']
            examples.append(InputExample(guid=i, mention=mention, mention_left=mention_left, mention_right=mention_right,entity_idx=entity_idx))
          else:
            mention = item['mention']
            cand_idx = item['cand_idx']
            entity_idx = item['entity_idx']
            # Top-K negatives
            if self.task_name == "teacher":
              cand_idx = cand_idx[:self.cand_num]
            # Retrieve Top-N then sample Top-K
            elif self.task_name == "mvd":
              cand_idx = random.sample(cand_idx,self.cand_num)
            # Assume golden entity in candidates when training
            if do_train and entity_idx not in cand_idx:
              cand_idx[-1] = entity_idx
            examples.append(InputExample(guid=i, mention=mention,cand_id=cand_idx,entity_idx=entity_idx))


        return examples
class ZeshelProcessor(DataProcessor):
    """Processor for the Zeshel dataset."""

    def _create_examples(self, lines,do_train:bool=True):
        """Creates examples for the training and dev sets."""

        examples = list()
        for (i, line) in enumerate(lines):
          item = json.loads(line)
          if self.task_name != "retriever" or (do_train == False and self.task_name == 'mvd'):
            entity_idx = item['label_id']
            mention = item['mention']
            mention_left = item['context_left']
            mention_right = item['context_right']
            domain = item['world']
            examples.append(InputExample(guid=i, mention=mention, mention_left=mention_left, mention_right=mention_right,entity_idx=entity_idx,domain=domain))
          else:
            mention = item['mention']
            domain = item['domain']
            cand_idx = item['cand_idx']
            entity_idx = item['entity_idx']
            # Top-K negatives
            if self.task_name == "teacher":
              cand_idx = cand_idx[:self.cand_num]
            # Retrieve Top-N then sample Top-K
            elif self.task_name == "mvd":
              cand_idx = random.sample(cand_idx,self.cand_num)
            # Assume golden entity in candidates when training
            if do_train and entity_idx not in cand_idx:
              cand_idx[-1] = entity_idx
            examples.append(InputExample(guid=i, mention=mention,cand_idx=cand_idx,entity_idx=entity_idx,domain=domain))

        return examples

def get_entity_examples(data_dir,view_type:str="local"):
        """See base class."""
        local_views,global_views = dict(),dict()
        for domain in WORLDS:
          local_views[domain],global_views[domain] = list(),list()
          f = open(os.path.join(data_dir,domain+".json"), 'r')
          lines = f.readlines()
          for line in lines:
              item = json.loads(line)
              if view_type == "local" or "global-local":
                local_views[domain].append(item['entity_ids'])
              if view_type == "global" or "global-local":
                global_views[domain].append(item['global_ids'])
          f.close()

        return local_views,global_views

def process_mention(tokenizer,mention,mention_left,mention_right,max_seq_length):
  # process mention format: [CLS] context_left [STR] mention [END] context_right [SEP]
  mention_tokens = ['[unused0]'] + tokenizer.tokenize(mention) + ['[unused1]']
  context_left = tokenizer.tokenize(mention_left)
  context_right = tokenizer.tokenize(mention_right)

  left_quota = (max_seq_length - len(mention_tokens)) // 2 - 1
  right_quota = max_seq_length - len(mention_tokens) - left_quota - 2
  left_add = len(context_left)
  right_add = len(context_right)
  if left_add <= left_quota:
      if right_add > right_quota:
          right_quota += left_quota - left_add
  else:
      if right_add <= right_quota:
          left_quota += right_quota - right_add

  mention_tokens = (
      context_left[-left_quota:] + mention_tokens + context_right[:right_quota]
  )
  if len(mention_tokens) > max_seq_length - 2:
    mention_tokens = mention_tokens[:(max_seq_length - 2)]
  mention_tokens = ['[CLS]'] + mention_tokens + ['[SEP]']
  mention_ids = tokenizer.convert_tokens_to_ids(mention_tokens)
  mention_padding = [0] * (max_seq_length - len(mention_ids))
  mention_ids += mention_padding
  return mention_ids,mention_tokens

def convert_examples_to_features(
      args,
      examples:List[InputExample],
      tokenizer,
      output_mode:str="retrieval",
      do_train:bool=True):
  """Loads a data file into a list of `InputBatch`s."""

  features = []

  task_name = args.task_name
  entity_dir = args.entity_data_dir
  max_seq_length = args.max_seq_length
  cand_num = args.cand_num
  view_type = args.train_view_type
  max_ent_length = args.max_ent_length
  max_view_num,max_view_length = args.max_view_num,args.max_view_length

  if do_train == False and output_mode == "retrieval":
    for (ex_index, example) in enumerate(examples):
      # Process mention
      mention_ids,mention_tokens = process_mention(tokenizer,example.mention,example.mention_left,example.mention_right,max_seq_length)

      features.append(
        InputFeatures(guid=example.guid,
                      mention_ids=mention_ids,
                      mention_tokens=mention_tokens,
                      entity_idx = example.entity_idx,
                      domain = example.domain)
                      )
      
  elif task_name == "retriever":
    local_views,global_views = get_entity_examples(entity_dir,view_type)
    for (ex_index, example) in enumerate(examples):
      # Process mention
      mention_ids,mention_tokens = process_mention(tokenizer,example.mention,example.mention_left,example.mention_right,max_seq_length)


      # Process entity
      if view_type == "global":
        entity_ids = global_views[example.domain][example.entity_idx]
        entity_ids = [101] + entity_ids[1:-2][:max_ent_length-2] + [102]
        entity_ids += [0] * (max_ent_length-len(entity_ids))
        assert len(entity_ids) == max_ent_length

      else:
        entity_ids = list()
        views = local_views[example.domain][example.entity_idx][:max_view_num]
        view_num = len(views)
        if view_num < max_view_num:
          view_padding = max_view_num - view_num
          views += [[0] * max_view_length for _ in range(view_padding)]
        for view_ids in views:
          entity_ids.append(view_ids)

      if ex_index == 0 and master_process(args):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("entity_idx: %s" % (example.entity_idx))
        logger.info("mention: %s" % (example.mention))
        logger.info("domain: %s" % (example.domain))
        logger.info("mention_ids: %s" % " ".join([str(x) for x in mention_ids]))
        if view_type == "local":
          logger.info("entity_ids: %s" % " ".join([str(x) for x in entity_ids[0]]))
        else:
          logger.info("entity_ids: %s" % " ".join([str(x) for x in entity_ids]))

      features.append(
        InputFeatures(mention_ids=mention_ids,
                      entity_ids=entity_ids,
                      domain = example.domain
                      )
                    )


  elif task_name == "mvd":
    local_views,global_views = get_entity_examples(entity_dir,view_type)
    for (ex_index, example) in enumerate(examples):
      label = example.cand_idx.index(example.entity_idx) if example.entity_idx in example.cand_idx else cand_num
      candidate_entity_ids,candidate_input_ids,candidate_segment_ids = list(),list(),list()
      # process mention
      mention_ids = tokenizer.convert_tokens_to_ids(example.mention)
      mention_padding = [0] * (max_seq_length-len(mention_ids))
      mention_ids_padding = mention_ids + mention_padding

      # process candidate entities
      for i in example.cand_idx:
        
        if view_type == "local":
          entity_ids,input_ids,segment_ids = list(),list(),list()
          views = local_views[example.domain][i][:max_view_num]
          view_num = len(views)
          if view_num < max_view_num:
            view_padding = max_view_num - view_num
            views += [[0] * max_view_length for _ in range(view_padding)]

          for view_ids in views:
            tmp_mention_ids = mention_ids.copy()
            # remove CLS and concate
            tmp_view_ids = view_ids[1:]
            men_view_ids = tmp_mention_ids + tmp_view_ids
            ce_padding = [0] * (max_view_length + max_seq_length - len(men_view_ids))
            men_view_ids += ce_padding

            entity_ids.append(view_ids)
            input_ids.append(men_view_ids)
            segment_ids.append(len(tmp_mention_ids))
        
        elif view_type == "global":
          tmp_mention_ids = mention_ids.copy()
          tmp_entity_ids = global_views[example.src][i][1:-2]
          tmp_entity_ids =  tmp_entity_ids[:max_ent_length-2] + [102]
          input_ids = tmp_mention_ids + tmp_entity_ids
          ce_padding = [0] * (max_ent_length + max_seq_length - len(input_ids))
          input_ids += ce_padding

          entity_ids = [101]+tmp_entity_ids
          entity_padding = [0] * (max_ent_length - len(entity_ids))
          entity_ids += entity_padding
          segment_ids = len(tmp_mention_ids)

        candidate_entity_ids.append(entity_ids)
        candidate_input_ids.append(input_ids)
        candidate_segment_ids.append(segment_ids)

      

      if ex_index == 0 and master_process(args):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("candidates_id: %s" % (example.cand_idx))
        logger.info("mention: %s" % (example.mention))
        logger.info("domain: %s" % (example.domain))
        logger.info("mention_ids: %s" % " ".join([str(x) for x in mention_ids_padding]))
        if view_type == "local":
          logger.info("entity_ids: %s" % " ".join([str(x) for x in candidate_entity_ids[0][0]]))
          logger.info("input_ids: %s" % " ".join([str(x) for x in candidate_input_ids[0][0]]))
        else:
          logger.info("entity_ids: %s" % " ".join([str(x) for x in candidate_entity_ids[0]]))
          logger.info("input_ids: %s" % " ".join([str(x) for x in candidate_input_ids[0]]))

      features.append(
        InputFeatures(guid=example.guid,
                      mention_ids=mention_ids_padding,
                      entity_ids=candidate_entity_ids,
                      input_ids=candidate_input_ids,
                      segment_ids=candidate_segment_ids,
                      label=label,
                      entity_idx = example.entity_idx,
                      mention_tokens = example.mention,
                      domain = example.domain)
                      )

  elif task_name == "teacher":
    local_views,global_views = get_entity_examples(entity_dir,view_type)
    for (ex_index, example) in enumerate(examples):

      label = example.cand_idx.index(example.entity_idx) if example.entity_idx in example.cand_idx else cand_num
      candidate_input_ids,candidate_segment_ids = list(),list()

      mention_tokens = example.mention
      mention_tokens = mention_tokens[:max_seq_length]
      mention_ids = tokenizer.convert_tokens_to_ids(mention_tokens)
      mention_padding = [0] * (max_seq_length-len(mention_ids))
      mention_ids_padding = mention_ids + mention_padding

      for i in example.cand_idx:
        
        if view_type == "local":
          views = local_views[example.domain][i]
          view_num = len(views)
          if view_num < max_view_num:
            pad_view = max_view_num - view_num
            views += [[0] * max_view_length for _ in range(pad_view)]
          views = views[:max_view_num]

          input_ids,segment_ids = list(),list()
          for view_ids in views:
            tmp_mention_ids = mention_ids.copy()
            # remove CLS and concate
            tmp_view_ids = view_ids[1:]
            men_view_ids = tmp_mention_ids + tmp_view_ids
            ce_padding = [0] * (max_view_length + max_seq_length - len(men_view_ids))
            men_view_ids += ce_padding

            input_ids.append(men_view_ids)
            segment_ids.append(len(tmp_mention_ids))

        elif view_type == "global":
          
          tmp_mention_ids = mention_ids.copy()
          tmp_entity_ids = global_views[example.src][i][1:-2]
          tmp_entity_ids =  tmp_entity_ids[:max_ent_length-2] + [102]
          input_ids = tmp_mention_ids + tmp_entity_ids
          ce_padding = [0] * (max_ent_length + max_seq_length - len(input_ids))
          input_ids += ce_padding
          segment_ids = len(tmp_mention_ids)
        
        candidate_input_ids.append(input_ids)
        candidate_segment_ids.append(segment_ids)

      features.append(
        InputFeatures(guid=example.guid,
                      input_ids=candidate_input_ids,
                      segment_ids=candidate_segment_ids,
                      label=label,
                      domain = example.domain)
                    )
  return features

processors = {
  "zeshel": ZeshelProcessor,
  "wikipedia":WikipediaProcessor
}

output_modes = {
  "retriever":"retrieval",
  "teacher":"rank",
  "mvd": "retrieval"
}
