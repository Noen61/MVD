import argparse
import json
import logging
import os
import random
import time
import sys
import requests
import nltk
from tqdm import tqdm
from datetime import timedelta
from transformers import BertTokenizer

BEGIN_ENT_TOKEN = "[START_ENT]"
END_ENT_TOKEN = "[END_ENT]"
url2id_cache = {}

WORLDS = {
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
}

domain_set = {}
domain_set['val'] = set(['coronation_street', 'muppets', 'ice_hockey', 'elder_scrolls'])
domain_set['test'] = set(['forgotten_realms', 'lego', 'star_trek', 'yugioh'])
domain_set['train'] = set(['american_football', 'doctor_who', 'fallout', 'final_fantasy', 'military', 'pro_wrestling', 'starwars', 'world_of_warcraft'])

def get_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--dataset",
                        default="zeshel",
                        type=str,
                        choices= ["zeshel","wikipedia"],
                        help="The type of dataset.")
    parser.add_argument("--mention_path", default=None, type=str, required=True,
                        help="The input mention dir. Should contain the .json files (or other data files) for the task.")
    parser.add_argument("--document_path", default=None, type=str, required=True,
                        help="The entity data dir. Should contain the .json files (or other data files) for the task.")
    parser.add_argument("--output_path", default=None, type=str, required=True,
                        help="The output dir. Should contain the .json files (or other data files) for the task.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input mention length after WordPiece tokenization.")
    parser.add_argument("--max_ent_length",
                        default=512,
                        type=int,
                        help="The maximum total input entity length after WordPiece tokenization.")
    parser.add_argument("--max_view_length",
                        default=40,
                        type=int,
                        help="The maximum total input view length within each entity after WordPiece tokenization.")
    args = parser.parse_args()
    return args

def get_entity_window(item, tokenizer,max_ent_len=512,max_view_len=40,dataset="zeshel"):

    CLS,ENT,SEP = '[CLS]','[SEP]','[SEP]'

    if dataset == "zeshel":
        title = item['title'].strip()
        text = item['text'].strip()
    else:
        title = item['wikipedia_title'].strip()
        text = item['text'][1:] if len(item['text']) > 1 else item['text']
        text = ' '.join(text)
        text = text.strip()

    title_tokens = tokenizer.tokenize(title)
    text_tokens = tokenizer.tokenize(text)
    window = (title_tokens + [ENT] + text_tokens)[:max_ent_len-2]
    window = [CLS] + window + [SEP]
    global_ids = tokenizer.convert_tokens_to_ids(window)

    entity_ids = tokenize_split_description(title_tokens, text,tokenizer,max_seq_len=max_view_len)
    window = {}
    window['entity_ids'] = entity_ids
    window['global_ids'] = global_ids
    return window


def tokenize_split_description(title, desc, tokenizer,max_seq_len=40):
    ENTITY_TAG = '[SEP]'
    CLS_TAG = '[CLS]'
    SEP_TAG = '[SEP]'
    title_text = title + [ENTITY_TAG]

    multi_sent = []
    pre_text = []

    for sent in nltk.sent_tokenize(desc.replace(' .', '.')):
        text = tokenizer.tokenize(sent)
        pre_text += text
        if len(pre_text) <= 5:
            continue
       
        whole_text = title_text + pre_text
        whole_text = [CLS_TAG] + whole_text[:max_seq_len - 2] + [SEP_TAG]
        tokens = tokenizer.convert_tokens_to_ids(whole_text)
        pre_text = []

        if len(tokens) < max_seq_len:
            tokens += [0] * (max_seq_len - len(tokens))
        assert len(tokens) == max_seq_len
        multi_sent.append(tokens)

    return multi_sent

def load_entity_dict(document_path,out_path,max_ent_len,max_view_len,dataset):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    out_path = os.path.join(out_path,"entity")
    os.makedirs(out_path,exist_ok=True)
    entity_dict = {}
    entity_map = {}
    if dataset == "zeshel":
        for src in WORLDS:
            fout = open(os.path.join(out_path,src+".json"), 'w')
            fname = os.path.join(document_path, src + ".json")
            assert os.path.isfile(fname), "File not found! %s" % fname
            doc_map = {}
            doc_list = []
            with open(fname, 'rt') as f:
                for line in f:
                    field = {}
                    line = line.rstrip()
                    item = json.loads(line)
                    window = get_entity_window(item, tokenizer,max_ent_len,max_view_len,dataset)
                    field = window
                    doc_id = item["document_id"]
                    field["text"] = item['text'].strip()
                    field['doc_id'] = len(doc_list)

                    doc_map[doc_id] = len(doc_list)
                    doc_list.append(field)
                    
                    fout.write('%s\n' % json.dumps(field))
            fout.close()

            entity_dict[src] = doc_list
            entity_map[src] = doc_map
    else:
        fout = open(os.path.join(out_path,"entity.json"), 'w')
        with open(document_path, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                item = json.loads(line)
                # split = item["idx"].split("curid=")
                # if len(split) > 1:
                #         wikipedia_id = int(split[-1].strip())
                # else:
                #     wikipedia_id = item["idx"].strip()
                entity = get_entity_window(item, tokenizer,max_ent_len,max_view_len,dataset)
                entity["wikipedia_id"] = item["wikipedia_id"]
                fout.write('%s\n' % json.dumps(entity))
            fout.close()
            f.close()
    return entity_dict, entity_map


def convert_data(mention_path, output_path,entity_dict, entity_map, mode,max_seq_len):
    if mode == "valid":
        fname = os.path.join(mention_path, "val.json")
    else:
        fname = os.path.join(mention_path, mode + ".json")

    fout = open(os.path.join(output_path, mode + ".jsonl"), 'wt')
    cnt = 0
    max_tok = max_seq_len
    with open(fname, 'rt') as f:
        for line in f:
            cnt += 1
            line = line.rstrip()
            item = json.loads(line)
            mention = item["text"].lower()
            src = item["corpus"]
            label_doc_id = item["label_document_id"]
            orig_doc_id = item["context_document_id"]
            start = item["start_index"]
            end = item["end_index"]

            # add context around the mention as well
            orig_id = entity_map[src][orig_doc_id]
            text = entity_dict[src][orig_id]["text"].lower()
            tokens = text.split(" ")

            assert mention == ' '.join(tokens[start:end + 1]) 

            mention_context_left = tokens[max(0, start - max_tok):start]
            mention_context_right = tokens[end + 1:min(len(tokens), end + max_tok + 1)]

            # entity info
            k = entity_map[src][label_doc_id]
           
            example = {}
            example["context_left"] = ' '.join(mention_context_left)
            example['context_right'] = ' '.join(mention_context_right)
            example["mention"] = mention
            example["label_id"] = k
            example['world'] = src
            fout.write(json.dumps(example))
            fout.write('\n')

    fout.close()


def _get_pageid_from_api(title, client=None):
    pageid = None

    title_html = title.strip().replace(" ", "%20")
    url = "https://en.wikipedia.org/w/api.php?action=query&titles={}&format=json".format(
        title_html
    )

    try:
        # Package the request, send the request and catch the response: r
        r = requests.get(url)

        # Decode the JSON data into a dictionary: json_data
        json_data = r.json()

        if len(json_data["query"]["pages"]) > 1:
            print("WARNING: more than one result returned from wikipedia api")

        for _, v in json_data["query"]["pages"].items():
            pageid = v["pageid"]
    except:
        pass

    return pageid

def extract_questions(filename):

    # all the datapoints
    global_questions = []

    # left context so far in the document
    left_context = []

    # working datapoints for the document
    document_questions = []

    # is the entity open
    open_entity = False

    # question id in the document
    question_i = 0

    with open(filename) as fin:
        lines = fin.readlines()

        for line in tqdm(lines):

            if "-DOCSTART-" in line:
                # new document is starting

                doc_id = line.split("(")[-1][:-2]

                # END DOCUMENT

                # check end of entity
                if open_entity:
                    document_questions[-1]["input"].append(END_ENT_TOKEN)
                    open_entity = False

                """
                #DEBUG
                for q in document_questions:
                    pp.pprint(q)
                    input("...")
                """

                # add sentence_questions to global_questions
                global_questions.extend(document_questions)

                # reset
                left_context = []
                document_questions = []
                question_i = 0

            else:
                split = line.split("\t")
                token = split[0].strip()

                if len(split) >= 5:
                    B_I = split[1]
                    mention = split[2]
                    #  YAGO2_entity = split[3]
                    Wikipedia_URL = split[4]
                    Wikipedia_ID = split[5]
                    # Freee_base_id = split[6]

                    if B_I == "I":
                        pass

                    elif B_I == "B":

                        title = Wikipedia_URL.split("/")[-1].replace("_", " ")

                        if Wikipedia_ID == "000":

                            if Wikipedia_URL in url2id_cache:
                                pageid = url2id_cache[Wikipedia_URL]
                            else:

                                pageid = _get_pageid_from_api(title)
                                url2id_cache[Wikipedia_URL] = pageid
                            Wikipedia_ID = pageid

                        q = {
                            "id": "{}:{}".format(doc_id, question_i),
                            "input": left_context.copy() + [BEGIN_ENT_TOKEN],
                            "mention": mention,
                            "Wikipedia_title": title,
                            "Wikipedia_URL": Wikipedia_URL,
                            "Wikipedia_ID": Wikipedia_ID,
                            "left_context": left_context.copy(),
                            "right_context": [],
                        }
                        document_questions.append(q)
                        open_entity = True
                        question_i += 1

                    else:
                        print("Invalid B_I {}", format(B_I))
                        sys.exit(-1)

                    # print(token,B_I,mention,Wikipedia_URL,Wikipedia_ID)
                else:
                    if open_entity:
                        document_questions[-1]["input"].append(END_ENT_TOKEN)
                        open_entity = False

                left_context.append(token)
                for q in document_questions:
                    q["input"].append(token)

                for q in document_questions[:-1]:
                    q["right_context"].append(token)

                if len(document_questions) > 0 and not open_entity:
                    document_questions[-1]["right_context"].append(token)

    # FINAL SENTENCE
    if open_entity:
        document_questions[-1]["input"].append(END_ENT_TOKEN)
        open_entity = False

    # add sentence_questions to global_questions
    global_questions.extend(document_questions)

    return global_questions


# store on file
def store_questions(questions, OUT_FILENAME):

    if not os.path.exists(os.path.dirname(OUT_FILENAME)):
        try:
            os.makedirs(os.path.dirname(OUT_FILENAME))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(OUT_FILENAME, "w+") as fout:
        for q in questions:
            json.dump(q, fout)
            fout.write("\n")


def convert_to_BLINK_format(questions):
    data = []
    for q in questions:
        datapoint = {
            "context_left": " ".join(q["left_context"]).strip(),
            "mention": q["mention"],
            "context_right": " ".join(q["right_context"]).strip(),
            "label_id": q["Wikipedia_ID"],
        }
        data.append(datapoint)
    return data

if __name__ == '__main__':
    args = get_args()
    os.makedirs(args.output_path, exist_ok=True)

    entity_dict, entity_map = load_entity_dict(args.document_path,args.output_path,args.max_ent_length,args.max_view_length,args.dataset)
    if args.dataset == "zeshel":
        convert_data(args.mention_path,args.output_path, entity_dict, entity_map, 'train',args.max_seq_length)
        convert_data(args.mention_path,args.output_path, entity_dict, entity_map, 'valid',args.max_seq_length)
        convert_data(args.mention_path,args.output_path, entity_dict, entity_map, 'test',args.max_seq_length)
    elif args.dataset == "wikipedia":

        # AIDA train/valid/test
        print("AIDA-YAGO2")
        in_aida_filename = os.path.join(args.mention_path,"AIDA/AIDA-YAGO2-dataset.tsv")
        aida_questions = extract_questions(in_aida_filename)

        train = []
        testa = []
        testb = []
        for element in aida_questions:
            if "testa" in element["id"]:
                testa.append(element)
            elif "testb" in element["id"]:
                testb.append(element)
            else:
                train.append(element)
        print("train: {}".format(len(train)))
        print("testa: {}".format(len(testa)))
        print("testb: {}".format(len(testb)))

        train_blink = convert_to_BLINK_format(train)
        testa_blink = convert_to_BLINK_format(testa)
        testb_blink = convert_to_BLINK_format(testb)

        out_train_aida_filename = os.path.join(args.output_path,"train.jsonl")
        store_questions(train_blink, out_train_aida_filename)
        out_testa_aida_filename = os.path.join(args.output_path,"valid.jsonl")
        store_questions(testa_blink, out_testa_aida_filename)
        out_testb_aida_filename = os.path.join(args.output_path,"test.jsonl")
        store_questions(testb_blink, out_testb_aida_filename)


        # ACE 2004
        print("ACE 2004")
        in_ace_filename = os.path.join(args.mention_path,"wned-datasets/ace2004/ace2004.conll")
        ace_questions = convert_to_BLINK_format(extract_questions(in_ace_filename))
        out_ace_filename = os.path.join(args.output_path,"ace2004_questions.jsonl")
        store_questions(ace_questions, out_ace_filename)
        print(len(ace_questions))


        # aquaint
        print("aquaint")
        in_aquaint_filename = os.path.join(args.mention_path,"wned-datasets/aquaint/aquaint.conll")
        aquaint_questions = convert_to_BLINK_format(extract_questions(in_aquaint_filename))
        out_aquaint_filename = os.path.join(args.output_path,"aquaint_questions.jsonl")
        store_questions(aquaint_questions, out_aquaint_filename)
        print(len(aquaint_questions))

        # clueweb - WNED-CWEB (CWEB) clueweb/clueweb.conll
        print("clueweb - WNED-CWEB (CWEB)")
        in_clueweb_filename = os.path.join(args.mention_path,"wned-datasets/clueweb/clueweb.conll")
        clueweb_questions = convert_to_BLINK_format(extract_questions(in_clueweb_filename))
        out_clueweb_filename = os.path.join(args.output_path,"clueweb_questions.jsonl")
        store_questions(clueweb_questions, out_clueweb_filename)
        print(len(clueweb_questions))


        # msnbc
        print("msnbc")
        in_msnbc_filename = os.path.join(args.mention_path,"wned-datasets/msnbc/msnbc.conll")
        msnbc_questions = convert_to_BLINK_format(extract_questions(in_msnbc_filename))
        out_msnbc_filename = os.path.join(args.output_path,"msnbc_questions.jsonl")
        store_questions(msnbc_questions, out_msnbc_filename)
        print(len(msnbc_questions))


        # wikipedia - WNED-WIKI (WIKI)
        print("wikipedia - WNED-WIKI (WIKI)")
        in_wnedwiki_filename = os.path.join(args.mention_path,"wned-datasets/wikipedia/wikipedia.conll")
        wnedwiki_questions = convert_to_BLINK_format(extract_questions(in_wnedwiki_filename))
        out_wnedwiki_filename = os.path.join(args.output_path,"wikipedia_questions.jsonl")
        store_questions(wnedwiki_questions, out_wnedwiki_filename)
        print(len(wnedwiki_questions))
