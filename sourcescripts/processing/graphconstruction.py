"""The graph construction process is expected to be completed three times using different methods. 
This involves applying three embedding techniques: CodeBERT, Word2Vec, and SentenceBER"""

import os
import sys
from dgl import load_graphs, save_graphs
import dgl
import torch as th
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import uutils.__utils__ as utls
from embeddmodel.codebert import CodeBertEmbedder 
from embeddmodel.sentencebert import SBERTEmbedder
from embeddmodel.word2vec import Word2VecEmbedder
from dataprocessing import dataset, feature_extraction
from dataprocessing import get_dep_add_lines_dataset


def check_validity(_id):
    """Check whether sample with id=_id has enough node/edges."""
    try:
        with open(f"{utls.processed_dir()}/dataset/before/{_id}.java.nodes.json", "r") as f:
            nodes = json.load(f)
            lineNums = set()
            for n in nodes:
                if "lineNumber" in n:
                    lineNums.add(n["lineNumber"])
                    if len(lineNums) > 1:
                        break
            if len(lineNums) <= 1:
                return False

        with open(f"{utls.processed_dir()}/dataset/before/{_id}.java.edges.json", "r") as f:
            edges = json.load(f)
            edge_set = set([i[2] for i in edges])
            if "REACHING_DEF" not in edge_set and "CDG" not in edge_set:
                return False
        return True

    except Exception as E:
        print(f"[ERROR] {E} -- Skipped ID: {_id}")
        return False

df = dataset()
incorrect_list = []
ids = []
for _id in tqdm(df.id.tolist(), desc="Validating samples"):
    if check_validity(_id):
        ids.append(_id)

df = df[df['id'].isin(ids)]

def initialize_lines_and_features(gtype="pdg", feat="all"):
    lines = get_dep_add_lines_dataset()
    lines = {k: set(list(v["removed"]) + v["depadd"]) for k, v in lines.items()}
    return lines, gtype, feat

def cache_codebert_method_level(df, codebert, _id):
    savedir = utls.get_dir(utls.cache_dir() / "Graph/codebert_method_level") 
    batch_texts = df.before.tolist()
    texts = ["</s> " + ct for ct in batch_texts]
    embedded = codebert.embed(texts).detach().cpu()
    th.save(embedded, savedir / f"{_id}.pt")
    
def cache_sbert_method_level(df, sbert, _id):
    savedir = utls.get_dir(utls.cache_dir() / "Graph/sbert_method_level") 
    batch_texts = df.before.tolist()
    texts = ["</s> " + ct for ct in batch_texts]
    embedded = th.tensor(sbert.embed(texts)).detach().cpu()  # th.tensor(sbert.embed(batch_texts)).detach().cpu()
    th.save(embedded, savedir / f"{_id}.pt")
    
def cache_worde2vec_method_level(df, word2vec, _id):
    savedir = utls.get_dir(utls.cache_dir() / "Graph/word2vec_method_level") 
    batch_texts = df.before.tolist()
    texts = ["</s> " + ct for ct in batch_texts]
    embedded = th.tensor(word2vec.embed(texts[0])).detach().cpu() # word2vec.embed(text) 
    th.save(embedded, savedir / f"{_id}.pt")

def process_item(_id, df, codebert=None, word2vec=None, sbert=None, lines=None, graph_type="pdg", feat="all"):
    if codebert:
        savedir = utls.get_dir(utls.cache_dir() / f"Graph/dataset_svuldet_codebert_{graph_type}") / str(_id)
    elif word2vec:
        savedir = utls.get_dir(utls.cache_dir() / f"Graph/dataset_svuldet_word2vec_{graph_type}") / str(_id)
    elif sbert:
        savedir = utls.get_dir(utls.cache_dir() / f"Graph/dataset_svuldet_sbert_{graph_type}") / str(_id)
    else:
        savedir = utls.get_dir(utls.cache_dir() / f"Graph/dataset_svuldet_randfeat_{graph_type}") / str(_id)

    if os.path.exists(savedir):
        g = load_graphs(str(savedir))[0][0]
        return g

    code, lineno, ei, eo, et = feature_extraction(
        f"{utls.processed_dir()}/dataset/before/{_id}.java", graph_type)
    vuln = [1 if i in lines[_id] else 0 for i in lineno] if _id in lines else [0 for _ in lineno]

    g = dgl.graph((eo, ei))
    code = [c.replace("\\t", "").replace("\\n", "") for c in code]

    if codebert:
        features = [codebert.embed([c]).detach().cpu() for c in code]
        g.ndata["_CODEBERT"] = th.cat(features)

    if word2vec: # 
        features = [word2vec.embed(c) for c in code]
        features = np.array(features)
        g.ndata["_WORD2VEC"] = th.tensor(features).float()
       
    if sbert: 
        features = [sbert.embed([c]) for c in code]
        features = np.array(features)
        g.ndata["_SBERT"] = th.tensor(features).reshape(g.number_of_nodes(), 384).float() # tensor.reshape(3, 384)

    g.ndata["_RANDFEAT"] = th.rand(size=(g.number_of_nodes(), 100))
    g.ndata["_LINE"] = th.tensor(lineno).int()
    g.ndata["_VULN"] = th.tensor(vuln).float()
    g.ndata["_FVULN"] = g.ndata["_VULN"].max().repeat((g.number_of_nodes(),))
    g.edata["_ETYPE"] = th.tensor(et).long()

    if codebert:
        emb_path = utls.cache_dir() / f"Graph/codebert_method_level/{_id}.pt"
        g.ndata["_FUNC_EMB"] = th.load(emb_path).repeat((g.number_of_nodes(), 1))
    if sbert:
        emb_path = utls.cache_dir() / f"Graph/sbert_method_level/{_id}.pt"
        g.ndata["_FUNC_EMB"] = th.load(emb_path).repeat((g.number_of_nodes(), 1))
    if word2vec:
        emb_path = utls.cache_dir() / f"Graph/word2vec_method_level/{_id}.pt"
        g.ndata["_FUNC_EMB"] = th.load(emb_path).repeat((g.number_of_nodes(), 1))

    g = dgl.add_self_loop(g)
    save_graphs(str(savedir), [g])
    return g

def cache_all_items(df, lines, graph_type="pdg", feat="all"):
    embedders = {
        "codebert": CodeBertEmbedder(f"{utls.cache_dir()}/embedmodel/CodeBERT"),
        "word2vec": Word2VecEmbedder(f"{utls.cache_dir()}/embedmodel/Word2vec/Word2vec/word2vec_model.bin"),
        "sbert": SBERTEmbedder(f"{utls.cache_dir()}/embedmodel/SentenceBERT")
    }

    for name, embedder in embedders.items():
        print(f"\n---> Processing with {name.upper()} <---")
        for _id in tqdm(df.sample(len(df)).id.tolist()):
            try:
                process_item(
                    _id, df,
                    codebert=embedder if name == "codebert" else None,
                    word2vec=embedder if name == "word2vec" else None,
                    sbert=embedder if name == "sbert" else None,
                    lines=lines,
                    graph_type=graph_type,
                    feat=feat
                )
            except Exception as e:
                incorrect_list.append(_id)
                print(f"Error processing {_id} with {name}: {e}")
            

def delete_incorrect_element(incorrect_list, graph_type = "pdg+raw"):
    g_c_path = f"{utls.cache_dir()}/Graph/dataset_svuldet_codebert_{graph_type}"
    g_w_path = f"{utls.cache_dir()}/Graph/dataset_svuldet_word2vec_{graph_type}"
    g_s_path = f"{utls.cache_dir()}/Graph/dataset_svuldet_sbert_{graph_type}"
    
    gm_c_path = f"{utls.cache_dir()}/Graph/codebert_method_level"
    gm_s_path = f"{utls.cache_dir()}/Graph/sbert_method_level"
    gm_w_path = f"{utls.cache_dir()}/Graph/word2vec_method_level"
    
    path_dirs = [g_w_path, g_s_path, g_c_path]
    path_dirs_m = [gm_c_path, gm_s_path, gm_w_path]
    for e in path_dirs:
        for j in tqdm(incorrect_list):
            path = f"{e}/{j}.pt" 
            if not os.path.exists(path):
                # print(f"The path '{path}' does not exist.")  
                continue  
            try:
                if os.path.isfile(path):
                    os.remove(path)
                    print(f"File '{path}' has been deleted.")
                elif os.path.isdir(path):
                    shutil.rmtree(path)
                    print(f"Directory '{path}' has been deleted.")
            except Exception as e:
                print(f"Error deleting '{path}': {e}")
    for e in path_dirs_m:
        for j in tqdm(incorrect_list):
            path = f"{e}/{j}.pt" 
            if not os.path.exists(path):
                # print(f"The path '{path}' does not exist.")  
                continue  
            try:
                if os.path.isfile(path):
                    os.remove(path)
                    print(f"File '{path}' has been deleted.")
                elif os.path.isdir(path):
                    shutil.rmtree(path)
                    print(f"Directory '{path}' has been deleted.")
            except Exception as e:
                print(f"Error deleting '{path}': {e}")
    print("Deleted incorrect graphs")  
        
    


if __name__ == "__main__":
    lines, graph_type, feat = initialize_lines_and_features(gtype="pdg+raw", feat="all")

    codebert = CodeBertEmbedder(f"{utls.cache_dir()}/embedmodel/CodeBERT")
    sbert = SBERTEmbedder(f"{utls.cache_dir()}/embedmodel/SentenceBERT")
    word2vec = Word2VecEmbedder(f"{utls.cache_dir()}/embedmodel/Word2vec/Word2vec/word2vec_model.bin")

    _ids = df.id.tolist()     
    print("[Infos] Function-level feature generation")
    for _id in tqdm(_ids):
        try:
            if not os.path.exists(utls.cache_dir() / f"Graph/codebert_method_level/{_id}.pt"):
                cache_codebert_method_level(df[df.id == _id], codebert, _id)
            if not os.path.exists(utls.cache_dir() / f"Graph/sbert_method_level/{_id}.pt"):
                cache_sbert_method_level(df[df.id == _id], sbert, _id)
            if not os.path.exists(utls.cache_dir() / f"Graph/word2vec_method_level/{_id}.pt"):
                cache_worde2vec_method_level(df[df.id == _id], word2vec, _id)
        except Exception as e:
            print(f"Error creating method-level embedding for {_id}: {e}")
            
    # with line level
    cache_all_items(df, lines, graph_type, feat)
    
    # delete wrong graphs
    delete_incorrect_element(incorrect_list)
    print("[Infos] Graphs construction: Done.")
