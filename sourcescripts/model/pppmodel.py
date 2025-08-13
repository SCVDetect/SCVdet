import os
import numpy as np
import sys
import json
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
from dgl.dataloading import GraphDataLoader
import dgl
from tqdm import tqdm
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import (f1_score, accuracy_score, precision_score,
                             recall_score, roc_auc_score, matthews_corrcoef)
from itertools import product
import matplotlib.pyplot as plt
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import uutils.__utils__ as utls
from processing.dataprocessing import dataset
import GLconstruction as GLs  
import torch.nn.functional as F
torch.set_float32_matmul_precision('medium')
from pytorch_lightning.strategies import DDPStrategy

class GraphFunctionDataset(Dataset):
    def __init__(self, df, graph_dir, split='train', verbose=True):
        self.split = split

        if self.split == 'train':
            self.df = df[df['label'] == split]
            if verbose:
                print(f"[{split}] Vulnerable: {sum(self.df.vul == 1)}, Non-Vulnerable: {sum(self.df.vul == 0)}")
            vuldf = self.df[self.df.vul == 1]
            nonvuldf = self.df[self.df.vul == 0].sample(len(vuldf), random_state=0) # 5 * 
            self.df = pd.concat([vuldf, nonvuldf])
            # self.df = self.df.sample(min(len(self.df), 2000))                           ########### -------------------------------------------------------------------------------------------->>>>
            self.graph_dir = graph_dir
            self.graph_ids = []
        else: 
            self.df = df[df['label'] == split]
            if verbose:
                print(f"[{split}] Vulnerable: {sum(self.df.vul == 1)}, Non-Vulnerable: {sum(self.df.vul == 0)}")
            vuldf = self.df[self.df.vul == 1]
            nonvuldf = self.df[self.df.vul == 0]#.sample(len(vuldf), random_state=0) # 5 * ########### -------------------------------------------------------------------------------------------->>>>
            self.df = pd.concat([vuldf, nonvuldf])
            # self.df = self.df.sample(min(len(self.df), 1500)) 
            self.graph_dir = graph_dir
            self.graph_ids = []

        for graph_id in tqdm(self.df['id'].tolist(), desc=f"---> Checking graphs for {split}"):
            graph_path = os.path.join(self.graph_dir, f"{graph_id}")
            if os.path.exists(graph_path):
                self.graph_ids.append(graph_id)

    def __len__(self):
        return len(self.graph_ids)

    def __getitem__(self, idx):
        graph_id = self.graph_ids[idx]
        graph_path = os.path.join(self.graph_dir, f"{graph_id}")
        g = dgl.load_graphs(graph_path)[0][0]
        vul_label = self.df[self.df['id'] == graph_id]['vul'].values[0]
        g.ndata['_FVULN'] = torch.tensor([vul_label] * g.num_nodes())
        return g
  
class GetlistGL:
    def __init__(self, df, config_grid, split='train'):
        self.param = config_grid['embedd_method']
        self.graph_dir = self.graph_path()
        self.dataset = GraphFunctionDataset(df, self.graph_dir, split)
        self.graph_list = []

    def get_graph_list(self):
        if not self.graph_list:  
            for idx in range(len(self.dataset)):
                g = self.dataset[idx]
                self.graph_list.append(g)
        return self.graph_list

    def graph_path(self):
        if self.param == "Codebert":
            return f"{utls.cache_dir()}/Graph/dataset_svuldet_codebert_pdg+raw"
        elif self.param == "Word2vec":
            return f"{utls.cache_dir()}/Graph/dataset_svuldet_word2vec_pdg+raw" 
        elif self.param == "Sbert":
            return f"{utls.cache_dir()}/Graph/dataset_svuldet_sbert_pdg+raw" 
        else: 
            raise ValueError("[Error] Provide a valid embedding model name: 'Codebert', 'Sbert', or 'Word2vec'")

    def dependency_graph_construction(graph_list, config_grid):
        graph_embeddings = [
            GLs.compute_graph_embedding(g, method=config_grid['graph_to_vec_method']).numpy()
            for g in graph_list
        ]
        gl, c_labels = GLs.create_function_level_dependency_graph(
            embeddings=graph_embeddings,
            method=config_grid['cluster_method'],
            eps=0.8,
            min_samples=2,
            n_clusters=2,
            similarity_threshold=config_grid['cos_sim_threshold']
        )
        return gl, c_labels

class MultiTaskGAT(nn.Module):
    def __init__(self, config, v1):
        super().__init__()
        self.embedd_method = config['embedd_method']
        self.glmethod = config['glmethod']
        self.base_dropout = config['dropout']
        self.v1 = nn.Parameter(torch.tensor(v1).float())  
        rand_dim = config['rand_feat_dim']
        femb_dim = config['func_emb_dim']
        emb_dim = config['embed_dim']
        hidden_feats = config['hidden_feats']
        num_heads = config['num_heads']

        self.rand_proj = nn.Linear(rand_dim, hidden_feats)
        self.femb_proj = nn.Linear(femb_dim, hidden_feats)
        self.emb_proj = nn.Linear(emb_dim, hidden_feats)

        self.node_comb_proj = nn.Linear(2 * hidden_feats, hidden_feats)
        self.func_comb_proj = nn.Linear(2 * hidden_feats, hidden_feats)

        self.gat1_node = dgl.nn.GATConv(hidden_feats, hidden_feats, num_heads,
                                        feat_drop=self.base_dropout, attn_drop=self.base_dropout)
        self.gat2_node = dgl.nn.GATConv(hidden_feats * num_heads, hidden_feats, 1,
                                        feat_drop=self.base_dropout, attn_drop=self.base_dropout)

        self.gat1_func = dgl.nn.GATConv(hidden_feats, hidden_feats, num_heads,
                                        feat_drop=self.base_dropout, attn_drop=self.base_dropout)
        self.gat2_func = dgl.nn.GATConv(hidden_feats * num_heads, hidden_feats, 1,
                                        feat_drop=self.base_dropout, attn_drop=self.base_dropout)
        
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Dropout(self.base_dropout),
            nn.Linear(hidden_feats, hidden_feats // 2),
            nn.ReLU(),
            nn.Dropout(self.base_dropout),
            nn.Linear(hidden_feats // 2, 2)
            )

        self.graph_mlp = nn.Sequential(
            nn.Linear(hidden_feats, 2)
            )

        self.v1_projector = nn.Linear(len(v1), hidden_feats)
        self.beta_mlp = nn.Sequential(
            nn.Linear(len(v1), 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def compute_beta(self):
        return self.beta_mlp(self.v1).squeeze()

    def forward(self, g):
        emb = g.ndata[self._select_emb()].float()
        func_emb = g.ndata['_FUNC_EMB'].float().detach()
        rand_feat = g.ndata['_RANDFEAT'].float()

        emb = self.emb_proj(emb)
        func_emb = self.femb_proj(func_emb)
        rand_feat = self.rand_proj(rand_feat)

        h_node = torch.cat([rand_feat, emb], dim=1)
        h_node = self.node_comb_proj(h_node)
        h_node = self.gat1_node(g, h_node).flatten(1)
        h_node = self.gat2_node(g, h_node).squeeze(1)
        g.ndata['h'] = h_node
        
        h_func = torch.cat([rand_feat, func_emb], dim=1)
        h_func = self.func_comb_proj(h_func)
        h_func = self.gat1_func(g, h_func).flatten(1)
        h_func = self.gat2_func(g, h_func).squeeze(1)
        g.ndata['h_func'] = h_func
        hg = dgl.mean_nodes(g, 'h_func')
        
        
        v1_proj = self.v1_projector(self.v1).to(h_node.device)
        if self.glmethod == "attention":
            v1_proj = v1_proj.unsqueeze(0)
            att_scores = torch.matmul(h_node, v1_proj.t()).squeeze(-1)
            g.ndata['att_score'] = att_scores
        elif self.glmethod == "dependency":
            v1_proj = v1_proj.unsqueeze(0).expand(h_node.shape[0], -1)
            g.ndata['dependency'] = v1_proj.detach()

        node_logits = self.node_mlp(h_node)
        graph_logits = self.graph_mlp(hg.detach())
        
        # print("node logits-------------------------\n", node_logits)
        # print("graph_logits -----------------------\n>", graph_logits)

        return node_logits, graph_logits

    def _select_emb(self):
        if self.embedd_method == "Codebert":
            return '_CODEBERT'
        elif self.embedd_method == "Word2vec":
            return '_WORD2VEC'
        elif self.embedd_method == "Sbert":
            return '_SBERT'
        else:
            raise ValueError(f"Unsupported embedding: {self.embedd_method}")


class LearnableWeightedLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, node_logits, node_labels, graph_logits, func_label):
        if self.pos_weight is not None:
            ce_node = nn.CrossEntropyLoss(weight=self.pos_weight.to(node_logits.device))
        else:
            ce_node = nn.CrossEntropyLoss()
        ce_func = nn.CrossEntropyLoss()

        node_loss = ce_node(node_logits, node_labels)
        func_loss = ce_func(graph_logits.view(1, -1), func_label.view(1))

        entropy = self.entropy_loss(node_logits)
        loss = (node_loss + func_loss/2 + 0.01 * entropy)
        return loss

    def entropy_loss(self, logits):
        p = torch.softmax(logits, dim=1)
        return -torch.mean(torch.sum(p * torch.log(p + 1e-8), dim=1))

def compute_node_class_weights(dataset):
    all_labels = []
    for g in dataset:
        all_labels.append(g.ndata['_VULN'].long())
    all_labels = torch.cat(all_labels)
    counts = torch.bincount(all_labels)
    if len(counts) < 2:
        return torch.tensor([1.0, 1.0])
    total = counts.sum().float()
    weights = total / (2 * counts.float())
    return weights

class LitSvulDetGAT(LightningModule):
    def __init__(self, config, v1, pos_weight=None):
        super().__init__()
        self.save_hyperparameters()
        self.model = MultiTaskGAT(config, v1)
        self.loss_fn = LearnableWeightedLoss(pos_weight)
        self.lr = config['lr']
        self.style = config['method'] 
        self.val_f1_history = []
        self.acc_history = []
        self.val_preds = []
        self.val_labels = []
        self.val_preds_nodes = []
        self.val_labels_nodes = []

    def forward(self, g):
        return self.model(g)

    def training_step(self, batch, batch_idx):
        """
        Adaptive training step
        """
        node_logits, graph_logits = self(batch)
        node_labels = batch.ndata['_VULN'].long()
        func_label = batch.ndata['_FVULN'][0].long()

        if self.style == "methodlevel":
            if (node_labels == 1).sum() > 0:
                ce_node = nn.CrossEntropyLoss(reduction='none')
                loss_per_node = ce_node(node_logits, node_labels)
                weights = torch.where(node_labels == 1, 1.0, 0.5).to(node_logits.device)
                node_loss = (loss_per_node * weights).mean()
                entropy = self.loss_fn.entropy_loss(node_logits)
            else:
                node_loss = torch.tensor(0.0, device=node_logits.device, requires_grad=True)
                entropy = torch.tensor(0.0, device=node_logits.device, requires_grad=True)

            ce_func = nn.CrossEntropyLoss()
            func_loss = ce_func(graph_logits.view(1, -1), func_label.view(1))
            loss = node_loss + func_loss + 0.01 * entropy

        elif self.style == "linelevel":
            if (node_labels == 1).sum() > 0:
                ce_node = nn.CrossEntropyLoss(reduction='none')
                loss_per_node = ce_node(node_logits, node_labels)
                weights = torch.where(node_labels == 1, 1.0, 0.5).to(node_logits.device)
                node_loss = (loss_per_node * weights).mean()
                entropy = self.loss_fn.entropy_loss(node_logits)
            else:
                node_loss = torch.tensor(0.0, device=node_logits.device, requires_grad=True)
                entropy = torch.tensor(0.0, device=node_logits.device, requires_grad=True)

            if self.current_epoch >= self.hparams.get("freeze_func_epochs", 500):
                ce_func = nn.CrossEntropyLoss()
                func_loss = ce_func(graph_logits.view(1, -1), func_label.view(1))
            else:
                func_loss = torch.tensor(0.0, device=graph_logits.device, requires_grad=True)

            loss = node_loss + func_loss + 0.01 * entropy

        else:
            loss = self.loss_fn(node_logits, node_labels, graph_logits, func_label)
            node_loss = torch.tensor(0.0, device=node_logits.device)
            func_loss = torch.tensor(0.0, device=node_logits.device)
            entropy = torch.tensor(0.0, device=node_logits.device)

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_loss_node", node_loss, prog_bar=True, sync_dist=True)
        self.log("train_loss_func", func_loss, prog_bar=True, sync_dist=True)
        # self.log("train_entropy", entropy, prog_bar=False, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        node_logits, graph_logits = self(batch)
        node_labels = batch.ndata['_VULN'].long()
        func_label = batch.ndata['_FVULN'][0].long()

        temperature = 0.7
        node_probs = torch.softmax(node_logits / temperature, dim=1)
        pred_node = (node_probs[:, 1] >= 0.5).long()
        pred = torch.argmax(graph_logits).item()

        self.val_preds.append(pred)
        self.val_labels.append(func_label.item())
        self.val_preds_nodes.append(pred_node.cpu())
        self.val_labels_nodes.append(node_labels.cpu())

    def on_validation_epoch_end(self):
        preds = torch.tensor(self.val_preds)
        labels = torch.tensor(self.val_labels)
        preds_nodes = torch.cat(self.val_preds_nodes)
        labels_nodes = torch.cat(self.val_labels_nodes)

        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        acc = accuracy_score(labels, preds)
        f1_node = f1_score(labels_nodes, preds_nodes, average="macro", zero_division=0)
        acc_node = accuracy_score(labels_nodes, preds_nodes)

        epoch = self.current_epoch
        self.val_f1_history.append((epoch, f1))
        self.acc_history.append((epoch, acc))

        self.log('val_f1', f1, prog_bar=True, sync_dist=True)
        self.log("val_f1_node", f1_node, prog_bar=True, sync_dist=True)

        self.val_preds.clear()
        self.val_labels.clear()
        self.val_preds_nodes.clear()
        self.val_labels_nodes.clear()

    def test_step(self, batch, batch_idx):
        node_logits, graph_logits = self(batch)
        node_labels = batch.ndata['_VULN'].long()
        func_label = batch.ndata['_FVULN'][0].long()

        node_probs = torch.softmax(node_logits, dim=1)
        node_preds = (node_probs[:, 1] >= 0.5).long()
        func_pred = torch.argmax(graph_logits).item()

        if not hasattr(self, 'test_node_preds'):
            self.test_node_preds, self.test_node_labels = [], []
            self.test_func_preds, self.test_func_labels = [], []

        self.test_node_preds.append(node_preds.cpu())
        self.test_node_labels.append(node_labels.cpu())
        self.test_func_preds.append(func_pred)
        self.test_func_labels.append(func_label.item())

    def on_test_epoch_end(self):
        node_preds = torch.cat(self.test_node_preds)
        node_labels = torch.cat(self.test_node_labels)
        func_preds = torch.tensor(self.test_func_preds)
        func_labels = torch.tensor(self.test_func_labels)

        func_pre = pd.DataFrame({"True_label": func_labels, "Prediction": func_preds})
        node_pre = pd.DataFrame({"True_label": node_labels, "Prediction": node_preds})
        func_pre.to_csv(f"{utls.outputs_dir()}/func_prediction.csv", index=False)
        node_pre.to_csv(f"{utls.outputs_dir()}/nodes_prediction.csv", index=False)  

        node_f1 = f1_score(node_labels, node_preds, average="macro", zero_division=0)
        node_acc = accuracy_score(node_labels, node_preds)
        node_precision = precision_score(node_labels, node_preds, average="macro", zero_division=0)
        node_recall = recall_score(node_labels, node_preds, average="macro", zero_division=0)

        func_f1 = f1_score(func_labels, func_preds, average="macro", zero_division=0)
        func_acc = accuracy_score(func_labels, func_preds)
        func_precision = precision_score(func_labels, func_preds, average="macro", zero_division=0)
        func_recall = recall_score(func_labels, func_preds, average="macro", zero_division=0)
        func_auroc = roc_auc_score(func_labels, func_preds)
        func_mcc = matthews_corrcoef(func_labels, func_preds)
        
        self.log_dict({
            'test_node_f1': node_f1,
            'test_node_acc': node_acc,
            'test_node_precision': node_precision,
            'test_node_recall': node_recall,
            'test_func_f1': func_f1,
            'test_func_acc': func_acc,
            'test_func_precision': func_precision,
            'test_func_recall': func_recall,
        }, prog_bar=True, sync_dist=True)

        metrics = {
            'Node F1': f"{node_f1:.5f}",
            'Node Accuracy': f"{node_acc:.5f}",
            'Node Precision': f"{node_precision:.5f}",
            'Node Recall': f"{node_recall:.5f}",
            'Function F1': f"{func_f1:.5f}",
            'Function Accuracy': f"{func_acc:.5f}",
            'Function Precision': f"{func_precision:.5f}",
            'Function Recall': f"{func_recall:.5f}",
        }
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(
            f"{utls.outputs_dir()}/test_metrics.csv",
            mode='a',
            header=not os.path.exists(f"{utls.outputs_dir()}/test_metrics.csv"),
            index=False
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class Savebestconfig:
    def save_to_json(data, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True) 
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
    def load_from_json(file_path):
        with open(file_path, 'r') as json_file:
            return json.load(json_file)
        

def get_or_compute_v1(df, config_grid, save_path=f"{utls.cache_dir()}/v1.npy"):
   
    def replace_nan_with_average(vec):
        nan_mask = np.isnan(vec)
        if np.all(nan_mask):
            return np.zeros_like(vec)
        elif np.any(nan_mask):
            mean_val = np.nanmean(vec)
            result = vec.copy()
            result[nan_mask] = mean_val
            return result
        return vec   

    if os.path.exists(save_path):
        v1 = np.load(save_path)
    else:
        gl_loader = GetlistGL(df, config_grid, split='train')
        graph_list = gl_loader.get_graph_list()
        gl, _ = GetlistGL.dependency_graph_construction(graph_list, config_grid)
        
        glnx = GLs.networkx_to_dgl(
            nx_graph=gl,
            feature_size=config_grid['gl_vec_length'],
            device='cpu'
        )
        v1 = GLs.compute_graph_embedding(
            glnx,
            method=config_grid['graph_to_vec_method']
        ).cpu().numpy()

        v1 = replace_nan_with_average(v1)
        np.save(save_path, v1)
    return v1

class FunctionLevelEvaluator:
    def __init__(self, model, config_grid):
        self.model = model
        self.config_grid = config_grid
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
    def _get_graph_path(self):
        
        param = self.config_grid['embedd_method']
        if param == "Codebert":
            return f"{utls.cache_dir()}/Graph/dataset_svuldet_codebert_pdg+raw"
        elif param == "Word2vec":
            return f"{utls.cache_dir()}/Graph/dataset_svuldet_word2vec_pdg+raw" 
        elif param == "Sbert":
            return f"{utls.cache_dir()}/Graph/dataset_svuldet_sbert_pdg+raw" 
        else: 
            raise ValueError("Provide a valid embedding model name: 'Codebert', 'Sbert', or 'Word2vec'")
        
    def load_test_data(self):
        """Load test graphs and their ground truth labels"""
        df = dataset()
        dftest = df[df['label'] == 'test']
        gpath = self._get_graph_path()
        
        self.test_graphs = []
        self.graph_ids = []
        self.func_labels = []
        
        for graph_id in dftest['id'].tolist():
            try:
                graph_path = os.path.join(gpath, f"{graph_id}")
                if os.path.exists(graph_path):
                    g = dgl.load_graphs(graph_path)[0][0]
                    self.test_graphs.append(g)
                    self.graph_ids.append(graph_id)
                    self.func_labels.append(dftest[dftest['id'] == graph_id]['vul'].values[0])
            except Exception as e:
                print(f"Error loading graph {graph_id}: {e}")
    
    def predict_function_level(self, threshold=0.5):

        if not hasattr(self, 'test_graphs'):
            self.load_test_data()
            
        results = []
        
        for g, graph_id, true_label in zip(self.test_graphs, self.graph_ids, self.func_labels):
            with torch.no_grad():
                node_logits, func_logits = self.model(g.to(self.device))
                func_pred_la = torch.argmax(func_logits).item()
                node_probs = torch.softmax(node_logits, dim=1)
                node_preds = (node_probs[:, 1] > threshold).long().cpu().numpy()
                node_labels = g.ndata['_VULN'].long().cpu().numpy()
                line_numbers = g.ndata['_LINE'].cpu().numpy()
                func_pred = 1 if np.any(node_preds == 1) else 0
                results.append({
                    'function_id': graph_id,
                    'true_label': true_label,
                    'func_level_pred': func_pred_la,
                    'prediction': func_pred,
                    'vulnerable_nodes_ratio': np.mean(node_preds),
                    'node_labels': node_labels.tolist(),       
                    'node_predictions': node_preds.tolist(),   
                    'line_numbers': line_numbers.tolist()    
                })
        
        return pd.DataFrame(results)
    
    def evaluate_function_level(self, threshold=0.5):

        pred_df = self.predict_function_level(threshold)
        y_true = pred_df['true_label'].values
        y_pred = pred_df['func_level_pred'].values
        metrics = {
            'function_accuracy': accuracy_score(y_true, y_pred),
            'function_precision': precision_score(y_true, y_pred, average="macro", zero_division=0),
            'function_f1': f1_score(y_true, y_pred, average="macro", zero_division=0),
            'function_recall': recall_score(y_true, y_pred, average="macro", zero_division=0),
            # 'function_auroc': roc_auc_score(y_true, y_pred),
            # 'function_mcc': matthews_corrcoef(y_true, y_pred),
            'vulnerable_nodes_ratio_mean': pred_df['vulnerable_nodes_ratio'].mean()
        }
        pred_df.to_csv(f"{utls.outputs_dir()}/function_level_predictions.csv", index=False)
        pd.DataFrame([metrics]).to_csv(f"{utls.outputs_dir()}/function_level_metrics.csv", index=False)
        
        return metrics
 
 
def train_with_param_trials(df, graph_dir, config_grid):
    seed_everything(123)
    num_cpus = torch.get_num_threads() - 2 if torch.get_num_threads() > 2 else torch.get_num_threads()
    train_set = GraphFunctionDataset(df, graph_dir, split='train')
    val_set = GraphFunctionDataset(df, graph_dir, split='val')
    test_set = GraphFunctionDataset(df, graph_dir, split='test')

    train_loader = GraphDataLoader(train_set, batch_size=1, shuffle=True, num_workers=num_cpus)
    val_loader = GraphDataLoader(val_set, batch_size=1, num_workers=num_cpus)
    test_loader = GraphDataLoader(test_set, batch_size=1, num_workers=num_cpus)

    best_mo_path = f"{utls.outputs_dir()}/best_confir.json"
    best_model_path = f"{utls.cache_dir()}/checkpoints/best_model.ckpt"

    if os.path.exists(best_mo_path) and os.path.exists(best_model_path):
        print("[Infos] Loading from pre-trained best config...")
        best_config = Savebestconfig.load_from_json(best_mo_path)
        final_model = LitSvulDetGAT.load_from_checkpoint(best_model_path, config=best_config)
        final_trainer = Trainer(logger=False, enable_progress_bar=True)
        print("\n=== Testing the best model on test set ===")
        final_trainer.test(final_model, test_loader)

        evaluator = FunctionLevelEvaluator(final_model, config_grid)
        metrics = evaluator.evaluate_function_level()
        print("\n\n[INFO] Function-level evaluation metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        return

    # No pre-trained model, run trials
    best_val_f1 = 0
    best_config = None
    trials = list(product(config_grid['hidden_feats'], config_grid['dropout'], config_grid['lr']))
    print(f"Running {len(trials)} trials...")

    v1 = get_or_compute_v1(df, config_grid)
    best_model = None

    for idx, (hidden, dropout, lr) in enumerate(trials):
        print(f"\n=== Trial {idx+1} ===")
        trial_config = {
            'in_feats': config_grid['in_feats'],
            'hidden_feats': hidden,
            'num_heads': config_grid['num_heads'],
            'dropout': dropout,
            'lr': lr,
            'embedd_method': config_grid['embedd_method'],
            'glmethod': config_grid['glmethod'],
            'rand_feat_dim': 100,
            'func_emb_dim': config_grid['in_feats'],
            'embed_dim': config_grid['in_feats'],
            'method': config_grid['method']
        }
        node_pos_weight = compute_node_class_weights(train_set)
        model = LitSvulDetGAT(trial_config, v1, node_pos_weight)

        checkpoint_path = f"{utls.cache_dir()}/checkpoints/trial_{idx+1}.ckpt"
        checkpoint_callback = ModelCheckpoint(
            monitor=config_grid['check_monitor'],
            dirpath=os.path.dirname(checkpoint_path),
            filename="best_model",
            save_top_k=1,
            mode='max',
        )

        early_stopping_callback = EarlyStopping(
            monitor=config_grid['check_monitor'],
            patience=config_grid['check_patience'],
            mode='max'
        )

        trainer = Trainer(
            max_epochs=config_grid['max_epochs'],
            callbacks=[checkpoint_callback, early_stopping_callback],
            logger=False,
            enable_progress_bar=True,
            strategy=DDPStrategy(find_unused_parameters=True)
        )
        
        trainer.fit(model, train_loader, val_loader)

        # Save metric history
        plt.plot(*zip(*model.val_f1_history), label=f'Trial {idx+1} - F1', marker='o')
        plt.plot(*zip(*model.acc_history), label=f'Trial {idx+1} - Accuracy', marker='s')
        plt.xlabel("Epoch")
        plt.ylabel("Validation Metric")
        plt.title("Validation Metric Score History")
        plt.legend()
        plt.savefig(f"{utls.get_dir(f'{utls.outputs_dir()}/train_hystory')}/trial_{idx+1}_metrics_history.png")
        plt.close()

        val_f1 = model.val_f1_history[-1][1] if model.val_f1_history else 0
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = model
            best_config = trial_config
            # Save best checkpoint
            if os.path.exists(checkpoint_callback.best_model_path):
                shutil.copy(checkpoint_callback.best_model_path, best_model_path)

    print(f"\n[INFO] Best model saved to: {best_model_path} with val_f1 = {best_val_f1:.4f}")
    print(f"[INFO] Best config: {best_config}")

    Savebestconfig.save_to_json(best_config, best_mo_path)

    # Test best model (in memory, not reloaded)
    print("\n=== Testing the best model on test set ===")
    seed_everything(123)
    final_trainer = Trainer(logger=False, enable_progress_bar=True)
    final_trainer.test(best_model, test_loader)

    # Run FunctionLevelEvaluator using final in-memory model
    evaluator = FunctionLevelEvaluator(best_model, config_grid)
    metrics = evaluator.evaluate_function_level()
    print("\n\n[INFO] Function-level evaluation metrics derived from node prediction:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == '__main__':
    df = dataset()
    config_grid = {
        "method":"linelevel", # 'linelevel', ## methodlevel, "linelevel",
        "embedd_method":  "Codebert", # "Codebert", "Sbert", or "Word2vec"
        'max_epochs': 50,
        'in_feats': 768,   #768, #384, #100 must e the same as the feature in graph 
        'check_patience': 20, # 2
        'batch_size':  "batch_idx",
        'num_heads': 4,
        'check_monitor':    'val_f1_node',         #'val_f1',  #'val_f1_node',
        'hidden_feats': [525],
        'lr': [1e-5],
        'dropout': [0.4],
        "graph_to_vec_method": "GraphSAGE",
        "cluster_method": "dbscan",
        "cos_sim_threshold": 0.06,
        "gl_vec_length": 100,
        "glmethod": "attention" # can "attention" or "dependency"
    }

    def graph_path(param=config_grid):
        method = config_grid['embedd_method']
        if method == "Codebert":
            return f"{utls.cache_dir()}/Graph/dataset_svuldet_codebert_pdg+raw"
        elif method == "Word2vec":
            return f"{utls.cache_dir()}/Graph/dataset_svuldet_word2vec_pdg+raw"
        elif method == "Sbert":
            return f"{utls.cache_dir()}/Graph/dataset_svuldet_sbert_pdg+raw"
        else:
            print("[Error] Provide a valid embedding model name…")

    graph_dir = graph_path(config_grid)
    train_with_param_trials(df, graph_dir, config_grid)
