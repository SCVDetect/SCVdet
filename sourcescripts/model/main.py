import os
import numpy as np
import sys
import json
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, WeightedRandomSampler
from dgl.dataloading import GraphDataLoader
import dgl
from tqdm import tqdm
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import (f1_score, accuracy_score, precision_score,
                             recall_score, roc_auc_score, matthews_corrcoef,
                             precision_recall_curve)
from itertools import product
import matplotlib.pyplot as plt
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import uutils.__utils__ as utls
from processing.dataprocessing import dataset
import GLconstruction as GLs  
import torch.nn.functional as F

class GraphFunctionDataset(Dataset):
    def __init__(self, df, graph_dir, split='train', verbose=True):
        self.split = split
        self.df = df[df['label'] == split]
        if verbose:
            print(f"[{split}] Vulnerable: {sum(self.df.vul == 1)}, Non-Vulnerable: {sum(self.df.vul == 0)}")

        
        # Balance classes only for training
        if self.split == 'train':
            vuldf = self.df[self.df.vul == 1]
            nonvuldf = self.df[self.df.vul == 0].sample(len(vuldf), random_state=0)
            self.df = pd.concat([vuldf, nonvuldf])
        
        self.graph_dir = graph_dir
        self.graph_ids = []
        self.node_weights = []  # For weighted sampling

        for graph_id in tqdm(self.df['id'].tolist(), desc=f"---> Checking graphs for {split}"):
            graph_path = os.path.join(self.graph_dir, f"{graph_id}")
            if os.path.exists(graph_path):
                self.graph_ids.append(graph_id)
                # Store node weights for sampling
                g = dgl.load_graphs(graph_path)[0][0]
                node_labels = g.ndata['_VULN'].long().cpu().numpy()
                self.node_weights.extend([1.0 if label == 1 else 0.1 for label in node_labels])  # Higher weight for vulnerable nodes

    def __len__(self):
        return len(self.graph_ids)

    def __getitem__(self, idx):
        graph_id = self.graph_ids[idx]
        graph_path = os.path.join(self.graph_dir, f"{graph_id}")
        g = dgl.load_graphs(graph_path)[0][0]
        vul_label = self.df[self.df['id'] == graph_id]['vul'].values[0]
        g.ndata['_FVULN'] = torch.tensor([vul_label] * g.num_nodes())
        return g

###--------------------
class LearnableWeightedLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))  
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
# class LearnableWeightedLoss(nn.Module):
#     def __init__(self, pos_weight=None):
#         super().__init__()
#         self.alpha = nn.Parameter(torch.tensor(0.5))  
#         self.pos_weight = pos_weight
#         self.ce = nn.CrossEntropyLoss(weight=self.pos_weight)

    def forward(self, node_logits, node_labels, graph_logits, func_label):
        node_loss = self.ce(node_logits, node_labels)
        func_loss = self.ce(graph_logits.view(1, -1), func_label.view(1))
        return self.alpha * node_loss + (1 - self.alpha) * func_loss

class MultiTaskGAT(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_heads, dropout,
                 embedd_method, glmethod, v1):
        super().__init__()
        self.embedd_method = embedd_method 
        self.glmethod = glmethod
        self.base_dropout = dropout
        self.v1 = nn.Parameter(torch.tensor(v1).float())
        
        # Temperature parameter for calibration
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        self.beta_mlp = nn.Sequential(
            nn.Linear(len(v1), 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid() 
        )

        self.gat1 = dgl.nn.GATConv(in_feats, hidden_feats, num_heads, 
                                   feat_drop=dropout, attn_drop=dropout)
        self.gat2 = dgl.nn.GATConv(hidden_feats * num_heads, hidden_feats, 1,
                                   feat_drop=dropout, attn_drop=dropout)

        # Node attention for graph-level prediction
        self.node_attn = nn.Linear(hidden_feats, 1)
        
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_feats, 2)
        )
        self.graph_mlp = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_feats, 2)
        )
        self.v1_projector = nn.Linear(len(v1), hidden_feats)

    def compute_beta(self):
        beta = self.beta_mlp(self.v1)
        return beta.squeeze()

    def forward(self, g):
        rand_feat = g.ndata['_RANDFEAT'].float()
        func_emb = g.ndata['_FUNC_EMB'].float()

        if self.embedd_method == "Codebert":
            emb = g.ndata['_CODEBERT'].float()
        elif self.embedd_method == "Word2vec":
            emb = g.ndata['_WORD2VEC'].float()
        elif self.embedd_method == "Sbert":
            emb = g.ndata['_SBERT'].float()
        else:
            raise ValueError(f"Unsupported embedding method: {self.embedd_method}")

        # Feature interpolation and concatenation
        func_emb = F.interpolate(func_emb.unsqueeze(0), size=emb.shape[1], mode='nearest').squeeze(0)
        rand_feat = F.interpolate(rand_feat.unsqueeze(0), size=emb.shape[1], mode='nearest').squeeze(0)
        h = torch.cat([rand_feat, func_emb, emb], dim=1)
        h = nn.Linear(h.shape[1], emb.shape[1]).to(h.device)(h)
        
        # Dynamic dropout
        beta = self.compute_beta()
        dynamic_dropout = self.base_dropout * (1 + beta.item())
        dynamic_dropout = min(dynamic_dropout, 0.8)
        self.gat1.feat_drop.p = dynamic_dropout
        self.gat1.attn_drop.p = dynamic_dropout
        self.gat2.feat_drop.p = dynamic_dropout
        self.gat2.attn_drop.p = dynamic_dropout

        # GAT layers
        h = self.gat1(g, h).flatten(1)
        h = self.gat2(g, h).squeeze(1)
        g.ndata['h'] = h

        # Node attention for graph-level prediction
        attn_weights = torch.sigmoid(self.node_attn(h))
        hg = torch.sum(h * attn_weights, dim=0)  # Weighted sum

        # Apply temperature scaling
        node_logits = self.node_mlp(h) / self.temperature
        graph_logits = self.graph_mlp(hg) / self.temperature

        return node_logits, graph_logits

class LitSvulDetGAT(LightningModule):
    def __init__(self, config, v1):
        super().__init__()
        self.save_hyperparameters()
        
        # Calculate class weights
        train_set = GraphFunctionDataset(dataset(), config['graph_dir'], 'train')
        node_labels = []
        for g in train_set:
            node_labels.extend(g.ndata['_VULN'].long().cpu().numpy())
        # vul_ratio = np.mean(node_labels)
        # pos_weight = torch.tensor([(1 - vul_ratio) / max(vul_ratio, 1e-8)])
        
        vul_ratio = np.mean(node_labels)
        class_weights = torch.tensor([
            1.0,  # non-vulnerable class
            (1 - vul_ratio) / max(vul_ratio, 1e-8)  # vulnerable class
            ], dtype=torch.float32)
        self.loss_fn = LearnableWeightedLoss(class_weights=class_weights)

        
        self.model = MultiTaskGAT(config['in_feats'], config['hidden_feats'], 
                                config['num_heads'], config['dropout'],
                                config['embedd_method'], config['glmethod'], v1)
        # self.loss_fn = LearnableWeightedLoss(pos_weight=pos_weight)
        self.lr = config['lr']
        self.val_f1_history = []
        self.acc_history = []
        self.optimal_threshold = 0.5  # Initialize with default
        
        # For threshold tuning
        self.val_probs = []
        self.val_labels = []

    def forward(self, g):
        return self.model(g)

    def training_step(self, batch, batch_idx):
        node_logits, graph_logits = self(batch)
        node_labels = batch.ndata['_VULN'].long()
        func_label = batch.ndata['_FVULN'][0].long()
        loss = self.loss_fn(node_logits, node_labels, graph_logits, func_label)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        node_logits, graph_logits = self(batch)
        node_labels = batch.ndata['_VULN'].long()
        func_label = batch.ndata['_FVULN'][0].long()

        # Store probabilities and labels for threshold tuning
        node_probs = torch.softmax(node_logits, dim=1)[:, 1]
        self.val_probs.extend(node_probs.cpu().numpy())
        self.val_labels.extend(node_labels.cpu().numpy())

        # Calculate metrics at current threshold
        pred_node = (node_probs > self.optimal_threshold).long()
        pred = torch.argmax(graph_logits).item()

        self.log_dict({
            'val_acc': accuracy_score([func_label.item()], [pred]),
            'val_f1': f1_score([func_label.item()], [pred], zero_division=0),
            'val_acc_node': accuracy_score(node_labels.cpu(), pred_node.cpu()),
            'val_f1_node': f1_score(node_labels.cpu(), pred_node.cpu(), zero_division=0),
        },  batch_size=batch_idx)

    def on_validation_epoch_end(self):
        # Find optimal threshold that maximizes F1
        if len(self.val_probs) > 0:
            precision, recall, thresholds = precision_recall_curve(
                self.val_labels, self.val_probs)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            self.optimal_threshold = thresholds[np.argmax(f1_scores)]
            self.log('optimal_threshold', self.optimal_threshold)
        
        self.val_probs.clear()
        self.val_labels.clear()

    def test_step(self, batch, batch_idx):
        node_logits, graph_logits = self(batch)
        node_labels = batch.ndata['_VULN'].long()
        func_label = batch.ndata['_FVULN'][0].long()
        
        # Use temperature-scaled probabilities
        node_probs = torch.softmax(node_logits, dim=1)[:, 1]
        node_preds = (node_probs > 0.5).long()  # Use fixed 0.5 threshold for testing
        func_pred = torch.argmax(graph_logits).item()

        return {
            'node_preds': node_preds,
            'node_labels': node_labels,
            'func_pred': func_pred,
            'func_label': func_label
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

#----------------------------------------
  
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

    @staticmethod
    def dependency_graph_construction(graph_list, config_grid):
        graph_embeddings = [
            GLs.compute_graph_embedding(g, method=config_grid['graph_to_vec_method']).numpy()
            for g in graph_list
        ]
        function_dependency_graph, cluster_labels = GLs.create_function_level_dependency_graph(
            embeddings=graph_embeddings,
            method=config_grid['cluster_method'],
            eps=0.8,
            min_samples=2,
            n_clusters=2,
            similarity_threshold=config_grid['cos_sim_threshold']
        )
        return function_dependency_graph, cluster_labels



class Savebestconfig:
    def save_to_json(data, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True) 
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
    def load_from_json(file_path):
        with open(file_path, 'r') as json_file:
            return json.load(json_file)
        

def get_or_compute_v1(df, config_grid, save_path=f"{utls.outputs_dir()}/v1.npy"):
   
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
        print(f"Loading v1 from {save_path}...")
        v1 = np.load(save_path)
    else:
        print(f"v1 not found. Computing and saving to {save_path}...")
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
        
def train_with_param_trials(df, graph_dir, config_grid):
    seed_everything(42)
    num_cpus = torch.get_num_threads()
    train_set = GraphFunctionDataset(df, graph_dir, split='train')
    val_set = GraphFunctionDataset(df, graph_dir, split='val')
    test_set = GraphFunctionDataset(df, graph_dir, split='test')

    train_loader = GraphDataLoader(train_set, batch_size=1, shuffle=True, num_workers=num_cpus)
    val_loader = GraphDataLoader(val_set, batch_size=1, num_workers=num_cpus)
    test_loader = GraphDataLoader(test_set, batch_size=1, num_workers=num_cpus)

    final_model_path = f"{utls.outputs_dir()}/final_model_{str(config_grid['max_epochs'])}epochs.ckpt"
    best_mo_path = f"{utls.outputs_dir()}/best_confir.json"

    if os.path.exists(best_mo_path):
        print("[Infos] Loading from pre-trained best config...")
        best_config = Savebestconfig.load_from_json(best_mo_path)
        best_model_path = f"{utls.cache_dir()}/checkpoints/best_model.ckpt"

        seed_everything(42)
        final_model = LitSvulDetGAT.load_from_checkpoint(best_model_path, config=best_config)
        final_trainer = Trainer(logger=False, enable_progress_bar=True)
        print("\n=== Testing the best model on test set ===")
        final_trainer.test(final_model, test_loader)
        return  

    best_val_f1 = 0
    best_model_path = None
    best_config = None

    trials = list(product(config_grid['hidden_feats'], config_grid['dropout'], config_grid['lr']))
    print(f"Running {len(trials)} trials...")

    v1 = get_or_compute_v1(df, config_grid)
    for idx, (hidden, dropout, lr) in enumerate(trials):
        print(f"\n=== Trial {idx+1} ===")
        # trial_config = {
        #     'in_feats': config_grid['in_feats'],
        #     'hidden_feats': hidden,
        #     'num_heads': config_grid['num_heads'],
        #     'dropout': dropout,
        #     'lr': lr,
        #     'embedd_method': config_grid['embedd_method'],
        #     'glmethod': config_grid['glmethod'],
        # }
        trial_config =  {
			'in_feats': config_grid['in_feats'],
			'hidden_feats': hidden,
			'num_heads': config_grid['num_heads'],
			'dropout': dropout,
			'lr': lr,
			'embedd_method': config_grid['embedd_method'],
			'glmethod': config_grid['glmethod'],
			'graph_dir': graph_dir  # <-- added to avoid KeyError
		}


        model = LitSvulDetGAT(trial_config, v1)
        checkpoint_path = f"{utls.cache_dir()}/checkpoints/trial_{idx+1}.ckpt"

        checkpoint_callback = ModelCheckpoint(
            monitor=config_grid['check_monitor'],  # 'val_f1'
            dirpath=os.path.dirname(checkpoint_path),
            filename=os.path.basename("best_model"),  
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
            enable_progress_bar=True
        )

        trainer.fit(model, train_loader, val_loader)

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
            best_model_path = checkpoint_callback.best_model_path
            best_config = trial_config

    print(f"\nBest model: {best_model_path} with val_f1 = {best_val_f1:.4f}")
    print(f"Best config: {best_config}\n[Infos] Saved")

    # Save best config
    Savebestconfig.save_to_json(best_config, best_mo_path)

    if best_model_path and os.path.exists(best_model_path):
        shutil.copy(best_model_path, f"{utls.cache_dir()}/checkpoints/best_model.ckpt")

    print("\n=== Testing the best model on test set ===")
    seed_everything(42)
    final_model = LitSvulDetGAT.load_from_checkpoint(best_model_path, config=best_config)
    final_trainer = Trainer(logger=False, enable_progress_bar=True)
    final_trainer.test(final_model, test_loader)



class FunctionLevelEvaluator:
    def __init__(self, model, config_grid):
        """
        Initialize the evaluator with model and configuration
        Args:
            model: The trained model (instance of LitSvulDetGAT)
            config_grid: Configuration dictionary containing parameters
        """
        self.model = model
        self.config_grid = config_grid
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
    def _get_graph_path(self):
        """Helper method to get graph path based on config"""
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
        """
        Predict function-level vulnerability based on node-level predictions
        """
        if not hasattr(self, 'test_graphs'):
            self.load_test_data()
            
        results = []
        
        for g, graph_id, true_label in zip(self.test_graphs, self.graph_ids, self.func_labels):
            with torch.no_grad():
                node_logits, _ = self.model(g.to(self.device))
                node_probs = torch.softmax(node_logits, dim=1)
                node_preds = (node_probs[:, 1] > threshold).long().cpu().numpy()
                node_labels = g.ndata['_VULN'].long().cpu().numpy()
                line_numbers = g.ndata['_LINE'].cpu().numpy()
                func_pred = 1 if np.any(node_preds == 1) else 0
                results.append({
                    'function_id': graph_id,
                    'prediction': func_pred,
                    'true_label': true_label,
                    'vulnerable_nodes_ratio': np.mean(node_preds),
                    'node_labels': node_labels.tolist(),       
                    'node_predictions': node_preds.tolist(),   
                    'line_numbers': line_numbers.tolist()    
                })
        
        return pd.DataFrame(results)
    
    def evaluate_function_level(self, threshold=0.5):
        """
        Evaluate function-level performance metrics
        Dictionary of evaluation metrics
        """
        pred_df = self.predict_function_level(threshold)
        y_true = pred_df['true_label'].values
        y_pred = pred_df['prediction'].values
        metrics = {
            'function_f1': f1_score(y_true, y_pred, average="macro", zero_division=0),
            'function_accuracy': accuracy_score(y_true, y_pred),
            'function_precision': precision_score(y_true, y_pred, average="macro", zero_division=0),
            'function_recall': recall_score(y_true, y_pred, average="macro", zero_division=0),
            'function_auroc': roc_auc_score(y_true, y_pred),
            'function_mcc': matthews_corrcoef(y_true, y_pred),
            'vulnerable_nodes_ratio_mean': pred_df['vulnerable_nodes_ratio'].mean()
        }
        pred_df.to_csv(f"{utls.outputs_dir()}/function_level_predictions.csv", index=False)
        pd.DataFrame([metrics]).to_csv(f"{utls.outputs_dir()}/function_level_metrics.csv", index=False)
        
        return metrics


if __name__ == '__main__':
    df = dataset()
    
    config_grid = {
        "embedd_method":  "Codebert",  # can be # "Codebert", "Sbert", or "Word2vec"
        'max_epochs': 3, #20, # 5, 3, #2, #10,  3 is good for now
        'in_feats': 768, #768, #384, #100 must e the same as the feature in graph 
        'check_patience': 2, # 10, 5
        'batch_size':  1, 
        'num_heads': 3, # 4
        'check_monitor': 'val_f1', # val_acc, 'val_f1' 
        'hidden_feats': [64], # , [64, 256]
        'lr': [5e-5], # [5e-5, 1e-4, 2e-4, 5e-4],
        'dropout': [0.3], #[0.1, 0.2, 0.3, 0.4, 0.5],
        # gl paramts
        "graph_to_vec_method": "GraphSAGE", # "GraphSAGE",  # or "Node2Vec"
        "cluster_method": "dbscan", #"dbscan",  # or "kmeans"
        "cos_sim_threshold": 0.06,
        "gl_vec_length": 100,
        "glmethod": "attention"  # can "attention" or "dependency"
    }
    
    def graph_path(param = config_grid):
        param = config_grid['embedd_method']
        if param == "Codebert":
            return f"{utls.cache_dir()}/Graph/dataset_svuldet_codebert_pdg+raw"
        elif param == "Word2vec":
            return f"{utls.cache_dir()}/Graph/dataset_svuldet_word2vec_pdg+raw" 
        elif param == "Sbert":
            return f"{utls.cache_dir()}/Graph/dataset_svuldet_sbert_pdg+raw" 
        else: 
            print(f"[Error] Provide a good embedding model name… is can be: 'Codebert', 'Sbert', or 'Word2vec'") 
            
    graph_dir = graph_path(config_grid)
    train_with_param_trials(df, graph_dir, config_grid)
    # best_model, best_config = train_with_param_trials(df, graph_dir, config_grid)
    
    best_mo_path = f"{utls.outputs_dir()}/best_confir.json"
    if os.path.exists(best_mo_path):
        best_config = Savebestconfig.load_from_json(best_mo_path)
        best_model_path = f"{utls.cache_dir()}/checkpoints/best_model.ckpt"
        final_model = LitSvulDetGAT.load_from_checkpoint(best_model_path, config=best_config)
        
        # Perform function-level evaluation
        evaluator = FunctionLevelEvaluator(final_model, config_grid)
        metrics = evaluator.evaluate_function_level()
        print("\n\n[INFO ] Function-level evaluation metrics dirived from nodes prediction:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
 
 