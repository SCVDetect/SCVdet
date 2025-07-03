import os
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import uutils.__utils__ as utls
from processing.dataprocessing import dataset

class GraphFunctionDataset(Dataset):
    def __init__(self, df, graph_dir, split='train', verbose=True):
        self.split = split
        if self.split == 'train':
            self.df = df[df['label'] == split]
            vuldf = self.df[self.df.vul == 1]
            nonvuldf = self.df[self.df.vul == 0].sample(len(vuldf), random_state=0) # 5 * 
            self.df = pd.concat([vuldf, nonvuldf])
            self.graph_dir = graph_dir
            self.graph_ids = []
        else: 
            self.df = df[df['label'] == split]
            vuldf = self.df[self.df.vul == 1]
            nonvuldf = self.df[self.df.vul == 0]#.sample(len(vuldf), random_state=0) # 5 * 
            self.df = pd.concat([vuldf, nonvuldf])
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


class MultiTaskGAT(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_heads, dropout, embedd_method):
        super().__init__()
        self.embedd_method = embedd_method  

        self.gat1 = dgl.nn.GATConv(in_feats, hidden_feats, num_heads, feat_drop=dropout, attn_drop=dropout)
        self.gat2 = dgl.nn.GATConv(hidden_feats * num_heads, hidden_feats, 1, feat_drop=dropout, attn_drop=dropout)

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

        # Match the dimension to emb
        func_emb = nn.functional.interpolate(func_emb.unsqueeze(0), size=emb.shape[1], mode='nearest').squeeze(0)
        rand_feat = nn.functional.interpolate(rand_feat.unsqueeze(0), size=emb.shape[1], mode='nearest').squeeze(0)

        h = torch.cat([rand_feat, func_emb, emb], dim=1)
        h = nn.Linear(h.shape[1], emb.shape[1]).to(h.device)(h)

        h = self.gat1(g, h).flatten(1)
        h = self.gat2(g, h).squeeze(1)
        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')

        node_logits = self.node_mlp(h)
        graph_logits = self.graph_mlp(hg)

        return node_logits, graph_logits    
      
class LearnableWeightedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))  # modify and add this param to the config
        self.ce = nn.CrossEntropyLoss()

    def forward(self, node_logits, node_labels, graph_logits, func_label):
        node_loss = self.ce(node_logits, node_labels)
        func_loss = self.ce(graph_logits.view(1, -1), func_label.view(1))
        loss = self.alpha * node_loss + (1 - self.alpha) * func_loss
        return loss
#+++++++++++++++++++++++++++++++++++++++++++


class LitSvulDetGAT(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.model = MultiTaskGAT(config['in_feats'], config['hidden_feats'], config['num_heads'], config['dropout'],
                                  config['embedd_method'])
        self.loss_fn = LearnableWeightedLoss()
        self.lr = config['lr']
        self.val_f1_history = []
        self.acc_history = []

        self.val_preds = []
        self.val_labels = []
        self.val_preds_nodes = []
        self.val_labels_nodes = []

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
        node_label = batch.ndata['_VULN'].long()
        func_label = batch.ndata['_FVULN'][0].long()

        pred_node = torch.argmax(node_logits, dim=1)
        pred = torch.argmax(graph_logits).item()

        self.val_preds.append(pred)
        self.val_labels.append(func_label.item())
        self.val_preds_nodes.append(pred_node.cpu())
        self.val_labels_nodes.append(node_label.cpu())

    def on_validation_epoch_end(self):
        preds = torch.tensor(self.val_preds)
        labels = torch.tensor(self.val_labels)
        preds_nodes = torch.cat(self.val_preds_nodes)
        labels_nodes = torch.cat(self.val_labels_nodes)
        
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        acc = accuracy_score(labels, preds)
        f1_node = f1_score(labels_nodes, preds_nodes, average="macro", zero_division=0)
        acc_node = accuracy_score(labels_nodes, preds_nodes)
        
        # history
        epoch = self.current_epoch
        self.val_f1_history.append((epoch, f1))
        self.acc_history.append((epoch, acc))
        
        # prog bar
        self.log("val_acc", acc, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)
        self.log("val_acc_node", acc_node, prog_bar=True)
        self.log("val_f1_node", f1_node, prog_bar=True)

        self.val_preds.clear()
        self.val_labels.clear()
        self.val_preds_nodes.clear()
        self.val_labels_nodes.clear()

    def on_test_epoch_end(self):
        node_preds = torch.cat(self.test_node_preds)
        node_labels = torch.cat(self.test_node_labels)
        func_preds = torch.tensor(self.test_func_preds)
        func_labels = torch.tensor(self.test_func_labels)
        
        func_pre = {"True_label": func_labels,
                    "Prediction": func_preds}
        node_pre = {"True_label": node_labels,
                    "Prediction": node_preds}
        func_pre = pd.DataFrame(func_pre)
        node_pre = pd.DataFrame(node_pre)
        func_pre.to_csv(f"{utls.outputs_dir()}/func_prediction.csv", index = False)
        node_pre.to_csv(f"{utls.outputs_dir()}/nodes_prediction.csv", index = False)
        
        # node level (macro)
        node_f1 = f1_score(node_labels, node_preds, average="macro", zero_division=0)
        node_acc = accuracy_score(node_labels, node_preds)
        node_precision = precision_score(node_labels, node_preds, average="macro", zero_division=0)
        node_recall = recall_score(node_labels, node_preds, average="macro", zero_division=0)
        node_auroc = roc_auc_score(node_labels, node_preds)  
        node_mcc = matthews_corrcoef(node_labels, node_preds)
        
        # Function-level (macro)
        func_f1 = f1_score(func_labels, func_preds, average="macro") # , zero_division=0
        func_acc = accuracy_score(func_labels, func_preds)
        func_precision = precision_score(func_labels, func_preds, average="macro") # , zero_division=0
        func_recall = recall_score(func_labels, func_preds, average="macro")# , zero_division=0
        func_auroc = roc_auc_score(func_labels, func_preds) 
        func_mcc = matthews_corrcoef(func_labels, func_preds)

        # Log metrics
        self.log_dict({
            'test_node_f1': node_f1,
            'test_node_acc': node_acc,
            'test_node_precision': node_precision,
            'test_node_recall': node_recall,
            'test_node_auroc': node_auroc,
            'test_node_mcc': node_mcc,
            'test_func_f1': func_f1,
            'test_func_acc': func_acc,
            'test_func_precision': func_precision,
            'test_func_recall': func_recall,
            'test_func_auroc': func_auroc,
            'test_func_mcc': func_mcc,
        }, prog_bar=True)

        # Save metrics to CSV
        metrics = {
            'Node F1': node_f1,
            'Node Accuracy': node_acc,
            'Node Precision': node_precision,
            'Node Recall': node_recall,
            'Node AUROC': node_auroc,
            'Node MCC': node_mcc,
            'Function F1': func_f1,
            'Function Accuracy': func_acc,
            'Function Precision': func_precision,
            'Function Recall': func_recall,
            'Function AUROC': func_auroc,
            'Function MCC': func_mcc
        }
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(f"{utls.outputs_dir()}/test_metrics.csv", mode='a', header=not os.path.exists(f"{utls.outputs_dir()}/test_metrics.csv"), index=False)

    def test_step(self, batch, batch_idx):
        node_logits, graph_logits = self(batch)
        node_labels = batch.ndata['_VULN'].long()
        func_label = batch.ndata['_FVULN'][0].long()
        node_preds = torch.argmax(node_logits, dim=1)
        func_pred = torch.argmax(graph_logits).item()
        

        if not hasattr(self, 'test_node_preds'):
            self.test_node_preds, self.test_node_labels = [], []
            self.test_func_preds, self.test_func_labels = [], []

        self.test_node_preds.append(node_preds)
        self.test_node_labels.append(node_labels)
        self.test_func_preds.append(func_pred)
        self.test_func_labels.append(func_label.item())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


#############
class Savebestconfig:
    def save_to_json(data, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True) 
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
    def load_from_json(file_path):
        with open(file_path, 'r') as json_file:
            return json.load(json_file)
        
def train_with_param_trials(df, graph_dir, config_grid):
    seed_everything(42)
    num_cpus = torch.get_num_threads()
    train_set = GraphFunctionDataset(df, graph_dir, split='train')
    val_set = GraphFunctionDataset(df, graph_dir, split='val')
    test_set = GraphFunctionDataset(df, graph_dir, split='test')

    train_loader = GraphDataLoader(train_set, batch_size= 1, shuffle=True, num_workers=num_cpus)
    val_loader = GraphDataLoader(val_set, batch_size=1,  num_workers=num_cpus)
    test_loader = GraphDataLoader(test_set, batch_size=1,  num_workers=num_cpus)

    final_model_path = f"{utls.outputs_dir()}/final_model_{str(config_grid['max_epochs'])}epochs.ckpt"
    
    if os.path.exists(final_model_path):
        best_val_f1 = 0
        best_model_path = None
        best_config = Savebestconfig.load_from_json(f"{utls.outputs_dir()}/best_confir.json")
    else: 
        best_val_f1 = 0
        best_model_path = None
        best_config = None
        

    trials = list(product(config_grid['hidden_feats'], config_grid['dropout'],  config_grid['lr']))
    print(f"Running {len(trials)} trials...")

    for idx, (hidden, dropout, lr, ) in enumerate(trials):
        print(f"\n=== Trial {idx+1} ===")
        trial_config = {
            'in_feats': config_grid['in_feats'],
            'hidden_feats': hidden,
            'num_heads': config_grid['num_heads'],
            'dropout': dropout, 
            'lr': lr,
            'embedd_method': config_grid['embedd_method'],
        }

        model = LitSvulDetGAT(trial_config)
        checkpoint_path = f"{utls.cache_dir()}/checkpoints/trial_{idx+1}.ckpt"

        checkpoint_callback = ModelCheckpoint(
            monitor= config_grid['check_monitor'], #   'val_f1', # val_acc, val_f1
            dirpath=os.path.dirname(checkpoint_path),
            filename=os.path.basename(checkpoint_path).replace('.ckpt', ''),
            save_top_k=1,
            mode='max',
            # save_weights_only=True   # --
        )

        early_stopping_callback = EarlyStopping(
            monitor= config_grid['check_monitor'], 
            patience= config_grid['check_patience'],
            mode='max'
        )

        trainer = Trainer(
            max_epochs= config_grid['max_epochs'],
            callbacks=[checkpoint_callback, early_stopping_callback],
            logger=False,
            enable_progress_bar=True
        )
        
        # new code to stop the trainin
        final_model_path = f"{utls.outputs_dir()}/final_model_{str(config_grid['max_epochs'])}epochs.ckpt"
        if not os.path.exists(final_model_path):
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
        else:  
            pass

    print(f"\nBest model: {best_model_path} with val_f1 = {best_val_f1:.4f}")
    
    print(f"Best config: {best_config}\n[Infos] Saved")
    # Save best Config
    Savebestconfig.save_to_json(best_config, f"{utls.outputs_dir()}/best_confir.json")

    print("\n=== Retraining Best Model for max Epochs ===")
    seed_everything(42)
    final_model = LitSvulDetGAT(best_config)
    final_trainer = Trainer(
        max_epochs= config_grid['max_epochs'], 
        logger=False,
        accelerator="auto"
    )
    
    

    # train model and test
    final_model_path = f"{utls.outputs_dir()}/final_model_{str(config_grid['max_epochs'])}epochs.ckpt"
    if os.path.exists(final_model_path):
        
        # Load and test final model
        print("[Infos]  Loading from pre-trained...")
        
        #  oldversion
        final_model = LitSvulDetGAT.load_from_checkpoint(final_model_path, config=best_config)
        final_trainer.test(final_model, test_loader)
        print("Testing complete. Training and testing history saved.")
    else:
        print("[Infos] Training")
        final_trainer.fit(final_model, train_loader, val_loader)
        # Save final model checkpoint
        final_model_path = f"{utls.outputs_dir()}/final_model_{str(config_grid['max_epochs'])}epochs.ckpt"
        final_trainer.save_checkpoint(final_model_path)
        print(f"Saved final model at: {final_model_path}")
        # Save final training history plot
        plt.plot(*zip(*final_model.val_f1_history))
        plt.xlabel("Epoch")
        plt.ylabel("Validation F1")
        plt.title(f"Final Training F1 History ({str(config_grid['max_epochs'])} Epochs)")
        plt.savefig(f"{utls.outputs_dir()}/final_model_f1_history.png")
        plt.close()
        # Load and test final model
        final_model = LitSvulDetGAT.load_from_checkpoint(final_model_path, config=best_config)
        final_trainer.test(final_model, test_loader)
        print("[Infos] Testing complete. Training and testing history saved.")



if __name__ == '__main__':
    df = dataset()
    
    # df = df[df['label'] == 'test']
    # df = df[df['id']==]
    
    # # use only same of data
    # df = df.sample(10000)  # remove later
    # graph_dir = f"{utls.cache_dir()}/Graph/dataset_svuldet_codebert_pdg+raw"
    
    
    config_grid = {
        "embedd_method": "Word2vec",  # can be # "Codebert", "Sbert", or "Word2vec"
        'max_epochs': 2, 
        'in_feats': 100, #768, #384, #100 change value for each  embedd_method.
        'check_patience': 2, 
        'batch_size':  1, 
        'num_heads': 4,
        'check_monitor':  'val_f1_node',# 'val_f1', # val_acc, or 'val_f1' , 'val_f1_node'
        'hidden_feats': [64], 
        'dropout': [0.4],
        'lr': [1e-5], # [1e-5,5e-5, 1e-4, 2e-4, 5e-4],
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
    
    best_model, best_config = train_with_param_trials(df, graph_dir, config_grid)
    
  
  
#   self.log("val_acc_node", acc_node, prog_bar=True)
#         self.log("val_f1_node", f1_node, prog_bar=True)