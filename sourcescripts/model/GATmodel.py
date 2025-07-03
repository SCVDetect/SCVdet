
import os
import sys
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
        self.df = df[df['label'] == split]
        vuldf = self.df[self.df.vul == 1]
        nonvuldf = self.df[self.df.vul == 0].sample(len(vuldf), random_state=0)
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
    def __init__(self, in_feats, hidden_feats, num_heads, dropout):
        super().__init__()
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
        codebert = g.ndata['_CODEBERT'].float()

        func_emb = nn.functional.interpolate(func_emb.unsqueeze(0), size=codebert.shape[1], mode='nearest').squeeze(0)
        rand_feat = nn.functional.interpolate(rand_feat.unsqueeze(0), size=codebert.shape[1], mode='nearest').squeeze(0)

        h = torch.cat([rand_feat, func_emb, codebert], dim=1)
        h = nn.Linear(h.shape[1], codebert.shape[1]).to(h.device)(h)

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
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.ce = nn.CrossEntropyLoss()

    def forward(self, node_logits, node_labels, graph_logits, func_label):
        node_loss = self.ce(node_logits, node_labels)
        func_loss = self.ce(graph_logits.view(1, -1), func_label.view(1))
        loss = self.alpha * node_loss + (1 - self.alpha) * func_loss
        return loss

class LitSvulDetGAT(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.model = MultiTaskGAT(config['in_feats'], config['hidden_feats'], config['num_heads'], config['dropout'])
        self.loss_fn = LearnableWeightedLoss()
        self.lr = config['lr']
        self.val_f1_history = []

        self.val_preds = []
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
        _, graph_logits = self(batch)
        func_label = batch.ndata['_FVULN'][0].long()
        pred = torch.argmax(graph_logits).item()
        self.val_preds.append(pred)
        self.val_labels.append(func_label.item())

    def on_validation_epoch_end(self):
        f1 = f1_score(self.val_labels, self.val_preds, zero_division=0)
        epoch = self.current_epoch
        self.val_f1_history.append((epoch, f1))
        self.log('val_f1', f1, prog_bar=True)
        self.val_preds.clear()
        self.val_labels.clear()

    def on_test_epoch_end(self):
        node_preds = torch.cat(self.test_node_preds)
        node_labels = torch.cat(self.test_node_labels)
        func_preds = torch.tensor(self.test_func_preds)
        func_labels = torch.tensor(self.test_func_labels)

        node_f1 = f1_score(node_labels, node_preds, average="macro", zero_division=0)
        node_acc = accuracy_score(node_labels, node_preds)
        node_precision = precision_score(node_labels, node_preds, average="macro", zero_division=0)
        node_recall = recall_score(node_labels, node_preds, average="macro", zero_division=0)
        node_auroc = roc_auc_score(node_labels, node_preds)  # Binary AUROC doesn't use average
        node_mcc = matthews_corrcoef(node_labels, node_preds)
        
        # Function-level (macro)
        func_f1 = f1_score(func_labels, func_preds, average="macro", zero_division=0)
        func_acc = accuracy_score(func_labels, func_preds)
        func_precision = precision_score(func_labels, func_preds, average="macro", zero_division=0)
        func_recall = recall_score(func_labels, func_preds, average="macro", zero_division=0)
        func_auroc = roc_auc_score(func_labels, func_preds)  # Binary AUROC
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

        self.test_node_preds.append(node_preds.cpu())
        self.test_node_labels.append(node_labels.cpu())
        self.test_func_preds.append(func_pred)
        self.test_func_labels.append(func_label.item())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def train_with_param_trials(df, graph_dir, config_grid, max_epochs=5):
    seed_everything(42)
    num_cpus = torch.get_num_threads()
    train_set = GraphFunctionDataset(df, graph_dir, split='train')
    val_set = GraphFunctionDataset(df, graph_dir, split='val')
    test_set = GraphFunctionDataset(df, graph_dir, split='test')

    train_loader = GraphDataLoader(train_set, batch_size=1, shuffle=True, num_workers=num_cpus)
    val_loader = GraphDataLoader(val_set, batch_size=1, num_workers=num_cpus)
    test_loader = GraphDataLoader(test_set, batch_size=1, num_workers=num_cpus)

    best_val_f1 = 0
    best_model_path = None
    best_config = None

    trials = list(product(config_grid['hidden_feats'], config_grid['dropout'], config_grid['lr']))
    print(f"Running {len(trials)} trials...")

    for idx, (hidden, dropout, lr) in enumerate(trials):
        print(f"\n=== Trial {idx+1} ===")
        trial_config = {
            'in_feats': config_grid['in_feats'],
            'hidden_feats': hidden,
            'num_heads': config_grid['num_heads'],
            'dropout': dropout,
            'lr': lr
        }

        model = LitSvulDetGAT(trial_config)
        checkpoint_path = f"{utls.cache_dir()}/checkpoints/trial_{idx+1}.ckpt"

        checkpoint_callback = ModelCheckpoint(
            monitor='val_f1',
            mode='max',
            save_top_k=1,
            dirpath=os.path.dirname(checkpoint_path),
            filename=os.path.basename(checkpoint_path).replace('.ckpt', '')
        )

        trainer = Trainer(
            max_epochs=max_epochs,
            callbacks=[checkpoint_callback, EarlyStopping(monitor='val_f1', patience=2, mode='max')],
            logger=False,
            accelerator="auto"
        )

        trainer.fit(model, train_loader, val_loader)

        # Plot and save F1 history for each trial
        plt.plot(*zip(*model.val_f1_history), label=f'Trial {idx+1}')
        plt.xlabel("Epoch")
        plt.ylabel("Validation F1")
        plt.title("Validation F1 Score History")
        plt.legend()
        plt.savefig(f"{utls.get_dir(f'{utls.outputs_dir()}/train_hystory')}/trial_{idx+1}_f1_history.png")
        plt.close()

        val_f1 = model.val_f1_history[-1][1] if model.val_f1_history else 0
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_path = checkpoint_path
            best_config = trial_config

    print(f"\nBest model: {best_model_path} with val_f1 = {best_val_f1:.4f}")
    print(f"Best config: {best_config}")

    best_model = LitSvulDetGAT.load_from_checkpoint(best_model_path, config=best_config)
    trainer = Trainer(logger=False)
    trainer.test(best_model, test_loader)
    best_configd = pd.DataFrame([best_config])
    best_configd.to_csv(f"{utls.outputs_dir()}/best_confir.csv", index = False)

    return best_model, best_config

if __name__ == '__main__':
    df = dataset()
    graph_dir = f"{utls.cache_dir()}/Graph/dataset_svuldet_codebert_pdg+raw"
    config_grid = {
        'in_feats': 768, # must e the same as the feature in graph 
        'num_heads': 4,
        'hidden_feats': [64], # , 256
        'dropout': [0.2, 0.3, 0.4],
        'lr': [1e-5],
    }
    best_model, best_config = train_with_param_trials(df, graph_dir, config_grid, max_epochs=100)
    
    #  {'in_feats': 768, 'hidden_feats': 64, 'num_heads': 4, 'dropout': 0.4, 'lr': 1e-05}
