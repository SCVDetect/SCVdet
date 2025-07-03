import os
import sys
import json
from pathlib import Path
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from gensim.models import Word2Vec
import torchmetrics
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing.dataprocessing import dataset
import uutils.__utils__ as utls

# Dataset Class for Word2Vec
class datasetDatasetWord2Vec:
    def __init__(self, partition="train", random_labels=False):
        self.df = dataset()
        self.df = self.df[self.df.label == partition]

        if partition in ["train", "val"]:
            vul = self.df[self.df.vul == 1]
            nonvul = self.df[self.df.vul == 0]#.sample(len(vul), random_state=0)
            self.df = pd.concat([vul, nonvul])

        self.sentences = [sentence.split() for sentence in self.df.before.tolist()]
        self.labels = self.df.vul.tolist()
        if random_labels:
            self.labels = torch.randint(0, 2, (len(self.df),)).tolist()

        # Train Word2Vec model for embedding
        self.w2v_model = Word2Vec(sentences=self.sentences, vector_size=100, window=5, min_count=1, workers=4)
        self.embeddings = self.generate_embeddings(self.sentences)

    def generate_embeddings(self, sentences):
        embeddings = []
        for sentence in sentences:
            vector = np.mean([self.w2v_model.wv[word] for word in sentence if word in self.w2v_model.wv], axis=0)
            embeddings.append(vector if vector is not None else np.zeros(100))
        return torch.tensor(embeddings, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# PyTorch Lightning DataModule
class datasetDatasetWord2VecDataModule(pl.LightningDataModule):
    def __init__(self, DataClass, batch_size: int = 64):
        super().__init__()
        self.train = DataClass(partition="train")
        self.val = DataClass(partition="val")
        self.test = DataClass(partition="test")
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, batch_size=self.batch_size, num_workers=15)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=15)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=15)

# Lightning Module for Word2Vec
class LitWord2Vec(pl.LightningModule):
    def __init__(self, lr: float = 1e-4):
        super().__init__()
        self.lr = lr
        self.save_hyperparameters()
        self.fc1 = torch.nn.Linear(100, 64)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc2 = torch.nn.Linear(64, 2)
        self.accuracy = torchmetrics.Accuracy(task="binary")
        self.f1_score = torchmetrics.F1Score(task="binary")
        self.precision = torchmetrics.Precision(task="binary")
        self.roc_auc = torchmetrics.AUROC(task="binary")
        self.pr_auc = torchmetrics.AveragePrecision(task="binary")

    def forward(self, embeddings):
        x = F.relu(self.fc1(embeddings))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

    def training_step(self, batch, batch_idx):
        embeddings, labels = batch
        logits = self(embeddings)
        labels = labels.type(torch.LongTensor).to(self.device)
        loss = F.cross_entropy(logits, labels)

        preds = F.softmax(logits, dim=1).argmax(dim=1)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.accuracy(preds, labels), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        embeddings, labels = batch
        logits = self(embeddings)
        labels = labels.type(torch.LongTensor).to(self.device)
        loss = F.cross_entropy(logits, labels)

        preds = F.softmax(logits, dim=1).argmax(dim=1)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy(preds, labels), prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        embeddings, labels = batch
        logits = self(embeddings)
        labels = labels.type(torch.LongTensor).to(self.device)
        loss = F.cross_entropy(logits, labels)

        preds = F.softmax(logits, dim=1).argmax(dim=1)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.accuracy(preds, labels), prog_bar=True)
        self.log("test_f1", self.f1_score(preds, labels), prog_bar=True)
        self.log("test_precision", self.precision(preds, labels), prog_bar=True)
        self.log("test_roc_auc", self.roc_auc(logits[:, 1], labels), prog_bar=True)
        self.log("test_pr_auc", self.pr_auc(logits[:, 1], labels), prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    

# finetune and save word2vec model

if __name__ == "__main__":
    data_module = datasetDatasetWord2VecDataModule(datasetDatasetWord2Vec, batch_size=32)
    model = LitWord2Vec()

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1)
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=10,
        callbacks=[checkpoint_callback]
    )
    
    embedmodel = f"{utls.cache_dir()}/embedmodel/Word2vec"
    
    if not os.path.exists(embedmodel):
        os.makedirs(embedmodel)
        # finetune the word2vec
        trainer.fit(model, data_module)
        trainer.test(model, data_module)
        # Save Metrics to CSV
        metrics = trainer.callback_metrics
        metrics_dict = {key: float(value) for key, value in metrics.items() if isinstance(value, torch.Tensor)}
        # metrics_df = pd.DataFrame([metrics_dict])
        # metrics_df.to_csv(f"{utls.outputs_dir()}/word2vec_metrics.csv", index=False)
        # print(f"Metrics saved in {utls.outputs_dir()}/word2vec_metrics.csv")
        # Save Word2Vec Model
        os.makedirs(f"{embedmodel}/Word2vec")
        data_module.train.w2v_model.save(f"{embedmodel}/Word2vec/word2vec_model.bin")
        print("Word2Vec model saved to word2vec_model.bin")
    else:
        print("A pre-trained Word2vec exits \nLoading from Pre-trained ...")
        # Load Word2Vec Model for Embedding


# class Word2VecEmbedder: # review theis class : current; Word2VecEmbedder has no atribute M_word2vec
#     def __init__(self, model_path):
#         if os.path.exists(model_path):
#             print("Loarding Word2vec model for embedding from cahe ...")
#             self.M_word2vec = Word2Vec.load(model_path)
#         else:   
#             print("Check and finetune Word2vec Model for later use")
#     # /home/rz.lekeufack/Rosmael/SvulDet/sourcescripts/storage/cache/embedmodel
#     def embed(self, text):
#         tokens_word = text.split()
#         model_path = f"{utls.cache_dir()}/embedmodel/Word2vec/Word2vec/word2vec_model.bin"
#         self.M_word2vec = Word2Vec.load(model_path)
#         embedding = np.mean([self.M_word2vec.wv[word] for word in tokens_word if word in self.M_word2vec.wv], axis=0)
#         if embedding is None:
#             embedding = np.zeros(100)
#         return embedding
    

class Word2VecEmbedder:
    def __init__(self, model_path):
        if os.path.exists(model_path):
            print("Loading Word2Vec model for embedding from cache ...")
            self.M_word2vec = Word2Vec.load(model_path)
        else:   
            print("Check and fine-tune Word2Vec Model for later use")
            self.M_word2vec = None  
    def embed(self, text):
        # model_path = f"{utls.cache_dir()}/embedmodel/Word2vec/Word2vec/word2vec_model.bin"
        # self.M_word2vec = Word2Vec.load(model_path)
        if self.M_word2vec is None:
            raise ValueError("Word2Vec model is not loaded. Please check the model path.")
        tokens_word = text.split()
        
        if not tokens_word:  
            return np.zeros(100)  

        embeddings = [self.M_word2vec.wv[word] for word in tokens_word if word in self.M_word2vec.wv]
        
        if not embeddings:  
            return np.zeros(100)

        embedding = np.mean(embeddings, axis=0)

        if np.isnan(embedding).any():
            embedding = np.zeros(100)

        return embedding

# loaded_model = Word2Vec.load(f"{embedmodel}/Word2vec/word2vec_model.bin")
# text = """
# # Load Word2Vec Model for Embedding
# loaded_model = Word2Vec.load("word2vec_model.bin")
# text = "the pare de papa morarac"
# words = text.split()
# embedding = np.mean([loaded_model.wv[word] for word in words if word in loaded_model.wv], axis=0)
# if embedding is None:
#     embedding = np.zeros(100)  # Default to zero vector if no words match
# print("Generated embedding:", embedding)"""

# words = text.split()
# embedding = np.mean([loaded_model.wv[word] for word in words if word in loaded_model.wv], axis=0)
# if embedding is None:
#     embedding = np.zeros(100)  # Default to zero vector if no words match
# print("Generated embedding:", embedding)

# mword2vec = f"{utls.cache_dir()}/embedmodel/Word2vec/Word2vec/word2vec_model.bin"

# M_word2vec = Word2VecEmbedder(mword2vec)

# print(M_word2vec.embed(text))