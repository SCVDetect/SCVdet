import os
import sys
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses
from torchmetrics import Accuracy, F1Score, Precision, Recall, MatthewsCorrCoef
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing.dataprocessing import dataset
import uutils.__utils__ as utls


# Dataset Class for Sentence-BERT
class datasetDatasetSBERT:
    def __init__(self, partition="train", random_labels=False):
        self.df = dataset()
        self.df = self.df[self.df.label == partition]

        if partition in ["train", "val"]:
            vul = self.df[self.df.vul == 1]
            nonvul = self.df[self.df.vul == 0]#.sample(len(vul), random_state=0)
            self.df = pd.concat([vul, nonvul])

        self.texts = self.df.before.tolist()
        self.labels = self.df.vul.tolist()
        if random_labels:
            self.labels = torch.randint(0, 2, (len(self.df),)).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# PyTorch Lightning DataModule for SBERT
class datasetDatasetSBERTDataModule(pl.LightningDataModule):
    def __init__(self, DataClass, batch_size: int = 64):
        super().__init__()
        self.train = DataClass(partition="train")
        self.val = DataClass(partition="val")
        self.test = DataClass(partition="test")
        self.batch_size = batch_size

    def train_dataloader(self):
        """Load the dataset for training"""
        return DataLoader(self.train, shuffle=True, batch_size=self.batch_size, 
                          collate_fn=self.collate_fn, num_workers=15)

    def val_dataloader(self):
        """Load the dataset for validation"""
        return DataLoader(self.val, batch_size=self.batch_size, 
                          collate_fn=self.collate_fn, num_workers=15)

    def test_dataloader(self):
        """Load the dataset for testing"""
        return DataLoader(self.test, batch_size=self.batch_size, 
                          collate_fn=self.collate_fn, num_workers=15)

    @staticmethod
    def collate_fn(batch):
        texts, labels = zip(*batch)
        return list(texts), torch.tensor(labels, dtype=torch.long)

# Lightning Module for SBERT
class LitSBERT(pl.LightningModule):
    def __init__(self, model_name="all-MiniLM-L6-v2", lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = SentenceTransformer(model_name)
        self.fc = torch.nn.Linear(self.model.get_sentence_embedding_dimension(), 2)
        self.accuracy = Accuracy(task="binary")
        self.f1 = F1Score(task="binary")
        self.precision = Precision(task="binary")
        self.recall = Recall(task="binary")
        self.mcc = MatthewsCorrCoef(task="binary")
        self.lr = lr

    def forward(self, texts):
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        logits = self.fc(embeddings)
        return logits

    def training_step(self, batch, batch_idx):
        texts, labels = batch
        logits = self(texts)
        loss = F.cross_entropy(logits, labels)
        preds = logits.argmax(dim=1)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.accuracy(preds, labels), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        texts, labels = batch
        logits = self(texts)
        loss = F.cross_entropy(logits, labels)
        preds = logits.argmax(dim=1)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy(preds, labels), prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        texts, labels = batch
        logits = self(texts)
        loss = F.cross_entropy(logits, labels)
        preds = logits.argmax(dim=1)
        metrics = {
            "test_loss": loss,
            "test_acc": self.accuracy(preds, labels),
            "test_f1": self.f1(preds, labels),
            "test_precision": self.precision(preds, labels),
            "test_recall": self.recall(preds, labels),
            "test_mcc": self.mcc(preds, labels),
        }
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

 ##   Training the SBERT Model
 
 
if __name__ == "__main__":
    data_module = datasetDatasetSBERTDataModule(datasetDatasetSBERT, batch_size=64)
    model = LitSBERT()

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=10, # 10
        callbacks=[checkpoint_callback]
    )
    
    embedmodel = f"{utls.cache_dir()}/embedmodel/SentenceBERT"
    if not os.path.exists(embedmodel):
        os.makedirs(embedmodel)
        # finetune sentenceBert
        trainer.fit(model, data_module)
        trainer.test(model, data_module)
        metrics = trainer.callback_metrics
        metrics_dict = {key: [value.item()] for key, value in metrics.items() if isinstance(value, torch.Tensor)}
        # metrics_df = pd.DataFrame(metrics_dict)
        # metrics_df.to_csv(f"{utls.outputs_dir()}/sbert_test_metrics.csv", index=False)
        # print(f"Sentencebert metrics saved to {utls.outputs_dir()}/sbert_test_metrics.csv")
        # Save Model
        model.model.save(embedmodel)
    else:    
        print("A pre-trained SentenceBert  exit; \nLoading model from pre-trained")


class SBERTEmbedder: # error with sbert together with shape (384, ) (2, )
    def __init__(self, model_path):
        if os.path.exists(model_path):
            print("Loading SentenceBERT for embedding from cache ...")
            self.Sbert_model = SentenceTransformer(model_path)
        else:  
            print("Please check and finetune the model for later use.")
            
    def embed(self, text):
        embedding = self.Sbert_model.encode(text)
        return embedding
    
    
    

# example of using
    # Load Model for Text Embedding
    
# Sbert_model = SentenceTransformer(embedmodel)
    
# text = """# Load Word2Vec Model for Embedding
# loaded_model = Word2Vec.load("word2vec_model.bin")
# text = "the pare de papa morarac"
# words = text.split()
# embedding = np.mean([loaded_model.wv[word] for word in words if word in loaded_model.wv], axis=0)
# if embedding is None:
#     embedding = np.zeros(100)  # Default to zero vector if no words match
# print("Generated embedding:", embedding)"""
    
# embedmodel = f"{utls.cache_dir()}/embedmodel/SentenceBERT"
# embedding = SBERTEmbedder(model_path= embedmodel)

# ssss = embedding.embed(text)



# print(f"Text Embedding: {ssss}")