
import os
import sys
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
import torchmetrics

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing.dataprocessing import dataset
import uutils.__utils__ as utls

# Dataset Class for CodeBERT
class datasetDatasetNLP:
    def __init__(self, partition="train", random_labels=False):
        self.df = dataset()
        self.df = self.df[self.df.label == partition]

        if partition in ["train", "val"]:
            vul = self.df[self.df.vul == 1]
            nonvul = self.df[self.df.vul == 0]#.sample(len(vul), random_state=0)
            self.df = pd.concat([vul, nonvul])

        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        tk_args = {"padding": True, "truncation": True, "return_tensors": "pt"}
        text = [tokenizer.sep_token + " " + ct for ct in self.df.before.tolist()]
        tokenized = tokenizer(text, **tk_args)

        self.labels = self.df.vul.tolist()
        if random_labels:
            self.labels = torch.randint(0, 2, (len(self.df),)).tolist()
        self.ids = tokenized["input_ids"]
        self.att_mask = tokenized["attention_mask"]  

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.ids[idx], self.att_mask[idx], self.labels[idx]


class datasetDatasetNLPDataModule(pl.LightningDataModule):
    def __init__(self, DataClass, batch_size: int = 64):
        super().__init__()
        self.train = DataClass(partition="train")
        self.val = DataClass(partition="val")
        self.test = DataClass(partition="test")
        self.batch_size = batch_size

    def train_dataloader(self):
        """Load the dataset for training"""
        return DataLoader(self.train, shuffle=True, batch_size=self.batch_size, num_workers=15)

    def val_dataloader(self):
        """Load the dataset for validation"""
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=15)

    def test_dataloader(self):
        """Load the dataset for testting"""
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=15)

# Lightning Module for CodeBERT
class LitCodeBERT(pl.LightningModule):
    def __init__(self, lr: float = 1e-4):
        super().__init__()
        self.lr = lr
        self.save_hyperparameters()
        self.bert = AutoModel.from_pretrained("microsoft/codebert-base")
        self.fc1 = torch.nn.Linear(768, 256)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 2)
        self.accuracy = torchmetrics.Accuracy(task="binary")
        self.f1_score = torchmetrics.F1Score(task="binary")
        self.precision = torchmetrics.Precision(task="binary")
        self.roc_auc = torchmetrics.AUROC(task="binary")
        self.pr_auc = torchmetrics.AveragePrecision(task="binary")

    def forward(self, ids, mask):
        bert_out = self.bert(ids, attention_mask=mask).pooler_output
        fc1_out = F.relu(self.fc1(bert_out))
        fc1_out = self.dropout(fc1_out)
        fc2_out = F.relu(self.fc2(fc1_out))
        fc3_out = self.fc3(fc2_out)
        return fc3_out

    def training_step(self, batch, batch_idx):
        ids, att_mask, labels = batch
        logits = self(ids, att_mask)
        labels = labels.type(torch.LongTensor).to(self.device)
        loss = F.cross_entropy(logits, labels)

        preds = F.softmax(logits, dim=1).argmax(dim=1)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.accuracy(preds, labels), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ids, att_mask, labels = batch
        logits = self(ids, att_mask)
        labels = labels.type(torch.LongTensor).to(self.device)
        loss = F.cross_entropy(logits, labels)

        preds = F.softmax(logits, dim=1).argmax(dim=1)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy(preds, labels), prog_bar=True)
        self.log("val_f1", self.f1_score(preds, labels), prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        ids, att_mask, labels = batch
        logits = self(ids, att_mask)
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

# Training the Model
if __name__ == "__main__":
    data_module = datasetDatasetNLPDataModule(datasetDatasetNLP, batch_size=32)
    model = LitCodeBERT()

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=5, # 10 # the good is 5
        callbacks=[checkpoint_callback]
    )
    
    embedmodel = f"{utls.cache_dir()}/embedmodel/CodeBERT"
    if not os.path.exists(embedmodel):
        os.makedirs(embedmodel)
        #finetune CodeBert
        trainer.fit(model, data_module)
        trainer.test(model, data_module)
        # Save Metrics and Codebert
        metrics = trainer.callback_metrics
        metrics_dict = {key: [value.item()] for key, value in metrics.items() if isinstance(value, torch.Tensor)}
        # metrics_df = pd.DataFrame(metrics_dict)
        # c_path = f"{utls.outputs_dir()}/codebert_metrics.csv"
        # metrics_df.to_csv(c_path, index=False)
        # print(f"Codebert metrics saved to {c_path}")
        codebert_path = embedmodel
        model.bert.save_pretrained(codebert_path)
        model.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        model.tokenizer.save_pretrained(codebert_path)

    else:
        print(f"A finetune CodeBERT model exit. \nLoading from a pre-trained.")


# Load the fine-tuned model for embedding
class CodeBertEmbedder:
    def __init__(self, model_path):
        
        if os.path.exists(model_path):
            print("Loading CodeBERT for embedding from cache ...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
        else:   
            cache_dir = utls.get_dir(f"{cache_dir()}/codebert_model")
            print("[Info] Loading Codebert...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/codebert-base", cache_dir=cache_dir
            )
            self.model = AutoModel.from_pretrained(
                "microsoft/codebert-base", cache_dir=cache_dir
            )
            self._dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model.to(self._dev)
    
    def embed(self, text):
        tokens = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            return self.model(tokens["input_ids"], tokens["attention_mask"]).pooler_output


# text = """# Load Word2Vec Model for Embedding
# loaded_model = Word2Vec.load("word2vec_model.bin")
# text = "the pare de papa morarac"
# words = text.split()
# embedding = np.mean([loaded_model.wv[word] for word in words if word in loaded_model.wv], axis=0)
# if embedding is None:
#     embedding = np.zeros(100)  # Default to zero vector if no words match
# print("Generated embedding:", embedding)"""
# # test functionallity
# save_path = embedmodel
# embedder = CodeBertEmbedder(save_path)
# # -0.1931, -0.6361,  0.2734,  0.7397, -0.8585, -0.8132, -0.1474,  0.8504,
# # -0.1931, -0.6361,  0.2734,  0.7397, -0.8585, -0.8132, -0.1474,  0.8504,
# embedding = embedder.embed(text)
# print(embedding)