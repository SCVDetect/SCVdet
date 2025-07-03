# SCVdet  

### Modeling Function-Level Relationships for Vulnerability Detection: A GNN-Based Approach  

This repository contains the implementation of our Graph Neural Network (GNN) model for vulnerability detection by modeling function-level relationships in source code.  

## 1. Dataset  & Data Processing 
We use two real-world datasets constructed from Java and C/C++ code:  
- **Dataset 1**: ProjectKB  
- **Dataset 2**: QEMU+FFmpeg  
For graph data extraction from source code, we use **Joern**, an open-source code analysis tool. The automated extraction process is implemented in the provided scripts. Run ``` ./installJoernanddata.sh``` to install.

```sh
chmod +x installJoernanddata.sh
./installJoernanddata.sh
```

## 3. Function-Level Relationship Modeling  
We approximate relationships using constructed graphs of individual source code functions.  

## Source Code Execution  

### 1. Data Processing & Graph Extraction  
The graph extraction time varies depending on the runtime environment and dataset size.  

```sh
cd sourcescripts  
python3 -B ./processing/process.py  
python3 -B ./processing/graphdata.py  
```
### 2. Node Feature Generation
We train sequence-based models (CodeBERT, Word2Vec, and SBERT) to generate node features, then construct graphs for model training.

```sh
python3 -B ./embeddmodel/codebert.py  
python3 -B ./embeddmodel/sentencebert.py  
python3 -B ./embeddmodel/word2vec.py  
python3 -B ./processing/graphconstruction.py 
```

### 3. Model Training & Testing
The model is trained and tested at both function and statement levels. Output (```./stoarge/outputs/```) includes:
 - Classification metrics
 - A CSV of model predictions for each function
 - Detailed predictions per code line (with line numbers)
 - Unique function IDs and corresponding source code (stored in ``` ./storage/processed/before```)

```sh
python3 -B ./model/scvuldetetc.py  
```



outputs_best_codebert_dbscan_kb


outputs
