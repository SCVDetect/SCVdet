import os
import torch as th
import torch.nn as nn
import torch
import dgl
from dgl.nn import SAGEConv
import networkx as nx
from dgl import load_graphs
from node2vec import Node2Vec
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import uutils.__utils__ as utls
from GATmain import  GraphFunctionDataset
import random

def compute_graph_embedding(graph, method='mean'):
    """--- Graph Embedding Computation --"""
    h = graph.ndata['_FUNC_EMB'].float()

    if method == 'mean':
        return h.mean(dim=0)

    elif method == 'max':
        return h.max(dim=0)[0]

    elif method == "Node2Vec":

        def node2vecmodel(graph):
            nxg = graph.to_networkx()
            if len(nxg.nodes()) > 2000:
                sampled_nodes = random.sample(list(nxg.nodes()), min(1000, len(nxg.nodes())))
                nxg = nxg.subgraph(sampled_nodes)
                nxg = nxg.to_undirected()
                largest_cc = max(nx.connected_components(nxg), key=len)
                nxg = nxg.subgraph(largest_cc)
            node2vec_model = Node2Vec(nxg, dimensions=128, walk_length=5, num_walks=5, workers=4 )
            node2vec_fitted = node2vec_model.fit(window=10, min_count=1, batch_words=2)
            embeddings = node2vec_fitted.wv
            node_embeddings = {int(node): embeddings[str(node)] for node in nxg.nodes()}
            valid_nodes = [node.item() for node in graph.nodes() if node.item() in node_embeddings]
            embedding_matrix = th.tensor(
                [node_embeddings[node] for node in valid_nodes],
                dtype=th.float
            )
            if len(embedding_matrix) > 0:
                return embedding_matrix.mean(dim=0)
            else:
                return th.zeros(128, dtype=th.float)  
        return node2vecmodel(graph)

    elif method == "GraphSAGE":
        class GraphSAGE(nn.Module):
            def __init__(self, in_feats, hidden_feats, out_feats):
                super(GraphSAGE, self).__init__()
                self.conv1 = SAGEConv(in_feats, hidden_feats, aggregator_type='mean')
                self.conv2 = SAGEConv(hidden_feats, out_feats, aggregator_type='mean')

            def forward(self, g, inputs):
                h = self.conv1(g, inputs)
                h = th.relu(h)
                h = self.conv2(g, h)
                return h

        def graphsage_model(graph):
            node_feats = graph.ndata['_FUNC_EMB'].float()
            in_feats = node_feats.shape[1]
            model = GraphSAGE(in_feats=in_feats, hidden_feats=64, out_feats=128)
            model.eval()
            with th.no_grad():
                node_embeddings = model(graph, node_feats)
            return node_embeddings.mean(dim=0)

        return graphsage_model(graph)
    else:
        raise ValueError(f"Unknown method: {method}")


def dependencygraphwithsimlabel(embeddings, labels, similarity_threshold=0.06, methodd = "kmean"):
    """# -- Dependency Graph Construction with similarity and labels--"""
    graph = nx.Graph()
    embeddings = np.array(embeddings)
    if methodd == "kmean":
        similarity_matrix = cosine_similarity(embeddings)
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if labels[i] == -1 or labels[j] == -1:
                    continue
                if labels[i] == labels[j] and similarity_matrix[i][j] >= similarity_threshold:
                    graph.add_edge(i, j)
        return graph
    else:
        for i, vector in enumerate(embeddings):
            graph.add_node(i, vector=vector, cluster=labels[i])
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                if labels[i] == labels[j] and labels[i] != -1:
                    if distance_matrix[i,j] <= eps:
                        graph.add_edge(i, j, weight=distance_matrix[i,j])
        
        return  graph 

def create_function_level_dependency_graph(embeddings, method, eps=0.6, min_samples=2, 
                                           n_clusters=2, similarity_threshold=0.06):
    if method.lower() == "dbscan":
        print("[Infos] Clustering method: DBSCAN")
        metric='cosine'
        vectors = np.array(embeddings)
        distance_matrix = pairwise_distances(vectors, metric=metric)
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        labels = clusterer.fit_predict(distance_matrix)
        dependency_graph = dependencygraphwithsimlabel(embeddings, labels, similarity_threshold)
        return  dependency_graph, labels 
    
    elif method.lower() == "kmeans":
        print("[Infos] Clustering method: KMeans")
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        labels = clusterer.fit_predict(embeddings)
        dependency_graph = dependencygraphwithsimlabel(embeddings, labels, similarity_threshold, methodd = "kmean")
        return dependency_graph, labels
    else:
        raise ValueError(f"Unsupported clustering method: {method}. Use 'dbscan' or 'kmeans'.")


def networkx_to_dgl(nx_graph, feature_size, device='cpu'):
    dgl_graph = dgl.from_networkx(nx_graph)
    dgl_graph = dgl_graph.to(device)
    num_nodes = dgl_graph.num_nodes()
    random_features = torch.randn(num_nodes, feature_size, device=device)
    dgl_graph.ndata['_FUNC_EMB'] = random_features
    return dgl_graph

def visualize_dependency_graph(dependency_graph, title="Function-Level Dependency Graph", filename="dependency_graph.png"):
    pos = nx.spring_layout(dependency_graph, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw(
        dependency_graph,
        pos,
        with_labels=True,
        node_color='skyblue',
        node_size=800,
        edge_color='gray'
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"[Saved] {filename}")


