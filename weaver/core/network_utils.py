import numpy as np
import networkx as nx

def build_similarity_graph(data, threshold=0.5):
    """
    构建样本相似图（基于 cosine 或 corr）
    """
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(data)
    graph = (sim > threshold).astype(int)
    G = nx.from_numpy_array(graph)
    return G
