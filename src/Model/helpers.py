import torch
import torch.nn.functional as F
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from src.Model.GraphEmbedder import GraphEmbedder


def evaluate(loader: DataLoader, model: GraphEmbedder | DataParallel):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            central_node_embs = batch['central_node_embedding']
            neighbor_embs = batch['node_embeddings']
            edge_embs = batch['edge_embeddings']

            outputs = model(central_node_embs, neighbor_embs, edge_embs, hide_central=True)
            loss = F.mse_loss(outputs, central_node_embs)
            total_loss += loss.item()
    model.train()
    return total_loss / len(loader)
