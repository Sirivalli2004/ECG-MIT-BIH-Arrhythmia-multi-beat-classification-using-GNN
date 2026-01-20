import os  
import sys 
import math 

import numpy as np 
import pandas as pd 

from sklearn.metrics import accuracy_score
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.utils.data as data 

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import GCNNorm 


def knn_imputation(df, n_neighbors=3):
    numerical_cols = df.select_dtypes(include=np.number).columns 
    df_numerical = df[numerical_cols]
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed_numerical = pd.DataFrame(imputer.fit_transform(df_numerical), columns=numerical_cols) 
    return df_imputed_numerical 


def create_knn_graph(X, k=5, mode='connectivity'):
    """
    Creates a k-nearest neighbors graph from tabular data.

    Args:
        X (np.ndarray): The feature data.
        k (int): The number of neighbors for each node.
        mode (str): Defines the type of graph ('connectivity' or 'distance').

    Returns:
        torch.Tensor: The edge index of the graph.
    """
    A = kneighbors_graph(X, k, mode=mode, include_self=False)
    # Convert the adjacency matrix to a COO format
    coo = A.tocoo()
    edge_index = torch.tensor([coo.row, coo.col], dtype=torch.long)
    return edge_index


class TabularGNN(nn.Module):
    """
    A Graph Neural Network model for tabular data classification.

    The model uses a simple Graph Convolutional Network (GCN) architecture.
    """
    def __init__(self, num_features, num_classes, hidden_channels=246):
        super(TabularGNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4= GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, num_classes)
    
    def forward(self, x, edge_index, edge_weight=None):
        """
        Defines the forward pass of the GNN.

        Args:
            x (torch.Tensor): The node features.
            edge_index (torch.Tensor): The graph's edge indices.
            edge_weight (torch.Tensor, optional): The edge weights.

        Returns:
            torch.Tensor: The log-softmax probabilities for each class.
        """
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv4(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)
    







def train_and_evaluate(model, data, epochs=1, lr=0.01):
    """
    Trains and evaluates the GNN model.

    Args:
        model (nn.Module): The GNN model.
        data (Data): The PyTorch Geometric data object.
        epochs (int): The number of training epochs.
        lr (float): The learning rate.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss() 
    best_test_acc = 0.0
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}') 
    
        model.eval()
        _, pred = model(data.x, data.edge_index).max(dim=1)
        test_acc = accuracy_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu())
        print(f'\nTest Accuracy: {test_acc:.4f}')
        print(f'Number of correct predictions: {torch.sum(pred[data.test_mask] == data.y[data.test_mask]).item()}/{len(data.test_mask)}')
        if test_acc > best_test_acc:
                best_test_acc = test_acc
        # ---- Final results ----
    correct = torch.sum(
        pred[data.test_mask] == data.y[data.test_mask]
    ).item()
    print("\nTraining finished!")
    print(f"Best Test Accuracy: {best_test_acc:.4f}")
    print(f"Number of correct predictions: {correct}/{data.test_mask.sum().item()}")


if __name__ == '__main__':

    os.system('cls' if os.name == 'nt' else 'clear') 


    dataset_name = 'balanced_data.csv'
    df = pd.read_csv(dataset_name)  
    print(df.head())

    num_samples = len(df) 
    num_features = 360 
    num_classes = 5 


    df_imputed = knn_imputation(df) 
    #df.fillna(df.mean(), inplace=True)
    #print(df_imputed.info()) 
    column_name = df.columns
    X = df_imputed[column_name[1:-1]]
    y = df_imputed[column_name[-1]]
    #print(X.head(), y.head())

    edge_index = create_knn_graph(X.to_numpy(), k=10)
    #print(edge_index.shape)

    # 3. Split data into training and testing sets
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X, y, range(num_samples), test_size=0.3, random_state=42, stratify=y
    )

    # Create masks for PyTorch Geometric
    train_mask = torch.zeros(num_samples, dtype=torch.bool)
    test_mask = torch.zeros(num_samples, dtype=torch.bool)
    train_mask[train_indices] = True
    test_mask[test_indices] = True 

    # # 4. Convert to PyTorch Geometric Data object
    x = torch.tensor(X.to_numpy(), dtype=torch.float)
    y = torch.tensor(y.to_numpy(), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)

    # # 5. Initialize and train the model
    model = TabularGNN(num_features, num_classes)

    # print("Starting GNN training...")
    train_and_evaluate(model, data, epochs=500, lr=0.01)
