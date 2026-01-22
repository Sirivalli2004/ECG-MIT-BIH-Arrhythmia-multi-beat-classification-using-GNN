import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.metrics import confusion_matrix, classification_report
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import k_hop_subgraph
import seaborn as sns

os.makedirs("outputs", exist_ok=True)

# -----------------------------
# Hyperparameters
# -----------------------------
EPOCHS = 50
LR = 0.01
KNN_K = 8
EXPLAIN_HOPS = 2
EXPLAIN_EPOCHS = 40

# -----------------------------
# Save hyperparameters
# -----------------------------
pd.DataFrame({
    "Parameter": ["Epochs", "Learning Rate", "KNN k", "Explain Hops", "Explain Epochs"],
    "Value": [EPOCHS, LR, KNN_K, EXPLAIN_HOPS, EXPLAIN_EPOCHS]
}).to_csv("outputs/hyperparameters.csv", index=False)

# -----------------------------
# Imputation (FIXED)
# -----------------------------
def knn_impute(df):
    cols = df.select_dtypes(include=np.number).columns
    imputer = KNNImputer(n_neighbors=3)   # <<< FIX
    df[cols] = imputer.fit_transform(df[cols])
    return df

# -----------------------------
# Graph
# -----------------------------
def build_graph(X, k):
    A = kneighbors_graph(X, k, mode="connectivity", include_self=False)
    coo = A.tocoo()
    edge_index = np.vstack((coo.row, coo.col))
    return torch.tensor(edge_index, dtype=torch.long)

# -----------------------------
# Model
# -----------------------------
class ECG_GNN(torch.nn.Module):
    def __init__(self, in_dim, classes):
        super().__init__()
        self.c1 = GCNConv(in_dim,128)
        self.c2 = GCNConv(128,128)
        self.c3 = GCNConv(128,classes)

    def forward(self,x,edge_index):
        x = self.c1(x,edge_index).relu()
        x = self.c2(x,edge_index).relu()
        x = self.c3(x,edge_index)
        return F.log_softmax(x,dim=1)

# -----------------------------
# Load
# -----------------------------
df = knn_impute(pd.read_csv("balanced_data.csv"))
X = df.iloc[:,1:-1].values
y = df.iloc[:,-1].values

edge_index = build_graph(X,KNN_K)

Xtr,Xte,Ytr,Yte,tr_idx,te_idx = train_test_split(
    X,y,range(len(X)),test_size=0.3,random_state=42,stratify=y)

train_mask = torch.zeros(len(X),dtype=torch.bool)
test_mask = torch.zeros(len(X),dtype=torch.bool)
train_mask[tr_idx]=True
test_mask[te_idx]=True

data = Data(
    x=torch.tensor(X,dtype=torch.float),
    y=torch.tensor(y,dtype=torch.long),
    edge_index=edge_index,
    train_mask=train_mask,
    test_mask=test_mask
)

# -----------------------------
# Train
# -----------------------------
model = ECG_GNN(X.shape[1],len(np.unique(y)))
opt = torch.optim.Adam(model.parameters(),lr=LR)

for i in range(EPOCHS):
    opt.zero_grad()
    out = model(data.x,data.edge_index)
    loss = F.nll_loss(out[data.train_mask],data.y[data.train_mask])
    loss.backward()
    opt.step()
    if i % 10 == 0:
        print(f"Epoch {i}  Loss {loss.item():.4f}")

# -----------------------------
# Predictions
# -----------------------------
model.eval()
pred = model(data.x,data.edge_index).argmax(dim=1)

# -----------------------------
# Class Distribution
# -----------------------------
plt.hist(pred[test_mask].cpu().numpy(),bins=5)
plt.title("Prediction Distribution")
plt.savefig("outputs/class_distribution.png",dpi=300)
plt.close()

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(data.y[test_mask].cpu(),pred[test_mask].cpu())
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png",dpi=300)
plt.close()

# -----------------------------
# Results CSV
# -----------------------------
report = classification_report(
    data.y[test_mask].cpu(),
    pred[test_mask].cpu(),
    output_dict=True
)
pd.DataFrame(report).transpose().to_csv("outputs/results.csv")

# -----------------------------
# Explainability (FAST & SAFE)
# -----------------------------
node = te_idx[0]
subset,edge_sub,map,_ = k_hop_subgraph(node,2,data.edge_index,relabel_nodes=True)

data_sub = Data(x=data.x[subset],edge_index=edge_sub,y=data.y[subset])

explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=EXPLAIN_EPOCHS),
    explanation_type="model",
    node_mask_type="attributes",
    edge_mask_type="object",
    model_config=dict(
        mode="multiclass_classification",
        task_level="node",
        return_type="log_probs"
    )
)

exp = explainer(data_sub.x,data_sub.edge_index,index=map)

exp.visualize_feature_importance(top_k=10)
plt.savefig("outputs/feature_importance.png",dpi=300)
plt.close()

exp.visualize_graph()
plt.savefig("outputs/subgraph.png",dpi=300)
plt.close()

# -----------------------------
# Methodology Diagram
# -----------------------------
plt.figure(figsize=(8,5))
plt.text(0.1,0.7,"ECG → Wavelet → Features → KNN Graph → GNN → Prediction → Explainability",fontsize=12)
plt.axis("off")
plt.savefig("outputs/methodology.png",dpi=300)
plt.close()

# -----------------------------
# Architecture Diagram
# -----------------------------
plt.figure(figsize=(8,5))
plt.text(0.1,0.7,"Input → GCN(128) → GCN(128) → Softmax(5)",fontsize=12)
plt.axis("off")
plt.savefig("outputs/architecture.png",dpi=300)
plt.close()

print("\nALL outputs saved in outputs/")
