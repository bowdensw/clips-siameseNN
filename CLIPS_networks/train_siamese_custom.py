# train_siamese.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random

# ----------------------------
# Define the Siamese Pair Classifier (aka build NN architecture)
# ----------------------------
class SiamesePairClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # shared branch layers (4 → 16 → 16)
        self.fc1 = nn.Linear(in_features=4,  out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=16)
        # merge & classify layers (32 → 16 → 2)
        self.fc3 = nn.Linear(in_features=32, out_features=16)
        self.fc4 = nn.Linear(in_features=16, out_features=2)

    def branch_forward(self, x):
        """
        Process a single 4-dim input through two ReLU layers to get a 16-dim embedding.
        """
        # x.shape = (batch_size, 4)
        h1 = F.relu(self.fc1(x))   # → (batch_size, 16)
        h2 = F.relu(self.fc2(h1))  # → (batch_size, 16)
        return h2                  # embedding ∈ R^16

    def forward(self, x1, x2):
        """
        x1, x2: each of shape (batch_size, 4)
        Returns: logits of shape (batch_size, 2)
                 where index 0 = “same category,” index 1 = “different category.”
        """
        e1 = self.branch_forward(x1)    # (batch_size, 16)
        e2 = self.branch_forward(x2)    # (batch_size, 16)
        # concatenate embeddings along dimension‐1
        cat = torch.cat([e1, e2], dim=1)  # (batch_size, 32)
        m   = F.relu(self.fc3(cat))       # (batch_size, 16)
        logits = self.fc4(m)              # (batch_size, 2)
        return logits

# ----------------------------
# Create a Pair Dataset
# ----------------------------
class PairDataset(Dataset):
    def __init__(self, num_examples=1000):
        
        self.pairs = [] # list to hold pairs of vectors, presented images next to each other
        self.labels = [] # list to hold labels (0 for same (dax/dax, fip/fip), 1 for different(dax/fip, fip/dax))

        # define anchors
        anchors = [[1, 1, 2, 2], [5, 5, 4, 4]]

        # define training sets
        training_set_1 = [[5, 5, 4, 4], [5, 4, 5, 4], [1, 2, 1, 2], [2, 2, 1, 1]]
        training_set_2 = [[1, 1, 2, 2], [1, 2, 1, 2], [5, 4, 5, 4], [4, 4, 5, 5]]

        # helper Function for normalization
        def normalize(vec):
            return [(x - 1) / 4 for x in vec]

        # add all (anchor, training_set_1[*]) pairs for anchor[0]
        for i, v2 in enumerate(training_set_1):
            v1_norm = normalize(anchors[0])
            v2_norm = normalize(v2)
            label = 0 if i < 2 else 1  #first two are same class, last two are different
            self.pairs.append((v1_norm, v2_norm))
            self.labels.append(label)

        #add all (anchor, training_set_2[*]) pairs for anchor[1]
        for i, v2 in enumerate(training_set_2):
            v1_norm = normalize(anchors[1])
            v2_norm = normalize(v2)
            label = 0 if i < 2 else 1
            self.pairs.append((v1_norm, v2_norm))
            self.labels.append(label)

    #returns the number of pairs in the dataset
    def __len__(self):
        return len(self.labels)

  
    #returns the pair (x1, x2) and label y for the given index
    def __getitem__(self, idx):
        x1, x2 = self.pairs[idx]
        y = self.labels[idx]
        x1_tensor = torch.tensor(x1, dtype=torch.float32)
        x2_tensor = torch.tensor(x2, dtype=torch.float32)
        y_tensor  = torch.tensor(y, dtype=torch.long)
        return x1_tensor, x2_tensor, y_tensor


# ----------------------------
# Training Function
# ----------------------------
def train_siamese():
    #hyperparameters
    batch_size = 2
    learning_rate = 1e-3
    num_epochs = 200

    #create the dataset & dataloader
    dataset = PairDataset(num_examples=2000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    #instantiate model, loss, optimizer
    model     = SiamesePairClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #training loop
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for x1_batch, x2_batch, y_batch in dataloader:
            #x1_batch.shape = (batch_size, 4)
            #x2_batch.shape = (batch_size, 4)
            #y_batch.shape  = (batch_size,)

            #forward pass → logits: (batch_size, 2)
            logits = model(x1_batch, x2_batch)
            loss   = criterion(logits, y_batch)

            #backprop + optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x1_batch.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs} – Loss: {epoch_loss:.4f}")

    #save the trained model weights
    torch.save(model.state_dict(), "siamese_pair_model.pth")
    print("Training complete. Model saved to siamese_pair_model.pth.")

# ----------------------------
#entry point
# ----------------------------
if __name__ == "__main__":
    train_siamese()
