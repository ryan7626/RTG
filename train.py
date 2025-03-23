import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from gesture_dataset import GestureDataset  # the class above
from model import HandGestureNet  # the model we defined earlier

# Load dataset
dataset = GestureDataset("gesture_data.csv")
num_classes = len(set(dataset.y.numpy()))
label_encoder = dataset.get_label_encoder()

# Split dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)

# Model
model = HandGestureNet(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for X, y in train_loader:
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss:.4f}")

    # Optional: Validation accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in val_loader:
            out = model(X)
            preds = torch.argmax(out, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = 100 * correct / total
    print(f"  ➤ Val Accuracy: {acc:.2f}%")

# Save model and label encoder
torch.save(model.state_dict(), "gesture_model.pt")
import joblib
joblib.dump(label_encoder, "gesture_labels.pkl")
print("✅ Model + label encoder saved!")
