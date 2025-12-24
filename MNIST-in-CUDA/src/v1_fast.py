"""
v1_fast.py - Optimized PyTorch Implementation
Removes timing overhead for fair comparison with CUDA implementations.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
TRAIN_SIZE = 10000
EPOCHS = 10
LEARNING_RATE = 0.01
BATCH_SIZE = 32

# Optimizations for speed
torch.backends.cudnn.benchmark = True  # Auto-tune convolution algorithms
torch.set_float32_matmul_precision("high")  # TF32 on Ampere+ (no effect on T4)

# Load and preprocess data
X_train_np = np.fromfile("data/X_train.bin", dtype=np.float32).reshape(60000, 784)
y_train_np = np.fromfile("data/y_train.bin", dtype=np.int32)

# MNIST normalization
mean, std = 0.1307, 0.3081
X_train_np = (X_train_np - mean) / std

# Pre-load ALL data to GPU (no per-batch transfers)
train_data = torch.from_numpy(X_train_np[:TRAIN_SIZE]).to("cuda")
train_labels = torch.from_numpy(y_train_np[:TRAIN_SIZE]).long().to("cuda")

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(1024, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Initialize model
model = MLP().cuda()

# He initialization to match other implementations
with torch.no_grad():
    for layer in [model.fc1, model.fc2]:
        fan_in = layer.weight.size(1)
        scale = (2.0 / fan_in) ** 0.5
        layer.weight.uniform_(-scale, scale)
        layer.bias.zero_()

# Try torch.compile (PyTorch 2.0+) - comment out if not available
try:
    model = torch.compile(model, mode="reduce-overhead")
    print("Using torch.compile()")
except:
    print("torch.compile() not available, using eager mode")

# Simple SGD (no momentum, no weight decay)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Warmup (important for fair timing!)
print("Warming up...")
for _ in range(10):
    dummy = model(train_data[:BATCH_SIZE])
    loss = criterion(dummy, train_labels[:BATCH_SIZE])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
torch.cuda.synchronize()

# Training
num_batches = TRAIN_SIZE // BATCH_SIZE

print("\nTraining...")
torch.cuda.synchronize()
start_time = time.time()

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    
    for i in range(num_batches):
        # Get batch (already on GPU)
        start_idx = i * BATCH_SIZE
        x = train_data[start_idx:start_idx + BATCH_SIZE]
        y = train_labels[start_idx:start_idx + BATCH_SIZE]
        
        # Forward
        outputs = model(x)
        loss = criterion(outputs, y)
        
        # Backward
        loss.backward()
        
        # Update
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
        
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch} loss: {epoch_loss / num_batches:.4f}")

torch.cuda.synchronize()
end_time = time.time()

print(f"\n=== PYTORCH OPTIMIZED TIMING ===")
print(f"Total training time: {end_time - start_time:.2f} seconds")
print(f"Time per epoch: {(end_time - start_time) / EPOCHS * 1000:.1f} ms")

