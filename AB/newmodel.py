
### Imports
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as NN
import torch.nn.functional as F
import torch.optim as optim
import wandb

### Config (sweep-friendly)
config = {
    "patches": 4,
    "embed_dim": 32,
    "epochs": 50,
    "learning_rate": 0.001,
    "batch_size": 128,
}
config["patch_dim"] = (28 // config["patches"]) ** 2
config["query_dim"] = config["embed_dim"]
config["num_patches"] = config["patches"] ** 2

### Data prep
bw_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=bw_transform, download=True)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=bw_transform, download=True)

config["num_classes"] = len(train_dataset.classes)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

### Patchify
def patchify(image, patches):
    B, C, H, W = image.shape
    patch_h = H // patches
    patch_w = W // patches
    p = image.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
    p = p.contiguous().view(B, C, patches * patches, patch_h * patch_w)
    return p.squeeze(1)

### Attention Head
class AttentionHead(NN.Module):
    def __init__(self, dim, query_dim):
        super().__init__()
        self.q_proj = NN.Linear(dim, query_dim)
        self.k_proj = NN.Linear(dim, query_dim)
        self.v_proj = NN.Linear(dim, query_dim)

    def forward(self, x):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_weights, V)

### Model
class MNISTModel(NN.Module):
    def __init__(self, cfg):
        super().__init__()
        self.patch_split = cfg["patches"]
        self.patch_embed = NN.Linear(cfg["patch_dim"], cfg["embed_dim"])
        self.pos_embed = NN.Parameter(torch.randn(1, cfg["num_patches"], cfg["embed_dim"]))
        self.attn = AttentionHead(cfg["embed_dim"], cfg["query_dim"])
        self.mlp = NN.Sequential(
            NN.Linear(cfg["query_dim"], cfg["query_dim"] * 2),
            NN.ReLU(),
            NN.Linear(cfg["query_dim"] * 2, cfg["query_dim"])
        )
        self.norm1 = NN.LayerNorm(cfg["embed_dim"])
        self.norm2 = NN.LayerNorm(cfg["query_dim"])
        self.classifier = NN.Linear(cfg["query_dim"], cfg["num_classes"])

    def forward(self, x):
        x = patchify(x, self.patch_split)               # (B, N, patch_dim)
        x = self.patch_embed(x)                         # (B, N, embed_dim)
        x = x + self.pos_embed                          # Add position
        x = self.attn(self.norm1(x))                    # Attention
        x = self.mlp(self.norm2(x))                     # Feedforward
        x = x.mean(dim=1)                               # Pooling
        return self.classifier(x)

### Save + W&B upload
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def upload_model_to_wandb(model, run, path):
    artifact = wandb.Artifact("ABMNIST2", type="model")
    artifact.add_file(path)
    run.log_artifact(artifact)
    print("Model uploaded to W&B")

### Training loop
def training():
    run = wandb.init(project="MLXTransformer", config=config)
    cfg = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTModel(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    loss_fn = NN.CrossEntropyLoss()

    for epoch in range(cfg["epochs"]):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        # Eval on test set each epoch (treated as val for now)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total * 100

        print(f"Epoch {epoch+1}/{cfg['epochs']}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")
        wandb.log({
            "loss": loss.item(),
            "epoch": epoch,
            "accuracy": accuracy
        })

    save_model(model, "ABMNIST2.pth")
    upload_model_to_wandb(model, run, "ABMNIST2.pth")
    wandb.finish()### Launch training

    
if __name__ == "__main__":
    training()

