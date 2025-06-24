
### Imports
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as NN
import torch.nn.functional as F
import torch.optim as optim
import wandb

### Config (sweep-friendly, but not sweeping yet
config = {
    "patches": 4, ### as in 4 by 4, makes it not die given odd numbers... This is a bit unclear should probably rename
    "embed_dim": 32,
    "epochs": 50,
    "learning_rate": 0.001,
    "batch_size": 128,
}
config["patch_dim"] = (28 // config["patches"]) ** 2
config["query_dim"] = config["embed_dim"]
config["num_patches"] = config["patches"] ** 2

### Pull and prepare data- This sets the 0-255 values to -1->1. Could also do normalized this was a reccomendation holdover from foundation project
bw_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=bw_transform, download=True)  ### Note, does the transform, but also doesnt do patchify, I put that later, though I think ultimately it doesnt matter as long as they're both done
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=bw_transform, download=True)

config["num_classes"] = len(train_dataset.classes)  ### Copied from Helen - Just sets 10 but scales to the dataset if they start using alphanumeric or something

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True) ### I only understand dataloaders as a necessary step on the way into the model particularly for arranging batches
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

### Patchify - Split image into X smaller images (technically already a tensor)
def patchify(image, patches):
    B, C, H, W = image.shape   ### I had some other notes here and lost em, my handcoded solution was both terrible and kinda assumed image over tensor
    patch_h = H // patches     ### B is batch C is channel, H is weight, W is width
    patch_w = W // patches     ### Divide height and width by the divisor var. not sure what it does with odds I think the // rounds or something simmilar
    p = image.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w) ### unfold is doing tensor dimension stuff. Simple explaination is making squares one 
    p = p.contiguous().view(B, C, patches * patches, patch_h * patch_w)
    return p.squeeze(1)

### Attention Head
### Boy lots to document here, see obsidian notes file for MLX day 11,12, and Attention is all you need.
### Current mental models:
### 1. What if we had a vector DB of meaningful updates we could apply, what if we arranged a NN to be forced into being that
### 2. Take a batch of patches (best thought about as N patches where N recreates original image).
###     - Make 3 copies
###     - Copy 1 is trained to say How much do i want to be changed WHEN IN POSITION X per other stuff
###     - Copy 2 is trained to say How much do i want to change THING IN POSITION X per other stuff
###     - Copy 3 is trained to say How much change do I want to impart
###     - Arrange copy 1 and 2 into a table, where each cell does dotproduct (matmul function here) to see if they agree
###     - Ergo, if X and Y want to mingle, mingle them Z much.

class AttentionHead(NN.Module):  ## We define this as another neural net basically, kinda feeds into mental model 1. above
    def __init__(self, dim, query_dim):
        super().__init__() ### Go up a class to NN.Module, do it's initialization function
        self.q_proj = NN.Linear(dim, query_dim) ### we make 1 layer, and dont connect it. # Query
        self.k_proj = NN.Linear(dim, query_dim)  # Key
        self.v_proj = NN.Linear(dim, query_dim)  # Value

    def forward(self, x):  ## Forward pass - Note backward pass done for all connected objects in the training loop - Murky on how this works at best
        Q = self.q_proj(x) # Query
        K = self.k_proj(x) # Key
        V = self.v_proj(x) # Value

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)  ## do dot product. Do you want to mingle, do they want to be mingled.
        attn_weights = F.softmax(attn_scores, dim=-1) # normalize
        return torch.matmul(attn_weights, V) # Apply values given QV agreement

### Main Model
class MNISTModel(NN.Module):
    def __init__(self, cfg):  ### cfg hopefully to pull from wandb sweep later
        super().__init__()
        self.patch_split = cfg["patches"]   ### Trying to keep things dynamic because we're due to change this later
        self.patch_embed = NN.Linear(cfg["patch_dim"], cfg["embed_dim"])  ### Patches to embedding of several
        self.pos_embed = NN.Parameter(torch.randn(1, cfg["num_patches"], cfg["embed_dim"]))  #### that random1 is only inituating a positional value to be trained later
        self.attn = AttentionHead(cfg["embed_dim"], cfg["query_dim"]) ### make an instance of the attentionhead class in the MNISTModel class
        self.mlp = NN.Sequential(   ### I didnt want the model to be the least deep possible, because that sounds bad, and adding a few layers should just be necessary for any sort of complex classification, which we are doing! Consider this block a hidden layer block
            NN.Linear(cfg["query_dim"], cfg["query_dim"] * 2),
            NN.ReLU(),
            NN.Linear(cfg["query_dim"] * 2, cfg["query_dim"])
        )
        self.norm1 = NN.LayerNorm(cfg["embed_dim"]) ### unspike any weird spikes we might have
        self.norm2 = NN.LayerNorm(cfg["query_dim"]) ### and again for query
        self.classifier = NN.Linear(cfg["query_dim"], cfg["num_classes"]) ### output layer needs nodes equal to each logit to be predicted

    def forward(self, x): ### this is super in the machine's style, I dont know other ways to do it, did lean on the bots here.
        x = patchify(x, self.patch_split)               # (B, N, patch_dim) ## Do patchification
        x = self.patch_embed(x)                         # (B, N, embed_dim) ## Do embedding
        x = x + self.pos_embed                          # Add position      ## Do positional encoding coz otherwise attention doesnt work
        x = self.attn(self.norm1(x))                    # Attention         ## Do attention head
        x = self.mlp(self.norm2(x))                     # Feedforward       ## Do generic hidden layers, let the model do more math. 2 layers let classifiers write curves
        x = x.mean(dim=1)                               # Pooling           ## I hate this step and can only conceptualize it as silly really, but I guess we need to smush down the size of everything. I figure maybe we can do this in the MLP layer tbh. CLS Token is apparently the normal thing to do.
        return self.classifier(x)

### Save + W&B upload
def save_model(model, path): #Locally
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def upload_model_to_wandb(model, run, path): # Remote
    artifact = wandb.Artifact("ABMNIST2", type="model")  ## 2 coz foundation project
    artifact.add_file(path)
    run.log_artifact(artifact)
    print("Model uploaded to W&B")

### Training loop
def training():
    run = wandb.init(project="MLXTransformer", config=config)
    cfg = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  ### Didnt include MPS dont care runs about the same tbh
    model = MNISTModel(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"])  ### Adam is good and I am Adam I shall use Adam. Adpative something momentum.
    loss_fn = NN.CrossEntropyLoss()    ### Just using the normal loss function here, dont believe there's any need for anything fancy

    for epoch in range(cfg["epochs"]):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()    ### resets gradients between batches/epochs
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()   ### Backprop - Do learning
            optimizer.step()

        # Eval on test set each epoch
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)  ## Much like how data passes in above
                outputs = model(images)
                predictions = torch.argmax(outputs, dim=1)  ## top predicted logit
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

