
### Imports
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as NN
import torch.nn.functional as F
import torch.optim as optim
import wandb
from multidigit_dataset import MultiDigitDataset  # custom dataset


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

## Import dataset
train_data = MultiDigitDataset(data_dir="data/multidigit", split="train")
val_data = MultiDigitDataset(data_dir="data/multidigit", split="val")
train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_data, batch_size=config["batch_size"])


config["num_classes"] = 13  ### unhardcode later

START_TOKEN_ID = 10
EOS_TOKEN_ID = 11
PAD_TOKEN_ID = 12
VOCAB_SIZE = 13



# --- Positional patch encoder ---
linear_proj = nn.Linear(PATCH_SIZE * PATCH_SIZE, EMBEDDING_DIM)
row_embed = nn.Embedding(4, EMBEDDING_DIM // 2)
col_embed = nn.Embedding(4, EMBEDDING_DIM // 2)

def patch_image(image):  # [1, 28, 28]
    patches = image.unfold(1, PATCH_SIZE, PATCH_SIZE).unfold(2, PATCH_SIZE, PATCH_SIZE)
    patches = patches.contiguous().view(-1, PATCH_SIZE * PATCH_SIZE)
    patch_embeddings = linear_proj(patches)
    positions = torch.arange(16, device=patch_embeddings.device)
    rows = positions // 4
    cols = positions % 4
    pos_embed = torch.cat([row_embed(rows), col_embed(cols)], dim=1)
    patch_embeddings = patch_embeddings + pos_embed
    return patch_embeddings

transform = transforms.Compose([
    transforms.ToTensor()
])

def patch_image_tensor(img_tensor):
    # Input: (B, 1, 28, 28)
    patches = img_tensor.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)  # (B, 1, 4, 4, 7, 7)
    patches = patches.contiguous().view(img_tensor.size(0), 16, -1)  # (B, 16, 49)
    
    patch_embeddings = linear_proj(patches)  # (B, 16, 48)

    # Add positional encoding
    positions = torch.arange(16, device=img_tensor.device)
    rows = positions // 4
    cols = positions % 4
    pos_embed = torch.cat([row_embed(rows), col_embed(cols)], dim=1)  # (16, 48)
    patch_embeddings = patch_embeddings + pos_embed.unsqueeze(0)  # (1, 16, 48)

    return patch_embeddings




### Patchify - Split image into X smaller images (technically already a tensor)
def patchify(image, patches):
    if image.dim() == 3:
        image = image.unsqueeze(1)  # Add channel dimension
    B, C, H, W = image.shape   ### I had some other notes here and lost em, my handcoded solution was both terrible and kinda assumed image over tensor
    patch_h = H // patches     ### B is batch C is channel, H is weight, W is width
    patch_w = W // patches     ### Divide height and width by the divisor var. not sure what it does with odds I think the // rounds or something simmilar
    p = image.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w) ### unfold is doing tensor dimension stuff. Simple explaination is making squares one 
    p = p.contiguous().view(B, C, patches * patches, patch_h * patch_w)
    return p.squeeze(1)


### Patchify - Split image into X smaller images (technically already a tensor)
def patchify(image, patches):
    if image.dim() == 3:
        image = image.unsqueeze(1)  # Add channel dimension
    B, C, H, W = image.shape
    patch_h = H // patches
    patch_w = W // patches
    patches = image.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)  # (B, C, patches, patches, patch_h, patch_w)
    patches = patches.contiguous().view(B, C, patches * patches, patch_h * patch_w)
    return patches.squeeze(1)  # (B, N, patch_dim)


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

    def forward(self, x_q, x_kv=None, mask=None):  ## Forward pass - Note backward pass done for all connected objects in the training loop - Murky on how this works at best
        if x_kv is None:  ### does attention or self attention as appropriate
            x_kv = x_q 
        Q = self.q_proj(x_q) # Query
        K = self.k_proj(x_kv) # Key
        V = self.v_proj(x_kv) # Value

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
        #self.classifier = NN.Linear(cfg["query_dim"], cfg["num_classes"]) ### output layer needs nodes equal to each logit to be predicted

    def forward(self, x): ### this is super in the machine's style, I dont know other ways to do it, did lean on the bots here.
        x = patchify(x, self.patch_split)               # (B, N, patch_dim) ## Do patchification
        x = x.view(x.size(0), x.size(1), -1)        # Flatten patches to (B, N, patch_dim) if needed
        x = self.patch_embed(x)                         # (B, N, embed_dim) ## Do embedding
        x = x + self.pos_embed                          # Add position      ## Do positional encoding coz otherwise attention doesnt work
        x = self.attn(self.norm1(x))                    # Attention         ## Do attention head
        x = self.mlp(self.norm2(x))                     # Feedforward       ## Do generic hidden layers, let the model do more math. 2 layers let classifiers write curves
        #x = x.mean(dim=1)                               # Pooling           ## I hate this step and can only conceptualize it as silly really, but I guess we need to smush down the size of everything. I figure maybe we can do this in the MLP layer tbh. CLS Token is apparently the normal thing to do.
        return x #self.classifier(x)



class MNISTDecoder(NN.Module):
    def __init__(self, cfg):  ### cfg hopefully to pull from wandb sweep later
        super().__init__()
        #self.patch_split = cfg["patches"]   ### Trying to keep things dynamic because we're due to change this later
        #self.patch_embed = NN.Linear(cfg["patch_dim"], cfg["embed_dim"])  ### Patches to embedding of several
        self.token_embed = NN.Embedding(VOCAB_SIZE, cfg["embed_dim"])
        self.time_embed = NN.Parameter(torch.randn(1, cfg["num_patches"], cfg["embed_dim"]))  
        self.attn = AttentionHead(cfg["embed_dim"], cfg["query_dim"]) ### make an instance of the attentionhead class in the MNISTModel class
        self.cross_attn = AttentionHead(cfg["embed_dim"], cfg["query_dim"])
        self.mlp = NN.Sequential(   ### I didnt want the model to be the least deep possible, because that sounds bad, and adding a few layers should just be necessary for any sort of complex classification, which we are doing! Consider this block a hidden layer block
            NN.Linear(cfg["query_dim"], cfg["query_dim"] * 2),
            NN.ReLU(),
            NN.Linear(cfg["query_dim"] * 2, cfg["query_dim"])
        )
        self.norm1 = NN.LayerNorm(cfg["embed_dim"]) ### unspike any weird spikes we might have
        self.norm2 = NN.LayerNorm(cfg["query_dim"]) ### and again for query
        self.norm3 = NN.LayerNorm(cfg["query_dim"]) ##########################################################################################   FIX ME
        self.classifier = NN.Linear(cfg["query_dim"], VOCAB_SIZE) ### output layer needs nodes equal to each logit to be predicted


    def forward(self, tgt_seq, encoder_output): ### this is super in the machine's style, I dont know other ways to do it, did lean on the bots here.
        B, T = tgt_seq.shape
        token = self.token_embed(tgt_seq)
        causal_mask = torch.tril(torch.ones(T, T, device=tgt_seq.device)).unsqueeze(0).repeat(B, 1, 1)
        x = token
        x = x + self.time_embed                          # Add time
        x = self.attn(self.norm1(x))                    # Attention         ## Do attention head
        x = self.mlp(self.norm2(x))                     # Feedforward       ## Do generic hidden layers, let the model do more math. 2 layers let classifiers write curves
        x = self.attn(self.norm1(x), mask=causal_mask)
        x = self.cross_attn(self.norm2(x), x_kv=encoder_output)
        x = self.mlp(self.norm3(x))
        return self.classifier(x)


class EncoderDecoderModel(NN.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, tgt_seq):
        patches = patchify(images, config["patches"])  # Same patchify as used in encoder
        encoder_output = self.encoder(patches)         # shape: [B, N, D]
        output = self.decoder(tgt_seq, encoder_output) # shape: [B, seq_len, vocab_size]
        return output

def create_decoder_input(labels, start_token, pad_token, max_len=3):
    # labels: [batch_size] — a single digit per sample for now
    batch_size = labels.size(0)

    # Convert to [batch_size, 1]
    labels = labels.unsqueeze(1)

    # Prepend START token, pad to max_len
    start = torch.full((batch_size, 1), start_token, dtype=torch.long, device=labels.device)
    combined = torch.cat([start, labels], dim=1)  # [B, 2]

    # Pad to max_len if needed
    if combined.size(1) < max_len:
        pad = torch.full((batch_size, max_len - combined.size(1)), pad_token, dtype=torch.long, device=labels.device)
        combined = torch.cat([combined, pad], dim=1)

    return combined  # shape: [B, max_len]


### Save + W&B upload
def save_model(model, path): #Locally
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def upload_model_to_wandb(model, run, path): # Remote
    artifact = wandb.Artifact("ABMNIST3", type="model")  ## 2 coz foundation project
    artifact.add_file(path)
    run.log_artifact(artifact)
    print("Model uploaded to W&B")

### Training loop
def training():
    run = wandb.init(project="MLXTransformer", config=config)
    cfg = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = MNISTModel(cfg).to(device)
    decoder = MNISTDecoder(cfg).to(device)
    model = EncoderDecoderModel(encoder, decoder).to(device)    

    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"])  ### Adam is good and I am Adam I shall use Adam. Adpative something momentum.
    loss_fn = NN.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)  # So padding doesn't count 

    model.train()
    for epoch in range(cfg["epochs"]):
        for images, tgt_seq, target_seq in train_loader: # Prepare image input
            images, tgt_seq, target_seq = images.to(device), tgt_seq.to(device), target_seq.to(device)
            logits = model(images, tgt_seq)
            loss = loss_fn(logits.view(-1, VOCAB_SIZE), target_seq.view(-1))
            optimizer.zero_grad()    ### resets gradients between batches/epochs
            loss.backward()   ### Backprop - Do learning
            optimizer.step()

        # Eval on test set each epoch
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
                for images, tgt_seq, target_seq in val_loader:
                    images, tgt_seq, target_seq = images.to(device), tgt_seq.to(device), target_seq.to(device)
                    logits = model(images, tgt_seq)
                    predictions = torch.argmax(logits, dim=-1)

                    # Optional: only count where not PAD
                    correct += (predictions == target_seq).sum().item()
                    total += target_seq.numel()

        accuracy = correct / total * 100
        print(f"Epoch {epoch+1}/{cfg['epochs']}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")
        wandb.log({
            "loss": loss.item(),
            "epoch": epoch,
            "accuracy": accuracy
        })

    ### Save and upload model

    save_model(model, "ABMNIST3.pth")
    upload_model_to_wandb(model, run, "ABMNIST3.pth")
    wandb.finish()### Launch training


if __name__ == "__main__":
    training()

