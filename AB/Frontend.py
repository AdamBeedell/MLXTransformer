## streamlitUi One.py


##pip install streamlit

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as NN
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from streamlit_drawable_canvas import st_canvas


def clear_actual_input(): 
  st.session_state["actual_input"] = ''  # add "text" as a key using the square brackets notation and set it to have the value '' 


#### load trained model

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
config["num_classes"] = 10 


### Pull and prepare data- This sets the 0-255 values to -1->1. Could also do normalized this was a reccomendation holdover from foundation project
bw_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


### Patchify - Split image into X smaller images (technically already a tensor)
def patchify(image, patches):
    B, C, H, W = image.shape   ### I had some other notes here and lost em, my handcoded solution was both terrible and kinda assumed image over tensor
    patch_h = H // patches     ### B is batch C is channel, H is weight, W is width
    patch_w = W // patches     ### Divide height and width by the divisor var. not sure what it does with odds I think the // rounds or something simmilar
    p = image.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w) ### unfold is doing tensor dimension stuff. Simple explaination is making squares one 
    p = p.contiguous().view(B, C, patches * patches, patch_h * patch_w)
    return p.squeeze(1)



### if GPU available, use that, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### recreate NN structure

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


model = MNISTModel(config) ## init
model.load_state_dict(torch.load("ABMNIST2.pth", map_location=device, weights_only=True))
model.to(device)
model.eval()  ## inference mode

transform = transforms.Compose([   ### same as during training
    transforms.Grayscale(num_output_channels=1), ### BW
    transforms.Resize((28, 28)), ### 28px square rez
    transforms.ToTensor(),  
    transforms.Normalize((0.5,), (0.5,)) ### -1 to 1
])

st.title("MNIST Digit Classifier")


def validate_canvas(image_array, avg_density=0.05, threshold=0.9):
    """
    Checks if the canvas drawing is too dense or touches the edge.
    """
    img = Image.fromarray(image_array.astype("uint8")).convert("L")
    tensor = transform(img).unsqueeze(0)  # (1, 28, 28)

    density = (tensor > threshold).float().mean().item()

    edges = torch.cat([
        tensor[..., 0, :],    # top
        tensor[..., -1, :],   # bottom
        tensor[..., :, 0],    # left
        tensor[..., :, -1]    # right
    ], dim=-1)
    edge_touched = (edges > threshold).any().item()

    if density > avg_density * 1.5 or edge_touched:
        st.markdown("## 😡 Whoa there!")
        st.error("Too much ink or touching the edge. Please try again.")
        st.snow()
        return False
    return True



col1, col2 = st.columns([3, 1])
with col1:
    st.write("Draw a digit below and get a prediction!")
    # Create a drawing canvas
    canvas_result = st_canvas(
        fill_color="black",  # Background color
        stroke_color="white",  # Draw in white (digits are typically white on black)
        stroke_width=10,  # Thickness of the drawn lines
        background_color="black",  # Black background
        height=280,  # Canvas size
        width=280,
        drawing_mode="freedraw",
        key="canvas"
    )

with col2:
    st.write("**Correct Label:**")
    actual = st.text_input("Enter the true digit (0-9)", key="actual_input").strip()   ### this and beloy trying to keep it to valid digits in the DB
    actual = int(actual) if actual.isdigit() and 0 <= int(actual) <= 9 else None

    submit_actual = st.button("Submit Label", on_click=clear_actual_input)

# Process the drawn image
if canvas_result.image_data is not None:
    
    if validate_canvas(canvas_result.image_data, avg_density=0.048):  # replace with your real avg
    # proceed with prediction

        image = Image.fromarray(canvas_result.image_data.astype("uint8"))  # Convert to PIL image
        image = image.convert("L")  # Convert to grayscale
        
        #st.image(image, caption="Your Drawing", width=150)   ### unhash this to show the drawing back

        # Apply the same transformation as training
        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension
        image = image.to(device)

        #Do inference
        with torch.no_grad():
            output = model(image)
            probabilities = F.softmax(output, dim=1)  # Convert logits to probabilities
            predicted_digit = torch.argmax(probabilities).item()
            confidence = torch.max(probabilities).item()

        # Display results
        st.write(f"**Prediction:** {predicted_digit}")
        st.write(f"**Confidence:** {confidence:.2%}")

        #log it
        #logthelogs(predicted_digit, confidence, actual) ### disabled for now


# Upload method
#canvas = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

#if canvas:
#    image = Image.open(canvas).convert("L")  # Convert to grayscale
#    image = ImageOps.invert(image)  # Invert colors (white digit on black background)
    
#    st.image(image, caption="Uploaded Digit", width=150)

    # Apply the transformation
#    image = transform(image)
#    image = image.unsqueeze(0)  # Add batch dimension (1, 1, 28, 28)
#    image = image.to(device)  #GPU to CPU Switch

    # Run inference
 #   with torch.no_grad():
 #       output = model(image)
 #       probabilities = F.softmax(output, dim=1)  # Convert logits to probabilities
 #       predicted_digit = torch.argmax(probabilities).item()
 #       confidence = torch.max(probabilities).item()

    # Display results
 #   st.write(f"**Prediction:** {predicted_digit}")
 #   st.write(f"**Confidence:** {confidence:.2%}")

if submit_actual:
    st.balloons()
    #logthelogs(predicted_digit, confidence, actual)




