import torch
print(torch.__version__)
print(torch.cuda.is_available())  ## looking for True
import torchvision
import torchvision.transforms as transforms

import torch.nn as NN
import torch.nn.functional as F
import torch.optim as optim






# Download dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=bwminus1to1, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=bwminus1to1, download=True)

# Create DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")


class MNISTModel(NN.Module):   ### This creates a class for our specific NN, inheriting from the pytorch equivalent
    def __init__(self):  
        super().__init__()  ## super goes up one level to the torch NN module, and initializes the net
        self.fc1 = NN.Linear(28 * 28, 256)  # First hidden layer (784 pixel slots, gradually reducing down)
        self.fc2 = NN.Linear(256, 128)  # half as many nodes
        self.fc3 = NN.Linear(128, 64)   # half as many nodes
        self.fc4 = NN.Linear(64, 10) # Output layer (64 -> 10, one for each valid prediction)

    def forward(self, x):  # feed forward
        x = x.view(-1, 28 * 28)  # Flatten input from (batch, 1, 28, 28) -> (batch, 784), applies to the tensor prepared above in the dataloader
        x = F.relu(self.fc1(x))  # Activation function (ReLU), no negatives, play with leaky ReLU later
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))  ### I think this should have been maybe softmax instead, as the default probability thig, though I dont actually understand why that is better over another normilization function, ultimately we're trying to hit the axium of probability that they sun to 1
        x = self.fc4(x)  # No activation here, end of the road ("cross-entropy expects raw logits" - which are produced here, the logits will be converted to probabilities later by the cross-entropy function during training and softmax during training and inference)
        return x
    
loss_function = NN.CrossEntropyLoss()  # using built-in loss function


model = MNISTModel() ##create the model as described abvoe

optimizer = optim.Adam(model.parameters(), lr=0.001) ### lr = learning rate, 0.001 is apparently a "normal" value. Adam is the optimizer chosen, also fairly default



##### do training

num_epochs = 30 ## passes through the dataset

for epoch in range(num_epochs):
    for images, lables in train_loader: #note uses batches defined earlier
        optimizer.zero_grad() ### reset gradients each time

        outputs = model(images) # forward pass
        loss = loss_function(outputs, lables)

        loss.backward() ## backprop method created by pytorch crossentropyloss function, very convenient
        optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")



### / training

## disbable training with 
# 
# model.eval()
# with torch.no_grad(): ...
# set back to training mode with model.train()



### evaluation:

correct = 0
total = 0

model.eval() 

with torch.no_grad():  # No need for gradients
    for images, labels in test_loader:
        outputs = model(images)  # Forward pass
        predictions = torch.argmax(outputs, dim=1)  # Get highest logit (most likely class)
        correct += (predictions == labels).sum().item()  # Count correct predictions
        total += labels.size(0)  # Track total samples

accuracy = correct / total * 100  # Convert to percentage
print(f"Test Accuracy: {accuracy:.2f}%")


### test accuracy  at 98.09% at 30 epochs with is probably overfit, but I wanted a proper period to leave it running for

torch.save(model.state_dict(), "mnist_model_one_weights.pth")
torch.save(model, "mnist_full_model.pth")


