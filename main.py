import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_rate = 0.001
import_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 250

train_ds = torchvision.datasets.MNIST(root="./data",
                                      train=True,
                                      transform=transforms.ToTensor(),
                                      download= True)

test_ds = torchvision.datasets.MNIST(root="./data",
                                      train=True,
                                      transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_ds,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_ds,
                                           batch_size=batch_size,
                                           shuffle=False)

examples = iter(test_loader)
example_data,_ = next(examples)

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(example_data[i][0],cmap='gray')

class neuralNet(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(neuralNet,self).__init__()
        self.l1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size,num_classes)

    def forward(self,x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = neuralNet(import_size,hidden_size,num_classes).to(device)



loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr= learning_rate)

total_steps = len(train_loader)
for epochs in range(num_epochs):
    for i,(image,label) in enumerate(train_loader):
        image = image.reshape(-1,28*28).to(device)
        label = label.to(device)

        outputs = model(image)
        l = loss(outputs,label)

        l.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i+1)%40 == 0:
            print(f"epoch {epochs+1}/{num_epochs} , step {i+1}/{total_steps}, loss {l.item():.4f}")


with torch.no_grad():
    correct = 0
    samples = len(test_loader.dataset)

    for image,label in test_loader:
        image = image.reshape(-1,28*28).to(device)
        label = label.to(device)

        outputs = model(image)

        _, predicted = torch.max(outputs,1)
        correct += (predicted == label).sum().item()

    acc = correct/samples*100
    print(f"Accuracy:%{acc}")

with torch.no_grad():
    examples = iter(test_loader)
    example_data,_ = next(examples)
    for i in range(120,125):
        image = test_loader.dataset[i][0].reshape(-1,28*28).to(device)
        output = model(image)
        _, predicted = torch.max(output,1)
        print("prediction:"+str(predicted[0].numpy()))
        print("label:"+ str(test_loader.dataset[i][1]))
        plt.imshow(example_data[i][0],cmap='gray')
        plt.show()