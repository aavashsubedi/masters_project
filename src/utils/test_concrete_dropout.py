import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import Tensor
from torchvision import datasets, transforms
import wandb
from tqdm import tqdm
import concrete_dropout
from concrete_dropout import ConcreteDropout


@concrete_dropout.concrete_regulariser
class MLPConcreteDropout(nn.Module):

    def __init__(self, n_hidden: int = 64) -> None:

        super().__init__()

        self.fc1 = nn.Linear(784, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_hidden)
        self.fc4 = nn.Linear(n_hidden, 10)

        w, d = 1e-6, 1e-3
        self.cd1 = ConcreteDropout(weight_regulariser=w, dropout_regulariser=d)
        self.cd2 = ConcreteDropout(weight_regulariser=w, dropout_regulariser=d)
        self.cd3 = ConcreteDropout(weight_regulariser=w, dropout_regulariser=d)
        self.cd4 = ConcreteDropout(weight_regulariser=w, dropout_regulariser=d)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        

    def forward(self, x: Tensor) -> Tensor:

        x = x.view(-1, 784)

        """
        # Without Concrete dropout
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)

        """
        x = self.cd1(x, nn.Sequential(self.fc1, self.relu))
        x = self.cd2(x, nn.Sequential(self.fc2, self.relu))
        x = self.cd3(x, nn.Sequential(self.fc3, self.relu))
        x = self.cd4(x, self.fc4)
        
        return x


def train(model, trainloader, optimiser, epoch, device):

    train_loss = 0.0

    model.train()
    for batch_idx, (data, labels) in enumerate(trainloader):
        data, labels = data.to(device), labels.to(device)

        optimiser.zero_grad()

        outputs = model(data)
        loss = F.cross_entropy(outputs, labels) + model.regularisation()
        train_loss += loss.item() * data.size(0)

        loss.backward()
        optimiser.step()

    train_loss /= len(trainloader.dataset)
    return train_loss


def test(model, testloader, device):

    correct = 0
    total = 0
    test_loss = 0.0

    model.eval()
    for batch_idx, (data, labels) in enumerate(testloader):
        data, labels = data.to(device), labels.to(device)

        outputs = model(data)
        p_outputs = F.softmax(outputs, dim=1)

        loss = F.cross_entropy(outputs, labels)
        test_loss += loss.item() * data.size(0)
        
        preds = p_outputs.argmax(dim=1, keepdim=True)
        correct += preds.eq(labels.view_as(preds)).sum().item()

    test_loss /= len(testloader.dataset)
    accuracy = correct / len(testloader.dataset)

    return test_loss, accuracy


def run():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda:0' else {}

    # define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # load / process data
    trainset = datasets.MNIST('./data',
                              train=True,
                              download=True,
                              transform=transform)

    testset = datasets.MNIST('./data',
                             train=False,
                             download=True,
                             transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=64, **kwargs)

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=1000, **kwargs)

    # define model / optimizer
    model = MLPConcreteDropout().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    wandb.init()
    wandb.watch(model)

    # run training
    for epoch in tqdm(range(1, 500)):
        train_loss = train(model, trainloader, optimiser, epoch, device)
        test_loss, accuracy = test(model, testloader, device)

        # extract dropout probabilities
        probs = [cd.p.cpu().data.numpy()[0] for cd in filter(
            lambda x: isinstance(x, ConcreteDropout), model.modules()
        )]

        wandb.log({"conc_drop_train_loss": train_loss})
        wandb.log({"conc_drop_test_loss": test_loss})
        wandb.log({"conc_drop_accuracy": accuracy})
        wandb.log({"conc_drop_layer1": probs[0]})
        wandb.log({"conc_drop_layer2": probs[1]})
        wandb.log({"conc_drop_layer3": probs[2]})
        wandb.log({"conc_drop_layer4": probs[3]})

run()