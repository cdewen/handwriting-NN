import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class_names = ['T-shirt/tsourop','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']

def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    custom_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_set=datasets.FashionMNIST('./data',train=True,
    download=True,transform=custom_transform)
    test_set=datasets.FashionMNIST('./data', train=False,
    transform=custom_transform)

    if training:
        return torch.utils.data.DataLoader(train_set,batch_size=64,
        shuffle=True)
    return torch.utils.data.DataLoader(test_set,batch_size=64, shuffle=False)



def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )




def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(T):
        correct = 0
        runningLoss = 0
        for _, (data, target) in enumerate(train_loader):
            opt.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            opt.step()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            runningLoss += loss.item() * data.size(0)
        print(f'Train Epoch: {epoch}\tAccuracy: {correct}/60000({100. * correct / 60000:.2f}%) Loss: {runningLoss / 60000:.3f}')


    


def evaluate_model(model, test_loader, criterion, show_loss = True):
    with torch.no_grad():
        runningLoss = 0.0
        batchCount = 0
        correct = 0  
        for _, (data, target) in enumerate(test_loader):
            output = model(data)
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            runningLoss += loss.item()
            batchCount += 1
        if show_loss:
            print(f'Average loss: {runningLoss / batchCount:.4f}')
        print(f'Accuracy: {100. * correct/10000:.2f}%')
    


def predict_label(model, test_images, index):
    with torch.no_grad():
            output = model(test_images[index])
            prob = F.softmax(output, dim=1)
            classIndices = torch.topk(prob, 3).indices
            for i in range(3):
                print(f'{class_names[classIndices[0][i].item()]}: {prob[0][classIndices[0][i].item()]*100:.2f}%')   

if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
