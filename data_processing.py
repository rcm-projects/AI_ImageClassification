import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
from sklearn.decomposition import PCA
from torch.utils.data import Subset

# returns datasets from cifar 10
def get_cifar10_data(target_type='resnet'):

    if target_type == 'resnet':
        # resize to 224x224
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # keep 32x32
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


    trainset_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset_full = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_indices = []
    test_indices = []
    class_counts_train = {i: 0 for i in range(10)}
    class_counts_test = {i: 0 for i in range(10)}

    #filter training and test data:

    for i in range(len(trainset_full)):
        _, label = trainset_full[i]
        if class_counts_train[label] < 500:
            train_indices.append(i)
            class_counts_train[label] += 1

    for i in range(len(testset_full)):
        _, label = testset_full[i]
        if class_counts_test[label] < 100:
            test_indices.append(i)
            class_counts_test[label] += 1


    train_subset = Subset(trainset_full, train_indices)
    test_subset = Subset(testset_full, test_indices)

    return train_subset, test_subset

def extract_features(dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load resnet
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    modules = list(model.children())[:-1]
    model = torch.nn.Sequential(*modules)
    model.to(device)
    model.eval()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    
    features = []
    labels = []

    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(device)
            output = model(imgs).squeeze() 
            features.append(output.cpu().numpy())
            labels.append(lbls.numpy())

    return np.vstack(features), np.concatenate(labels)

def process_data():
    # get 224x224 images for resnet
    train_data, test_data = get_cifar10_data(target_type='resnet')
    

    X_train_512, y_train = extract_features(train_data)
    X_test_512, y_test = extract_features(test_data)

    # apply PCA
    pca = PCA(n_components=50)
    X_train_50 = pca.fit_transform(X_train_512)
    X_test_50 = pca.transform(X_test_512)

    return (X_train_50, y_train), (X_test_50, y_test)

# to verify: (should be 5000, 50)
if __name__ == "__main__":
    (X_tr, y_tr), (X_te, y_te) = process_data()
    print(f"Training Data Shape: {X_tr.shape}")