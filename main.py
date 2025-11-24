import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset, Subset
import torchvision
import torchvision.transforms as transforms

from data_processing import process_data, get_cifar10_data
from models import CustomGaussianNB, CustomDecisionTree, MLP, VGG11

def evaluate_model(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"--- {model_name} ---")
    print(f"accuracy: {acc:.4f}, precision: {prec:.4f}, recall: {rec:.4f}, F1: {f1:.4f}")
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'confusion matrix: {model_name}')
    plt.ylabel('true label')
    plt.xlabel('predicted label')
    plt.savefig(f'cm_{model_name.replace(" ", "_")}.png')
    plt.close()

def train_pytorch_model(model, train_loader, epochs=10, learning_rate=0.01, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    model.to(device)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    return model

def main():
    #load data
    (X_train, y_train), (X_test, y_test) = process_data()
    
    # NAIVE BAYES
    nb_custom = CustomGaussianNB()
    nb_custom.fit(X_train, y_train)
    evaluate_model(y_test, nb_custom.predict(X_test), "NB Custom")
    
    # Scikit-Learn
    nb_sklearn = GaussianNB()
    nb_sklearn.fit(X_train, y_train)
    evaluate_model(y_test, nb_sklearn.predict(X_test), "NB Sklearn")

    # DECISION TREE
    dt_custom = CustomDecisionTree(max_depth=10) 
    dt_custom.fit(X_train, y_train)
    evaluate_model(y_test, dt_custom.predict(X_test), "DT Custom")
    
    # Scikit-Learn
    dt_sklearn = DecisionTreeClassifier(criterion='gini', max_depth=50)
    dt_sklearn.fit(X_train, y_train)
    evaluate_model(y_test, dt_sklearn.predict(X_test), "DT Sklearn")

    # MLP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tensor = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_tensor = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    train_loader_mlp = DataLoader(train_tensor, batch_size=64, shuffle=True)
    mlp = MLP()
    mlp = train_pytorch_model(mlp, train_loader_mlp, epochs=20, device=device)
    mlp.eval()
    all_preds = []
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test).to(device)
        outputs = mlp(X_test_t)
        _, predicted = torch.max(outputs.data, 1)
        all_preds = predicted.cpu().numpy()
    evaluate_model(y_test, all_preds, "MLP")




    # CNN (VGG11)
    transform_cnn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_raw, test_raw = get_cifar10_data(target_type='raw')
    train_loader_cnn = DataLoader(train_raw, batch_size=32, shuffle=True)
    test_loader_cnn = DataLoader(test_raw, batch_size=32, shuffle=False)
    cnn = VGG11()
    cnn = train_pytorch_model(cnn, train_loader_cnn, epochs=10, device=device)
    cnn.eval()
    all_preds_cnn = []
    all_labels_cnn = []
    with torch.no_grad():
        for images, labels in test_loader_cnn:
            images = images.to(device)
            outputs = cnn(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds_cnn.extend(predicted.cpu().numpy())
            all_labels_cnn.extend(labels.numpy())
    evaluate_model(all_labels_cnn, all_preds_cnn, "CNN VGG11")
    torch.save(mlp.state_dict(), "mlp_model.pth")
    torch.save(cnn.state_dict(), "cnn_model.pth")

if __name__ == "__main__":
    main()