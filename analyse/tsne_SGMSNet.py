import os
import time
import warnings
import matplotlib.pyplot as plt
from PIL import Image
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

if __name__ == '__main__':
    # 需要根据需求进行更改的参数
    # ————————————————————————————————————————————————————————————————
    test_dir = '../RE45tp0.2/test'
    model_path = "../ECSGNet_RE45tp0.2_time2-best.pth"
    num_classes = 45
    batch_size = 16
    show_figure = True
    normalization = True
    display_labels = ("1", "2", "3", "4", "5", "6", "7", "8",
                      "9", "10", "11", "12", "13", "14", "15",
                      "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29",
                      "30", "31", "32", "32", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45")
    # ————————————————————————————————————————————————————————————————
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 3.0)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.44127703, 0.4712498, 0.43714803], std=[0.18507297, 0.18050247, 0.16784933])

    ])

    # 加载数据
    test_data = datasets.ImageFolder(root=test_dir, transform=transform)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    net = torch.load(model_path)
    net.to(device)

    y_true = []
    y_pred = []

    net.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.cuda() if torch.cuda.is_available() else inputs
            labels = labels.cuda() if torch.cuda.is_available() else labels
            _, outputs = net(inputs, inputs)
            y_true.append(labels.cpu().numpy())
            y_pred.append(outputs.cpu().numpy())

    features = np.concatenate(y_pred, axis=0)
    labels = np.concatenate(y_true, axis=0)

    # 使用 t-SNE 降维
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    features_2d = tsne.fit_transform(features)

    # 可视化
    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap("tab10", num_classes)

    for class_idx in range(num_classes):
        idx = labels == class_idx
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1], label=display_labels[class_idx], alpha=0.7, s=15)

    # plt.legend(loc='best', fontsize='small')
    plt.legend(fontsize='8', loc='upper center', bbox_to_anchor=(1.06, 1.16), ncol=1)  # 1列图例
    plt.title('SGMSNet')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True)
    plt.show()
