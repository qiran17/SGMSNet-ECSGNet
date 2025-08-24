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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':
    # 需要根据需求进行更改的参数
    # ————————————————————————————————————————————————————————————————
    test_dir = '../AIDtp0.2/test'
    model_path = "../ECSGNet_AIDtp0.2_time1-best.pth"
    num_class = 30
    batch_size = 16
    show_figure = True
    normalization = True
    # 模型不同类别的名称，默认已0，1，2，3......代替不同的类别名称，想要自定义的话，可以自己传列表进来。
    display_labels =('Airport', 'Bareland', 'Baseballfield', 'Beach', 'Bridge', 'Center', 'Church', 'Commercial', 'DenseResidential',
    'Desert', 'Farmland', 'Forest', 'Industrial', 'Meadow', 'MediumResidential', 'Mountain', 'Park', 'Parking',
    'Playground', 'Pond', 'Port', 'RailwayStation', 'Resort', 'River', 'School', 'SparseResidential', 'Square',
    'Stadium', 'StorageTanks', 'Viaduct')
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


    class CustomImageFolder(datasets.DatasetFolder):
        def __init__(self, root, transform=None, target_transform=None, loader=Image.open):
            super().__init__(root, loader, extensions=None, transform=transform, target_transform=target_transform,
                             is_valid_file=lambda x: True)

        def __getitem__(self, index):
            path, target = self.samples[index]
            try:
                sample = self.loader(path)
            except Exception as e:
                print(f"Error loading image: {path}")
                return None, None

            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

            return sample, target

    # 加载数据
    test_data = CustomImageFolder(root=test_dir, transform=transform)

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
            _ , outputs = net(inputs,inputs)
            outputs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize='true')
    formatted_cm = np.round(cm, decimals=2)
    # 打印混淆矩阵
    print("Confusion Matrix: ")
    for i in range(0, len(formatted_cm)):
        print(formatted_cm[i])
        print("____________________________________________")

    formatted_cm[formatted_cm == 0] = None

    # 画出混淆矩阵
    disp = ConfusionMatrixDisplay(confusion_matrix=formatted_cm, display_labels=display_labels)
    plt.figure(figsize=(50, 50))
    plt.rcParams['font.size'] = 2
    disp.plot(include_values="no_zero", cmap="Blues", ax=None, xticks_rotation=90)

    plt.savefig('../ConfusionMatrix_AID.png', dpi=500, bbox_inches='tight')
    plt.show()

