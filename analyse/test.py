import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# 1. 加载模型和数据集
# 假设你已经有一个训练好的模型
class SimpleModel(nn.Module):
    def __init__(self, pretrained_model):
        super(SimpleModel, self).__init__()
        # 提取特征部分
        self.feature_extractor = nn.Sequential(*list(pretrained_model.children())[:-1])  # 去掉最后的分类层

    def forward(self, x):
        x = self.feature_extractor(x)
        return x.view(x.size(0), -1)  # 拉平成 (batch_size, feature_dim)

# 加载训练好的模型
pretrained_model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
model = SimpleModel(pretrained_model).eval()

# 数据加载
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder(root="path_to_NWPU-RESISC45", transform=data_transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

# 2. 提取特征
features = []
labels = []

with torch.no_grad():
    for inputs, targets in data_loader:
        inputs = inputs.cuda() if torch.cuda.is_available() else inputs
        outputs = model(inputs)
        features.append(outputs.cpu().numpy())
        labels.append(targets.cpu().numpy())

features = np.concatenate(features, axis=0)
labels = np.concatenate(labels, axis=0)

# 3. 使用 t-SNE 降维
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
features_2d = tsne.fit_transform(features)

# 4. 可视化
plt.figure(figsize=(10, 8))
num_classes = len(dataset.classes)
colors = plt.cm.get_cmap("tab10", num_classes)

for class_idx in range(num_classes):
    idx = labels == class_idx
    plt.scatter(features_2d[idx, 0], features_2d[idx, 1], label=dataset.classes[class_idx], alpha=0.7, s=15)

plt.legend(loc='best', fontsize='small')
plt.title('t-SNE Visualization of NWPU-RESISC45 Features')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(True)
plt.show()
