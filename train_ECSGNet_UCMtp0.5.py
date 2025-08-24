import torch
from torchvision import transforms, datasets
from PIL import Image
import math
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from center_loss import CenterLoss
from ECSGNet_module.ECSGNet import ECSGNet_s
import torch.optim.lr_scheduler as lr_scheduler

model_name='ECSGNet_UCMtp0.5_time3'
# 定义数据集路径
train_dir = './UCMtp0.5/train'
val_dir = './UCMtp0.5/val'
test_dir = './UCMtp0.5/test'
# batch_size与类别数
batch_size = 16
n_classes = 21
# 定义数据预处理
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
        super().__init__(root, loader, extensions=None,transform=transform, target_transform=target_transform,
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

# 创建训练集、验证集和测试集的数据加载器
train_data = CustomImageFolder(root=train_dir, transform=transform)
val_data = CustomImageFolder(root=val_dir, transform=transform)
test_data = CustomImageFolder(root=test_dir, transform=transform)
# 过滤掉加载失败的样本
train_data.samples = [(path, target) for path, target in train_data.samples if path is not None]
val_data.samples = [(path, target) for path, target in val_data.samples if path is not None]
# 加载数据集
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)


# 定义模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ECSGNet_s(pretrained=True)
num_fr = model.head.in_features
model.head = nn.Linear(num_fr, n_classes)
weights_dict = torch.load('pre_model/SG-Former-S.pth', map_location=device)["state_dict_ema"]  # 预训练模型
# 删除有关分类类别的权重
for k in list(weights_dict.keys()):
    if "head" in k:
        del weights_dict[k]
model.load_state_dict(weights_dict, strict=False)
model.to(device)

# 定义损失函数和优化器
# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9,weight_decay=5e-4)
# 中心损失函数
loss_weight = 0.001
center_loss = CenterLoss(num_classes=21, feat_dim=512, use_gpu=True)
optimzer_center = optim.SGD(model.parameters(), lr=0.001, momentum=0.9,weight_decay=5e-4)

lf = lambda x: ((1 + math.cos(x * math.pi / 100)) / 2) * (1 - 0.01) + 0.01  # cosine
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # 设定优优化器更新的时刻表

# 训练参数
epochs = 50  # 定义训练轮数
train_losses = []
val_losses = []
train_acc = []
val_acc = []
Best_ACC = 0  # 记录最高得分

# 训练阶段
def train(epoch):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    step = 1
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        feats, outputs = model(inputs,inputs)
        loss = criterion(outputs, labels) + loss_weight * center_loss(feats, labels)

        optimizer.zero_grad()
        optimzer_center.zero_grad()
        loss.backward()
        optimizer.step()
        optimzer_center.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        # print
        train_loss = loss.item()
        train_accuracy = (predicted == labels).sum().item() / labels.size(0) * 100
        print(f"Epoch [{epoch + 1}/{epochs}], "
              f"Step [{step}/{len(train_loader)}], "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        step += 1

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train
    train_losses.append(train_loss)
    train_acc.append(train_accuracy)

    # 验证阶段
def val(epoch):
    global Best_ACC
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            feats, outputs = model(inputs,inputs)
            loss = criterion(outputs, labels) + loss_weight * center_loss(feats,labels)

            running_val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_loss = running_val_loss / len(val_loader)
    val_accuracy = 100 * correct_val / total_val
    acc = val_accuracy
    # 统计最佳val精度
    if acc > Best_ACC:
        torch.save(model, f"{model_name}-best.pth")
        Best_ACC = acc
    val_losses.append(val_loss)
    val_acc.append(val_accuracy)

    print(f"Epoch [{epoch + 1}/{epochs}] - "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")


def lr_decay():
    global optimizer
    for params in optimizer.param_groups:
        params['lr'] *= 0.1
        lr = params['lr']
        print("Learning rate adjusted to {}".format(lr))


def main():
    for epoch in range (epochs):
        train(epoch)
        scheduler.step()
        val(epoch)
        if (epoch + 1) % 10 == 0:
            lr_decay()
    print(f"best Acc: {Best_ACC:.2f}%")
    # 保存模型
    # torch.save(model, f"{model_name}.pth")

    # 绘制损失和精度曲线
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('loss_acc.jpg')
    plt.show()

if __name__ == '__main__':
    main()