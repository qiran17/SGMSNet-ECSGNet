import torch
import torchvision
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils_2 import ConfusionMatrix
import json

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    test_dir = '../AIDtp0.2/test'
    # train_set = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transforms.ToTensor())
    # train_set = torchvision.datasets.SVHN(root='./data',  download=True, transform=transforms.ToTensor())
    # data = torch.cat([d[0] for d in DataLoader(train_set)])
    # mean=data.mean(dim=[0, 2, 3])
    # std=data.std(dim=[0, 2, 3])
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
    #validate_dataset = datasets.CIFAR10(root='./data', train=False, transform=train_transform)
    test_data = CustomImageFolder(root=test_dir, transform=transform)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=True)

    # 加载网络
    #net = ESGormer_s()
    model_weight_path = "../ECSGNet_AIDtp0.2_time1-best.pth"
    # net.load_state_dict(torch.load(model_weight_path, map_location=device))
    net = torch.load(model_weight_path)
    net.to(device)

    # 类别
    #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    #classes = ("airplane", "airport", "baseball_diamond", "basketball_court", "beach", "bridge", "chaparral", "church", "circular_farmland", "cloud", "commercial_area", "dense_residential", "desert", "forest", "freeway", "golf_course", "ground_track_field", "harbor", "industrial_area", "intersection", "island", "lake", "meadow", "medium_residential", "mobile_home_park", "mountain", "overpass", "palace", "parking_lot", "railway", "railway_station", "rectangular_farmland", "river", "roundabout", "runway", "sea_ice", "ship", "snowberg", "sparse_residential", "stadium", "storage_tank", "tennis_court", "terrace", "thermal_power_station", "wetland")
    classes = ('Airport', 'Bareland', 'Baseballfield', 'Beach', 'Bridge', 'Center', 'Church', 'Commercial', 'DenseResidential', 'Desert', 'Farmland', 'Forest', 'Industrial', 'Meadow', 'MediumResidential', 'Mountain', 'Park', 'Parking', 'Playground', 'Pond', 'Port', 'RailwayStation', 'Resort', 'River', 'School', 'SparseResidential', 'Square', 'Stadium', 'StorageTanks' ,'Viaduct')
    #classes = ('agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings', 'chaparral', 'denseresidential', 'forest', 'freeway', 'golfcourse', 'harbor', 'intersection', 'mediumresidential', 'mobilehomepark', 'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential', 'storagetanks', 'tenniscourt')
    labels = [label for label in classes]
    confusion = ConfusionMatrix(num_classes=30, labels=labels)#RE45:45,AID:30,UCM:21

    net.eval()
    with torch.no_grad():
        for test_data in tqdm(test_loader):
            test_images, test_labels = test_data
            _ , outputs = net(test_images.to(device),test_images.to(device))
            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            confusion.update(outputs.to("cpu").numpy(), test_labels.to("cpu").numpy())  # 更新混淆矩阵的值
    confusion.summary()  # 计算指标
