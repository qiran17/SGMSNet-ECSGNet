#from models import mobilenetv2, ResNeXt, SENet
#from models.SGMSNet.ECSGNet import ECSGNet_s
#from models.ResNet_Atten6 import resnet50
import torch.nn as nn

model_list = ['SGMSNet场景分类']


def get_model(model_name):
    if model_name == 'SGMSNet场景分类':
        model_path = r'F:\PyCharm 2022.2\program\Project_practice\classification_GUI\models\fmodels\ECSGNet_RE45tp0.2_time2-best.pth'
        model_class = ['airplane', 'airport', 'baseball_diamond', 'basketball_court', 'beach', 'bridge', 'chaparral',
                       'church', 'circular_farmland', 'cloud', 'commercial_area', 'dense_residential', 'desert',
                       'forest', 'freeway', 'golf_course', 'ground_track_field', 'harbor', 'industrial_area',
                       'intersection', 'island', 'lake', 'meadow', 'medium_residential', 'mobile_home_park', 'mountain',
                       'overpass', 'palace', 'parking_lot', 'railway', 'railway_station', 'rectangular_farmland',
                       'river', 'roundabout', 'runway', 'sea_ice', 'ship', 'snowberg', 'sparse_residential', 'stadium',
                       'storage_tank', 'tennis_court', 'terrace', 'thermal_power_station', 'wetland']

    return  model_path, model_class
