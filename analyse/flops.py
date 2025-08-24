import torch
from thop import profile
#from ECSGNet_module.ECSGNet import ECSGNet_s
from  sgformer.sgformer import sgformer_s
# from
# 定义模型
model = sgformer_s()
input = torch.randn(16, 3, 224, 224)  # 以模型期望的输入尺寸为例

# 计算 FLOPs
flops, params = profile(model, inputs=(input, ))
print(f"FLOPs: {flops / 10**9} G")
print(f"Params: {params / 10**6} M")
