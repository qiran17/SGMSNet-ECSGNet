import os


def count_images(path):
    image_count = 0

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".tif"):  # 根据需要修改文件类型后缀名
                image_count += 1

    return image_count


# 调用函数并输出结果
image_num1 = count_images("./UCMtp0.5/train")
image_num2 = count_images("./UCMtp0.5/val")
image_num3 = count_images("./UCMtp0.5/test")
print("UCMtp0.5train数据集数量为:", image_num1)
print("UCMtp0.5val数据集数量为:", image_num2)
print("UCMtp0.5test数据集数量为:", image_num3)

image_num4 = count_images("./UCMtp0.8/train")
image_num5 = count_images("./UCMtp0.8/val")
image_num6 = count_images("./UCMtp0.8/test")

print("UCMtp0.8train数据集数量为:", image_num4)
print("UCMtp0.8val数据集数量为:", image_num5)
print("UCMtp0.8test数据集数量为:", image_num6)

image_num7 = count_images("./AIDtp0.2/train")
image_num8 = count_images("./AIDtp0.2/val")
image_num9 = count_images("./AIDtp0.2/test")

print("AIDtp0.2rain数据集数量为:", image_num7)
print("AIDtp0.2val数据集数量为:", image_num8)
print("AIDtp0.2test数据集数量为:", image_num9)

image_num10 = count_images("./AIDtp0.5/train")
image_num11 = count_images("./AIDtp0.5/val")
image_num12 = count_images("./AIDtp0.5/test")

print("AIDtp0.5rain数据集数量为:", image_num10)
print("AIDtp0.5val数据集数量为:", image_num11)
print("AIDtp0.5test数据集数量为:", image_num12)

image_num13 = count_images("./RE45tp0.2/train")
image_num14 = count_images("./RE45tp0.2/val")
image_num15 = count_images("./RE45tp0.2/test")

print("RE45tp0.2train数据集数量为:", image_num13)
print("RE45tp0.2val数据集数量为:", image_num14)
print("RE45tp0.2test数据集数量为:", image_num15)

image_num16 = count_images("./RE45tp0.5/train")
image_num17 = count_images("./RE45tp0.5/val")
image_num18 = count_images("./RE45tp0.5/test")

print("RE45tp0.5train数据集数量为:", image_num16)
print("RE45tp0.5val数据集数量为:", image_num17)
print("RE45tp0.5test数据集数量为:", image_num18)