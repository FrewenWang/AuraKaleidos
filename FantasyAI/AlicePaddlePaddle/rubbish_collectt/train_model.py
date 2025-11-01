from paddlex import transforms as T
import paddlex as pdx

# 数据训练的增强的算子
train_transforms = T.Compose([
    T.RandomCrop(crop_size=224),
    T.RandomHorizontalFlip(),
    T.Normalize()])

# 定义评价集的数据增强算子
eval_transforms = T.Compose([
    T.ResizeByShort(short_size=256),
    T.CenterCrop(crop_size=224),
    T.Normalize()
])

# 定义训练集
# 数据集目录
#
train_dataset = pdx.datasets.ImageNet(
    data_dir='datasets',
    file_list='datasets/train.txt',
    label_list='datasets/labels.txt',
    transforms=train_transforms,
    shuffle=True)

# 定义评价集
# 数据集目录
#
eval_dataset = pdx.datasets.ImageNet(
    data_dir='datasets',
    file_list='datasets/eval.txt',
    label_list='datasets/labels.txt',
    transforms=eval_transforms)

num_classes = len(train_dataset.labels)
# 定义分类模型
model = pdx.cls.MobileNetV3_small(num_classes=num_classes)

#  开始进行模型训练
# 训练的轮数
# 单次导入显卡(CPU)的图片张数
# lr_decay_epochs
# num_epochs 训练的轮数
# lr_decay_epochs  每次一下子导入到显卡中的图片的数据
# eval_dataset=eval_dataset,  评价集
# lr_decay_epochs=[4, 6, 8], 学习率的衰减(也就是步长)，我们现在用的比较简单4、6、8直接衰减
model.train(num_epochs=10,
            train_dataset=train_dataset,
            train_batch_size=64,
            eval_dataset=eval_dataset,
            lr_decay_epochs=[4, 6, 8],
            save_dir='output/mobilenetv3_small',
            use_vdl=True)
