import torchvision

# 准备的测试数据集(DataLoader里面)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备好对应的dataset数据集
# CIFAR10: 60000张32x32的彩色图片，10个类别，50000训练集，10000测试集
# CIFAR10数据集的标签

# test_data = torchvision.datasets.CIFAR10("./data", train=False, transform=torchvision.transforms.ToTensor())
# 如果之前数据没有下载过，则需要下载
test_data = torchvision.datasets.CIFAR10(
    "./data", train=False, transform=torchvision.transforms.ToTensor(), download=True)

# /**
#  * DataLoader: 封装了数据集，并支持多线程读取数据
#  * 参数：
#  *      dataset: 数据集
#  *      batch_size: 批次大小(每次加载的数据)
#  *      shuffle: 是否打乱数据（是否打乱数据）
#  *      num_workers: 多线程读取数据
#  *      drop_last: 丢弃最后一个不完整的批次
#  */
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

# 测试数据集中第一张图片及target。打印图片及标签
img, target = test_data[0]
print(img.shape)
print(target)

# 进行调用for循环，来进行在测试集进行遍历
writer = SummaryWriter("logs")
for epoch in range(1):
    #如果：drop_last=True 每批次是634张图片，我们一共10000张测试图片，所以一共10000/64=156.25，向下取整为156
    #如果：drop_last=False 每批次是634张图片，我们一共10000张测试图片，所以一共10000/64=156.25，向上取整为157
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print("imgs.shape: ", imgs.shape)
        # print("targets: ", targets)
        # imgs.shape:  torch.Size([4, 3, 32, 32])    # 4张三通道的32*32图片
        # targets:  tensor([3, 5, 8, 7])             # 4张图片的标签
        writer.add_images("Epoch: {}".format(epoch), imgs, step)
        step = step + 1

writer.close()

# 上面的代码执行率完毕之后：
# torch.Size([3, 32, 32])
# 3
# 然后接着执行：
# tensorboard --logdir="logs"
# 打开浏览器查看
# http://localhost:6006/ 
