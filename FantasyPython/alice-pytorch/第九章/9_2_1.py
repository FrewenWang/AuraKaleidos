# import numpy as np
#
# max_length = 80         #设置获取的文本长度为80
# labels = []             #用以存放label
# context = []            #用以存放汉字文本
# vocab = set()           #
# with open("../dataset/cn/ChnSentiCorp.txt", mode="r", encoding="UTF-8") as emotion_file:
#     for line in emotion_file.readlines():
#         line = line.strip().split(",")
#
#         # labels.append(int(line[0]))
#         if int(line[0]) == 0:
#             labels.append(0)    #这里由于我们在后面直接采用Pytorch自带的crossentroy函数，所以这里直接输入0，否则输入[1,0]
#         else:
#             labels.append(1)
#         text = "".join(line[1:])
#         context.append(text)
#         for char in text: vocab.add(char)   #建立vocab和vocab编号
#
# voacb_list = list(sorted(vocab))
# # print(len(voacb_list))
# token_list = []
# #下面的内容是对context内容根据vocab进行token处理
# for text in context:
#     token = [voacb_list.index(char) for char in text]
#     token = token[:max_length] + [0] * (max_length - len(token))
#     token_list.append(token)
#
#
# seed = 17
# np.random.seed(seed);np.random.shuffle(token_list)
# np.random.seed(seed);np.random.shuffle(labels)
#
# dev_list = np.array(token_list[:170])
# dev_labels = np.array(labels[:170])
#
# token_list = np.array(token_list[170:])
# labels = np.array(labels[170:])

import torch
from torchvision import models
model = models.resnet18(pretrained = False)

image = torch.rand(size=(3,3,128,128))
result = model(image)
print(result.shape)

# device = "cuda"
# model = get_model().to(device)
# model = torch.compile(model)
# optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
#
# loss_func = torch.nn.CrossEntropyLoss()
#
#
# batch_size = 128
# train_length = len(labels)
# for epoch in (range(21)):
#     train_num = train_length // batch_size
#     train_loss, train_correct = 0, 0
#     for i in (range(train_num)):
#         start = i * batch_size
#         end = (i + 1) * batch_size
#
#         batch_input_ids = torch.tensor(token_list[start:end]).to(device)
#         batch_labels = torch.tensor(labels[start:end]).to(device)
#
#         pred = model(batch_input_ids)
#
#         loss = loss_func(pred, batch_labels.type(torch.uint8))
#
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         train_loss += loss.item()
#         train_correct += ((torch.argmax(pred, dim=-1) == (batch_labels)).type(torch.float).sum().item() / len(batch_labels))
#
#     train_loss /= train_num
#     train_correct /= train_num
#     print("train_loss:", train_loss, "train_correct:", train_correct)
#
#     test_pred = model(torch.tensor(dev_list).to(device))
#     correct = (torch.argmax(test_pred, dim=-1) == (torch.tensor(dev_labels).to(device))).type(torch.float).sum().item() / len(test_pred)
#     print("test_acc:",correct)
#     print("-------------------")
#












