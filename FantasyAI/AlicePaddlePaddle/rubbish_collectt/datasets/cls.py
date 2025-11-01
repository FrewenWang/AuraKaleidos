import os
import random


#
def read_file_datas():
    FileNamelist = []
    # 读取train.txt中的文件内筒
    file = open('train.txt', 'r+')
    # 逐行读取
    for line in file:
        line = line.strip('\n')  # 删除每一行的\n
        FileNamelist.append(line)
    # print('len ( FileNamelist ) = ' ,len(FileNamelist))
    file.close()
    return FileNamelist


def write_datas_to_file(listInfo):
    file_handle_train = open('train.txt', mode='w')
    file_handle_eval = open("eval.txt", mode='w')
    i = 0
    for idx in range(len(listInfo)):
        str = listInfo[idx]
        # 查找最后一个 “_”的位置
        ndex = str.rfind('_')
        # print('ndex = ',ndex)
        # 截取字符串
        str_houZhui = str[(ndex + 1):]
        # print('str_houZhui = ',str_houZhui)
        str_Result = str + '\n'  # + str_houZhui+'\n'
        # print(str_Result)
        if (i % 6 != 0):
            file_handle_train.write(str_Result)
        else:
            file_handle_eval.write(str_Result)
        i += 1
    file_handle_train.close()
    file_handle_eval.close()


path = "../"
res = os.listdir(path)
print(res)

with open("train.txt", "w") as f:
    for i in res:
        if (os.path.isdir(i)):
            path1 = path + i
            res2 = os.listdir(path1)
            for j in res2:
                f.write(path1 + "/" + j + " " + i + "\n")

listFileInfo = read_file_datas()
# 打乱列表中的顺序
random.shuffle(listFileInfo)
write_datas_to_file(listFileInfo)
