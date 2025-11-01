"""
LMDB工具类
"""
import os
try:
    import lmdb
except:
    os.system('pip3 install lmdb')
    import lmdb
import pickle
import numpy as np

class LMDB(object):
    """
    LMDB工具类
    """
    def __init__(self, lmdb_path, mode='w'):
        self.mode = mode
        self.ct = 0
        if mode == 'w':
            self.env = lmdb.open(lmdb_path, map_size=1099511627776)
            self.txn = self.env.begin(write=True)
        elif mode == 'r':
            self.env = lmdb.open(lmdb_path)
            self.txn = self.env.begin()
        else:
            raise ValueError("mode can only be w/r.")

    def insert(self, record):
        """
        插入单个数据 key: 自增整型id, val：pickle格式的图片与标注内容
        :param record:
        :return:
        """
        if self.mode != 'w':
            raise ValueError("insert can only be used in write mode.")
        self.txn.put(str(self.ct).encode(), pickle.dumps(record))
        self.ct += 1

    def read_single(self):
        """
        TODO: 未完成
        :return:
        """
        if self.mode != 'r':
            raise ValueError("read_single can only be used in read mode.")
        cursor = self.txn.cursor()

    def read_all(self):
        """
        使用curosor遍历全部数据，通过list返回所有图片与标注数据
        :return:
        """
        if self.mode != 'r':
            raise ValueError("read_single can only be used in read mode.")
        records = []
        for key, val in self.txn.cursor():
            rec = pickle.loads(val)
            records.append(rec)
        return records

    def write_lmdb(self):
        """
        数据插入完毕后，调用此函数commit才完成数据写入
        :return:
        """
        if self.mode != 'w':
            raise ValueError("write_lmdb can only be used in write mode.")
        self.txn.commit()
        self.env.close()