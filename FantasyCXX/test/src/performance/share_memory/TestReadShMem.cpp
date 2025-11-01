//
// Created by Frewen.Wang on 2022/11/20.
//
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional> // 包含 std::greater
#include "aura/aura_utils/utils/AuraLog.h"
#include "gtest/gtest.h"
#include "shmDatadef.h"

const static char *TAG = "TestReadShMem";

using namespace std;

/**
 * 文章参考：
 * https://zhaoxuhui.top/blog/2019/08/21/eigen-note-1.html#1%E5%BA%93%E7%9A%84%E5%AE%89%E8%A3%85
 */
class TestReadShMem : public testing::Test {
public:
	static void SetUpTestSuite() {
		ALOGD(TAG, "SetUpTestSuite");
	}
	
	static void TearDownTestSuite() {
		ALOGD(TAG, "TearDownTestSuite");
	}
};

TEST_F(TestReadShMem, testStdGreater) {
	ALOGD(TAG, "============== testStdGreater ==============");
	void *shm = NULL;
	struct stuShareMemory *stu;
	// 创建共享内存
	// key_t:    长整型（唯一非零），系统建立IPC通讯 （ 消息队列、 信号量和 共享内存） 时必须指定一个ID值。
	//               通常情况下，该id值通过ftok函数得到，由内核变成标识符，要想让两个进程看到同一个信号集，只需设置key值不变就可以。
	// size:     指定共享内存的大小，它的值一般为一页大小的整数倍（未到一页，操作系统向上对齐到一页，但是用户实际能使用只有自己所申请的大小）。
	// shmflg：  是一组标志，创建一个新的共享内存，将shmflg 设置了IPC_CREAT标志后，共享内存存在就打开。
	//               而IPC_CREAT | IPC_EXCL则可以创建一个新的，唯一的共享内存，如果共享内存已存在，返回一个错误。一般我们会还或上一个文件权限
	//  成功返回共享内存的ID, 出错返回-1
	//  详解参数传入： 创建功能内存的ID值是1234， 然后申请的大小是一个结构体的大小，IPC_CREAT标志共享内存存在就打开。
	int shmid = shmget((key_t)1234, sizeof(struct stuShareMemory), 0666|IPC_CREAT);
	if(shmid == -1) {
		printf("shmget err.\n");
		return;
	}
	//  挂接操作 —— 创建共享存储段之后，将进程连接到它的地址空间
	// shm_id ：是由shmget函数返回的共享内存标识。
	// shm_addr ：指定共享内存连接到当前进程中的地址位置，通常为空，表示让系统来选择共享内存的地址。
	// shm_flg ：是一组标志位，通常为 0。
	shm = shmat(shmid, (void*)0, 0);
	if(shm == (void*)-1)
	{
		printf("shmat err.\n");
		return;
	}

	stu = (struct stuShareMemory*)shm;

	stu->iSignal = 1;

	//while(true)  //如果需要多次写入，可以启用while
	{
		if(stu->iSignal != 0)
		{
			printf("current txt : %s", stu->chBuffer);
			stu->iSignal = 0;
		}
		else
		{
			sleep(10);
		}
	}

	shmdt(shm);
	// 操作共享内存
	// 参数：
	//  shmid 是shmget函数返回的共享内存标志符
	//  cmd: 要采取的操作，它可以取下面的三个值：
	//      IPC_STAT：把shmid_ds结构中的数据设置为共享内存的当前关联值，即用共享内存的当前关联值覆盖shmid_ds的值。
	//      IPC_SET：如果进程有足够的权限，就把共享内存的当前关联值设置为shmid_ds结构中给出的值。
	//       IPC_RMID：删除共享内存段。
	//   buf ：一个结构指针，它指向共享内存模式和访问权限的结构。
	//        shmid_ds结构至少包括以下成员
	//         struct shmid_ds
	//            {
	//              uid_t shm_perm.uid;
	//              uid_t shm_perm.gid;
	//              mode_t shm_perm.mode;
	//            };
	//  int shmctl(int shm_id, int cmd, struct shmid_ds *buf);
	// 成功返回0，出错返回-1
	shmctl(shmid, IPC_RMID, 0);

	std::cout << "end progress." << endl;
}

