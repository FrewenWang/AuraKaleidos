//
// Created by Frewen.Wang on 25-3-23.
//

#ifndef SHMDATADEF_H
#define SHMDATADEF_H

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/shm.h>
#include <iostream>
using namespace std;

#define SHARE_MEMORY_BUFFER_LEN 1024

/**
* 定义共享内存的结构体
*/
struct stuShareMemory{
	int iSignal;
	char chBuffer[SHARE_MEMORY_BUFFER_LEN];
	
	stuShareMemory(){
		iSignal = 0;
		memset(chBuffer,0,SHARE_MEMORY_BUFFER_LEN);
	}
};

#endif //SHMDATADEF_H
