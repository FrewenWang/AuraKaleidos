//
// Created by Frewen.Wang on 25-3-13.
//
//实现一个拥有如下方法的线程安全有限阻塞队列：
//
//BoundedBlockingQueue(int capacity) 构造方法初始化队列，其中capacity代表队列长度上限。
//void enqueue(int element) 在队首增加一个element. 如果队列满，调用线程被阻塞直到队列非满。
//int dequeue() 返回队尾元素并从队列中将其删除. 如果队列为空，调用线程被阻塞直到队列非空。
//int size() 返回当前队列元素个数。
//你的实现将会被多线程同时访问进行测试。每一个线程要么是一个只调用enqueue方法的生产者线程，要么是一个只调用dequeue方法的消费者线程。size方法将会在每一个测试用例之后进行调用。
//
//请不要使用内置的有限阻塞队列实现，否则面试将不会通过。


#include <semaphore.h>
#include <queue>

using namespace std;

class BoundedBlockingQueue {
public:
  BoundedBlockingQueue(int capacity) {
    sem_init(&s1, 0, capacity);
    sem_init(&s2, 0, 0);
  }

  void enqueue(int element) {
    sem_wait(&s1);
    q.push(element);
    sem_post(&s2);
  }

  int dequeue() {
    sem_wait(&s2);
    int ans = q.front();
    q.pop();
    sem_post(&s1);
    return ans;
  }

  int size() {
    return q.size();
  }

private:
  queue<int> q;
  sem_t s1, s2;
};