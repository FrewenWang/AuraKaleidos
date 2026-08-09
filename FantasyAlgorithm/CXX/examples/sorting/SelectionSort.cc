//
// Created by Frewen.Wang on 25-3-13.
//
#include<iostream>

using namespace std;

class Solution {
public:
  /**
    *  时间复杂度：O(n^2)
    */
  void selectionSort(vector<int> &a){
    int len = a.size();
    int minIndex = 0;
    /// 因为他要和后面的数据进行比较，所以他小于size() -1
    for (int i = 0; i < len - 1; i++) //需要循环次数
    {
      ///
      minIndex = i;                     //最小下标
      for (int j = i + 1; j < len; j++) //访问未排序的元素
      {
        //
        if (a[j] < a[minIndex]) {
          minIndex = j; //找到最小的
        }

      }
      // 然后交换到a[i] 和 a[minIndex
      swap(a[i], a[minIndex]);
    }
  }

};