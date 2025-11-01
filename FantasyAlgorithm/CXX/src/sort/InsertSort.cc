//
// Created by Frewen.Wang on 25-3-13.
//
#include<iostream>

using namespace std;

class Solution {
public:
  /**
   *  时间复杂度（O(n^2)）
   */
  void InsertSort(int arr[], int size) {
    int temp;
    int j;
    /// 默认第一个元素已经是排序完成的。
    for (int i = 1; i < size; i++) {
      // 找到待插入的排序的数据
      temp = arr[i];
      /// 并且找到当前待插入数据的索引处。并且认为上一个上一个数据已经是插入完成的
      // 所以其实我们就是看我们的数据
      j = i - 1;
      // 判断之前排序完成的数据，是否够大于我们这个temp的待排数据。
      // 如果大于。我们我们其实就要进行向后移动。
      // 也就是 arr[j + 1] = arr[j]
      // 然后不断地j--
      while (j >= 0 && arr[j] > temp) {
        arr[j + 1] = arr[j];
        j--;
      }
      /// 知道找到这个对应的j. 所以这个对应的j的下一个问题就是我们的temp
      arr[j + 1] = temp;
    }
  }

};