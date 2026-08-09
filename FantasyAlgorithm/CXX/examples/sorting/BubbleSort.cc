//
// Created by Frewen.Wang on 25-3-13.
//
#include<iostream>

using namespace std;

class Solution {
public:
    void BubbleSort(int arr[], int size) {
        for (int i = 0; i < size - 1; ++i)
            // 注意冒泡排序的算法后面是因为已经排序过的。
            for (int j = 0; j < size - 1 - i; ++j)
                if (arr[j] > arr[j + 1]) {
                    std::swap(arr[j], arr[j + 1]);
                }
    }

};