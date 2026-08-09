//
// Created by frewen on 25-2-28.
//
//82. 删除排序链表中的重复元素 II
//    题目链接：https://leetcode.cn/problems/remove-duplicates-from-sorted-list-ii/description/?envType=study-plan-v2&envId=top-interview-150
//给定一个已排序的链表的头 head ， 删除原始链表中所有重复数字的节点，只留下不同的数字 。返回 已排序的链表 。
//
//
//
//示例 1：
//
//
//输入：head = [1,2,3,3,4,4,5]
//输出：[1,2,5]
//示例 2：
//
//
//输入：head = [1,1,1,2,3]
//输出：[2,3]
//
//
//提示：
//
//链表中节点数目在范围 [0, 300] 内
//-100 <= Node.val <= 100
//题目数据保证链表已经按升序 排列
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */

struct ListNode {
     int val;
     ListNode *next;
     ListNode() : val(0), next(nullptr) {}
     ListNode(int x) : val(x), next(nullptr) {}
     ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution {
public:
  ListNode* deleteDuplicates(ListNode* head) {

    int target = -1000;
    int target_count = 0;
    // 先定义一个头结点
    ListNode* dummy = new ListNode(0, head);

    // 首先让要返回的目标节点指向头部的这个哑巴节点
    // 重点一： 这个是之前没有想到的。 我们可以定义一个哑巴节点。
    // 我们先让我们的目标链表指向哑巴节点。
    // 顶多到最后，我们返回dummy->next的节点就可以了！！！
    ListNode* curr = dummy;
    // 进行while循环。我们依次进行循环两个，也就是头结点后面的两个节点
    // 判断这两个节点是否相同。
    // 如果不相同的话，我们就可以让我们的目标链表指向下一个节点。因为不相同，说明只有一个数据
    // 如果的话，我们记录下来下一个节点和下下个节点的值。并且把这个值存储下来
    // 然后，只有后面的节点。一直等于这个值，我们就让next的节点。一直指向next next
    while(curr->next && curr->next->next) {
      if(curr->next->val == curr->next->next->val) {
        int repeatVal = curr->next->val;
        while(curr->next && curr->next->val == repeatVal) {
          curr->next = curr->next->next;
        }

      } else {
        curr = curr->next;
      }
    }
    return dummy->next;
  }
};