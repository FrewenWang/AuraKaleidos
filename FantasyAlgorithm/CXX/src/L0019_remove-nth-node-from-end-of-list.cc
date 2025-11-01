//
// Created by frewen on 25-2-28.
//
//19. 删除链表的倒数第 N 个结点
//题目链接：https://leetcode.cn/problems/remove-nth-node-from-end-of-list/description/?envType=study-plan-v2&envId=top-interview-150
//给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。
//
//
//
//示例 1：
//
//
//输入：head = [1,2,3,4,5], n = 2
//输出：[1,2,3,5]
//示例 2：
//
//输入：head = [1], n = 1
//输出：[]
//示例 3：
//
//输入：head = [1,2], n = 1
//输出：[1]
//
//
//提示：
//
//链表中结点的数目为 sz
//1 <= sz <= 30
//0 <= Node.val <= 100
//1 <= n <= sz
//
//
//进阶：你能尝试使用一趟扫描实现吗？

struct ListNode {
     int val;
     ListNode *next;
     ListNode() : val(0), next(nullptr) {}
     ListNode(int x) : val(x), next(nullptr) {}
     ListNode(int x, ListNode *next) : val(x), next(next) {}
};
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
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        // 首先我们先计算这个单链表的所有节点的个数
        int total = 0;
        // 这个计算所有节点个数
        ListNode *count_node = head;
        while(count_node) {
            total++;
            count_node = count_node->next;
        }

        ListNode *dummy = new ListNode(0,head);
        int index = 0;
        ListNode *curr = dummy;
        int removeIndex = total - n;
        while(curr) {
            if(index  == removeIndex && curr->next) {
                curr->next = curr->next->next;
            } else if(index  == removeIndex && !curr->next) {
                curr->next = nullptr;
                break;
            }
            index++;
            curr = curr->next;
        }
        return dummy->next;
    }
};