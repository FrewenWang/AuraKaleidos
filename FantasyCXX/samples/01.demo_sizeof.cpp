#include <cstdint>
#include <cstdio>


struct Hello
{
    int id;         // 4个字节
    short age;      // 2个字节
    short agent;    // 2个字节
    double score;   // 8个字节
};

struct st_girl
{
    char name[50];    // 50个字节  // 姓名
    int age;          // 4个字节   // 年龄
    int height;       // 4个字节    // 身高，单位：厘米cm
    char sc[30];      // 30个字节   // 身材，火辣；普通；飞机场。
    char yz[30];      // 30个字节   // 颜值，漂亮；一般；歪瓜裂枣。
};

/**
 * sizeof，看起来还真不简单，总结起来还是一大堆的东西，
 * 不过这是笔试面试中出现比较频繁的，我也是考过才觉得很重要，有些规则如果不注意，还真是拿到一道题目摸不着头脑，
 * 所有总结一下，方面忘记的时候瞄一瞄，其中重点是struct的空间大小部分。
 * @return
 */
int main()
{
    printf("====================普通数据结构sizeof字节数===============================\n");
    /// 在64位的设备上(话说现在基本都是64位的设备)
    printf("sizeof(char)==%lu\n", sizeof(char));    // 1个字节，8位
    printf("sizeof(short)==%lu\n", sizeof(short));
    printf("sizeof(int)==%lu\n", sizeof(int));
    printf("sizeof(float)==%lu\n", sizeof(float));
    printf("sizeof(long)==%lu\n", sizeof(long));
    printf("sizeof(double)==%lu\n", sizeof(double));
    printf("sizeof(long long)==%lu\n", sizeof(long long));

    printf("sizeof(unsigned char)==%lu\n", sizeof(unsigned char));
    printf("sizeof(unsigned short)==%lu\n", sizeof(unsigned short));
    printf("sizeof(unsigned int)==%lu\n", sizeof(unsigned int));
    printf("sizeof(unsigned long)==%lu\n", sizeof(unsigned long));
    printf("sizeof(unsigned long long)==%lu\n", sizeof(unsigned long long));


    printf("====================指针相关结构sizeof字节数===============================\n");
    char *p;
    printf("指针长度8个字节sizeof(double *p)==%lu\n", sizeof(p));
    printf("指针char数据1个字节sizeof(*p)==%lu\n", sizeof(*p));


    printf("====================struct相关结构sizeof字节数=============================\n");
    printf("sizeof(Hello)==%lu\n", sizeof(Hello));


    st_girl queen{};
    printf("sizeof(struct st_girl) %lu\n", sizeof(st_girl));
    printf("sizeof(queen) %lu\n", sizeof(queen));

    printf("====================uint结构sizeof字节数==============================\n");
    printf("sizeof(uint8_t) %lu\n", sizeof(uint8_t));
    printf("sizeof(uint16_t) %lu\n", sizeof(uint16_t));
    printf("sizeof(uint32_t) %lu\n", sizeof(uint32_t));
    printf("sizeof(uint64_t) %lu\n", sizeof(uint64_t));

}
