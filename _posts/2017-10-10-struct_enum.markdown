---
layout: post
title: C语言中的结构和联合
date: 2017-10-10 17:58:48 +0800
categories: CS
tags: C 
img: https://raw.githubusercontent.com/wwg1996/wwg1996.github.io/master/images/c.jpg
---
* 
{:toc}

## 结构

> 结构用来存储不同类型的变量

1. **结构声明**
> 形式: `struct tag { member-list } variable-list;`, 成员变量的名字和类型必须声明，标签和变量至少声明其一。

```c
struct {
    int a;
    char b;
    float c;
} x;

struct {
    int a;
    char b;
    float c;
} y[20], *z;

// z = &x  是非法的，无标签情况下
```

```c
struct SIMOLE {
    int a;
    char b;
    float c;
};

struct SIMPLE x;
struct SIMPLE y[20], *z;

z = &x //合法，标签允许多个声明使用同一个成员列表，创建铁一中类型的结构
```

> 还可以使用`typedef`关键词，这时候Simple是个类型名

```c
typedef struct {
    int a;
    char b;
    float c;
} Simple;

Simple x；
Simple y[20], *z;
```
2. **结构成员的访问**
* 直接访问

> `.`操作符的左操作是结构变量的名字，右操作数是访问成员的名字\
> `comp.sa[4].csa`为comp结构的一个结构数组，在选择第四个结构数组元素的成员c

* 间接访问

> `->`操作符的左操作是一个指向结构体的指针，右操作是一个结构\
> `cp->a`相当于`(*cp).a

3. **结构的自引用**

> 结构内部包含一个该类型本身的成员是非法的(无限嵌套)，但是可声明指向该结构的指针，更加高级的ADT中，链表和树，每个结构都会指向下链表一个元素或数的下一个分支。

注意这种形式是错的

```c
typedef struct {
    int a;
    SELF_REF *b;
} SELF_REF;
```
类型名需要在使用之前就定义，这里可以先定义一个结构标签来定义b

```c
typedef struct SELF_REF_TAG {
    int a;
    struct SELF_REF_TAG *b;
} SELF_REF;
```

4. **结构的初始化**
> 类似于多维数组的初始化

```c
struct INIT_EX {
    int a;
    short b[10];
    Simple c;
} x = {
        10,
        { 1, 2, 3, 4, 5 }
        { 25, 'x', 1.9 }
};
```

## 联合

> 联合的声明方式和结构类似，但是一个联合所有成员存储在同一个位置。通过访问不同类型的联合成员，内存中位的组合可以被解释成不同的东西。


```c
struct VARIABLE {
    enum { INT, FLOAT, STRING } type;
    union {
            int i;
            float f;
            char *s;
    } value;
};
```
