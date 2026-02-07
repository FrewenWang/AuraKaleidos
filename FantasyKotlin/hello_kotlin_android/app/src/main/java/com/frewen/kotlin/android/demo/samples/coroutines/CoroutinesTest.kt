package com.frewen.kotlin.android.demo.samples.coroutines

import kotlinx.coroutines.*

/**
 * @filename: CoroutinesTest
 * @introduction:
 * @author: Frewen.Wong
 * @time: 2019-08-25 10:36
 * Copyright ©2019 Frewen.Wong. All Rights Reserved.
 */
class CoroutinesTest {
    fun test() {
        /**
         * 在 GlobalScope 中启动了一个新的协程，这意味着新协程的生命周期只受整个应用程序的生命周期限制。
         */
        GlobalScope.launch {
            // 在后台启动一个新的协程并继续
            delay(1000L) // 非阻塞的等待 1 秒钟（默认时间单位是毫秒）
            println("World!") // 在延迟后打印输出
        }
        println("Hello,") // 协程已在等待时主线程还在继续
        Thread.sleep(2000L) // 阻塞主线程 2 秒钟来保证 JVM 存活
    }

    fun runningBlockTest() {
        GlobalScope.launch {
            // 在后台启动一个新的协程并继续
            delay(1000L)
            println("World!")
        }
        println("Hello,") // 主线程中的代码会立即执行
        runBlocking {
            // 但是这个表达式阻塞了主线程
            delay(2000L)  // ……我们延迟 2 秒来保证 JVM 的存活
        }
    }

    /**
     * 这里的 runBlocking<Unit> { …… } 作为用来启动顶层主协程的适配器。
     * 我们显式指定了其返回类型 Unit，因为在 Kotlin 中 main 函数必须返回 Unit 类型。

     */
    fun topRunningBlockTest() = runBlocking {
        GlobalScope.launch {
            // 在后台启动一个新的协程并继续
            delay(1000L)
            println("World!")
        }
        println("Hello,") // 主线程中的代码会立即执行
        delay(2000L)  // ……我们延迟 2 秒来保证 JVM 的存活
    }

    /**
     * 这里的 runBlocking<Unit> { …… } 作为用来启动顶层主协程的适配器。
     * 我们显式指定了其返回类型 Unit，因为在 Kotlin 中 main 函数必须返回 Unit 类型。

     */
    fun topRunningBlockJobTest() = runBlocking {
        val job = GlobalScope.launch {
            // 启动一个新协程并保持对这个作业的引用
            delay(1000L)
            println("World!")
        }
        println("Hello,")
        job.join() // 等待直到子协程执行结束
        println("End,")
    }

    fun coroutineScopeTest() = runBlocking {
        // this: CoroutineScope
        launch {
            delay(200L)
            println("Task from runBlocking")
        }
        coroutineScope {
            // 创建一个协程作用域
            launch {
                delay(500L)
                println("Task from nested launch")
            }

            delay(100L)
            println("Task from coroutine scope") // 这一行会在内嵌 launch 之前输出
        }

        println("Coroutine scope is over") // 这一行在内嵌 launch 执行完毕后才输出
    }
}


//kotlin的入口函数
//fun main(args: Array<String>) {   // 从Kotlin1.3以后，入口的main函数的参数已经不是必须的
fun main() {
    println("-------测试协程相关的代码----Begin--------")
    var coroutinesTest = CoroutinesTest()
    println("-------test----Begin--------")
    coroutinesTest.test()
    println("-------runningBlockTest----Begin--------")
    coroutinesTest.runningBlockTest()


    println("-------topRunningBlockJobTest----Begin--------")
    coroutinesTest.topRunningBlockJobTest()


    println("-------coroutineScopeTest----Begin--------")
    coroutinesTest.coroutineScopeTest()
}