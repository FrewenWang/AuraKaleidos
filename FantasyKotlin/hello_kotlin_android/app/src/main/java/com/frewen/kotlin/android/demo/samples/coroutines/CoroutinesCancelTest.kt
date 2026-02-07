package com.frewen.kotlin.android.demo.samples.coroutines

import kotlinx.coroutines.*

/**
 * @filename: CoroutinesCancelTest
 * @introduction:
 * @author: Frewen.Wong
 * @time: 2019-08-25 17:01
 * Copyright ©2019 Frewen.Wong. All Rights Reserved.
 */
class CoroutinesCancelTest {
    /**
     * 启动一个协程
     */
    fun cancelTest() = runBlocking {
        val job = launch {
            repeat(1000) { i ->
                println("job: I'm sleeping $i ...")
                delay(500L)
            }
        }
        delay(1300L) // 延迟一段时间
        println("main: I'm tired of waiting!")
        //job.cancel() // 取消该作业
        //job.join() // 等待作业执行结束
        //也有一个可以使 Job 挂起的函数 cancelAndJoin 它合并了对 cancel 以及 join 的调用。
        job.cancelAndJoin()
        println("main: Now I can quit.")
    }

    /**
     * 协程的取消是 协作 的。一段协程代码必须协作才能被取消。
     * 所有 kotlinx.coroutines 中的挂起函数都是 可被取消的 。
     * 它们检查协程的取消， 并在取消时抛出 CancellationException。
     * 然而，如果协程正在执行计算任务，并且没有检查取消的话，那么它是不能被取消的，
     * 就如如下示例代码所示：
     */
    fun cannotCancelTest() = runBlocking {
        val startTime = System.currentTimeMillis()
        val job = launch(Dispatchers.Default) {
            var nextPrintTime = startTime
            var i = 0
            while (i < 5) { // 一个执行计算的循环，只是为了占用 CPU
                // 每秒打印消息两次
                if (System.currentTimeMillis() >= nextPrintTime) {
                    println("job: I'm sleeping ${i++} ...")
                    nextPrintTime += 500L
                }
            }
        }
        delay(1300L) // 延迟一段时间
        println("main: I'm tired of waiting!")
        //job.cancel() // 取消该作业
        //job.join() // 等待作业执行结束
        //也有一个可以使 Job 挂起的函数 cancelAndJoin 它合并了对 cancel 以及 join 的调用。
        job.cancelAndJoin()
        println("main: Now I can quit.")
    }


    fun canCancelTest() = runBlocking {
        val startTime = System.currentTimeMillis()
        val job = launch(Dispatchers.Default) {
            var nextPrintTime = startTime
            var i = 0
            while (isActive) { // 一个执行计算的循环，只是为了占用 CPU
                // 每秒打印消息两次
                if (System.currentTimeMillis() >= nextPrintTime) {
                    println("job: I'm sleeping ${i++} ...")
                    nextPrintTime += 500L
                }
            }
        }
        delay(1300L) // 延迟一段时间
        println("main: I'm tired of waiting!")
        //job.cancel() // 取消该作业
        //job.join() // 等待作业执行结束
        //也有一个可以使 Job 挂起的函数 cancelAndJoin 它合并了对 cancel 以及 join 的调用。
        job.cancelAndJoin()
        println("main: Now I can quit.")
    }

    fun cancelFinallyTest() = runBlocking {
        val startTime = System.currentTimeMillis()
        val job = launch {
            try {
                repeat(1000) { i ->
                    println("job: I'm sleeping $i ...")
                    delay(500L)
                }
            } finally {
                println("job: I'm running finally")
            }
        }
        delay(1300L) // 延迟一段时间
        println("main: I'm tired of waiting!")
        job.cancelAndJoin() // 取消该作业并且等待它结束
        println("main: Now I can quit.")
    }
}

fun main() {

    var coroutinesTest = CoroutinesCancelTest();
    println("-------CoroutinesCancelTest--cancelTest--Begin--------")
    coroutinesTest.cancelTest()
    println("-------CoroutinesCancelTest--cancelTest--End--------")


    println("-------CoroutinesCancelTest--cannotCancelTest--Begin--------")
    coroutinesTest.cannotCancelTest()
    println("-------CoroutinesCancelTest--cannotCancelTest--End--------")

    println("-------CoroutinesCancelTest--canCancelTest--Begin--------")
    coroutinesTest.canCancelTest()
    println("-------CoroutinesCancelTest--canCancelTest--End--------")

    println("-------CoroutinesCancelTest--cancelFinallyTest--Begin--------")
    coroutinesTest.cancelFinallyTest()
    println("-------CoroutinesCancelTest--cancelFinallyTest--End--------")
}