package com.frewen.github.library.network.core;

/**
 * @filename: IProgramEnv
 * @introduction:
 * @author: Frewen.Wong
 * @time: 2020/4/11 20:17
 *         Copyright ©2020 Frewen.Wong. All Rights Reserved.
 */
public interface IProgramEnv {

    String getProduction();

    String getPreProduction();

    String getTest();

    String getDev();
}
