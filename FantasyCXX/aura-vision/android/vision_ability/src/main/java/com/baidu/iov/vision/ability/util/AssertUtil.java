package com.baidu.iov.vision.ability.util;

/**
 * assert工具
 *
 * Created by liwendong on 2018/5/9.
 */
public class AssertUtil {

    public static void assertArg(boolean success, String err) {
        if (!success) {
            throw new IllegalArgumentException(err);
        }
    }

    public static void assertNotNull(Object arg, String err) {
        assertArg(arg != null, err);
    }

    public static void assertStringNotEmpty(String arg, String err) {
        assertArg(!IovStringUtils.isEmpty(arg), err);
    }
}
