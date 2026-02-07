package com.frewen.android.github.samples.ipc.server;

import android.content.Context;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;

import androidx.annotation.Nullable;

/**
 * DBOpenHelpler
 */
public class DBOpenHelpler extends SQLiteOpenHelper {
    private static final String DB_NAME = "student_provider.db";
    /**
     * 学生表
     */
    private static final String STUDENT_TABLE_NAME = "student";
    /**
     * 教师表
     */
    private static final String CLASS_TABLE_NAME = "class";
    /**
     * 数据库表的版本号
     */
    private static final int DB_VERSION = 1;

    /**
     * 学生管理信息表
     */
    private static final String CREATE_STUDENT_TABLE = "CREATE TABLE IF NOT EXISTS "
            + STUDENT_TABLE_NAME + "(_id INTEGER PRIMARY KEY," + "name TEXT,sex INT)";
    private static final String CREATE_CLASS_TABLE = "CREATE TABLE IF NOT EXISTS "
            + CLASS_TABLE_NAME + "(_id INTEGER PRIMARY KEY," + "name TEXT)";

    /**
     * 构造函数
     * @param context
     */
    public DBOpenHelpler(@Nullable Context context) {
        super(context, DB_NAME, null, DB_VERSION);
    }

    @Override
    public void onCreate(SQLiteDatabase db) {
        db.execSQL(CREATE_STUDENT_TABLE);
        db.execSQL(CREATE_CLASS_TABLE);
    }

    @Override
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {

    }
}
