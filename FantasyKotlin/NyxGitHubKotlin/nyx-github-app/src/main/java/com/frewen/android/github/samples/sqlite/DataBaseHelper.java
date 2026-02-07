package com.frewen.android.github.samples.sqlite;

import android.content.Context;
import android.database.DatabaseErrorHandler;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;

import androidx.annotation.Nullable;

import static com.frewen.android.github.samples.sqlite.BaseColumns.ID;
import static com.frewen.android.github.samples.sqlite.UserColumns.TABLE_NAME;
import static com.frewen.android.github.samples.sqlite.UserColumns.USER_AGE;
import static com.frewen.android.github.samples.sqlite.UserColumns.USER_CONTENT;
import static com.frewen.android.github.samples.sqlite.UserColumns.USER_CREATE_DATE;
import static com.frewen.android.github.samples.sqlite.UserColumns.USER_NAME;
import static com.frewen.android.github.samples.sqlite.UserColumns.USER_TELEPHONE;

/**
 * @filename: DataBaseHelper
 * @introduction:
 * @author: Frewen.Wong
 * @time: 2020-03-02 09:30
 *         Copyright ©2020 Frewen.Wong. All Rights Reserved.
 */
public class DataBaseHelper extends SQLiteOpenHelper {
    /**
     * integer: 表示整型;
     * real: 表示浮点型;
     * text: 表示文本型;
     * blob:表示二进制类型
     * primary key: 表示主键;
     * autoincrement: 表示自增长
     */
    public static final String SQL_CREATE_TABLE = "CREATE TABLE " + TABLE_NAME + " ("
            + ID + " INTEGER PRIMARY KEY,"
            + USER_NAME + " VARCHAR(50),"
            + USER_AGE + " INTEGER,"
            + USER_TELEPHONE + " VARCHAR(11),"
            + USER_CONTENT + " TEXT,"
            + USER_CREATE_DATE + " INTEGER"
            + ");";

    /**
     * DataBaseHelper 数据库访问助手的构造函数
     * @param context 上下文对象
     * @param name    数据库名称
     * @param factory 数据库工厂
     * @param version 数据库版本号
     */
    public DataBaseHelper(@Nullable Context context, @Nullable String name, @Nullable SQLiteDatabase.CursorFactory factory,
            int version) {
        this(context, name, factory, version, null);
    }

    public DataBaseHelper(@Nullable Context context, @Nullable String name, @Nullable SQLiteDatabase.CursorFactory factory,
            int version, @Nullable DatabaseErrorHandler errorHandler) {
        super(context, name, factory, version, errorHandler);
    }

    @Override
    public void onCreate(SQLiteDatabase db) {
        db.execSQL(SQL_CREATE_TABLE);

    }

    @Override
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
        onCreate(db);
    }
}
