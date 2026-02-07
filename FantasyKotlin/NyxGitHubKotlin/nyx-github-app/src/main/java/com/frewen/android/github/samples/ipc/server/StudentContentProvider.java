package com.frewen.android.github.samples.ipc.server;

import android.content.ContentProvider;
import android.content.ContentValues;
import android.database.Cursor;
import android.net.Uri;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

/**
 * 需要继承ContentProvider类并实现六个抽象方法即可：onCreate、query、update、insert、delete和getType。
 * 这六个抽象方法都很好理解
 * <p>
 * 根据Bundle的原理，我们知道这六个方法均运行在ContentProvider进程中。
 * 除了onCreate由系统回调并且运行在主线程中。其他五个方法均由外界回调并且运行在Bundle线程池中
 */
public class StudentContentProvider extends ContentProvider {
    private static final String TAG = "StudentContentProvider";

    @Override
    public boolean onCreate() {
        Log.d(TAG, "onCreate() called on thread:" + Thread.currentThread().getName());
        return false;
    }

    @Nullable
    @Override
    public Cursor query(@NonNull Uri uri, @Nullable String[] projection, @Nullable String selection,
                        @Nullable String[] selectionArgs, @Nullable String sortOrder) {
        Log.d(TAG, "query() called on thread:" + Thread.currentThread().getName());
        return null;
    }

    /**
     * getType用来返回一个Uri请求所对应的MIME类型（媒体类型），
     * 比如图片、视频等，这个媒体类型还是有点复杂的，
     * 如果我们的应用不关注这个选项，可以直接在这个方法中返回null
     *
     * @param uri
     * @return
     */
    @Nullable
    @Override
    public String getType(@NonNull Uri uri) {
        Log.d(TAG, "getType() called on thread:" + Thread.currentThread().getName());
        return null;
    }

    @Nullable
    @Override
    public Uri insert(@NonNull Uri uri, @Nullable ContentValues values) {
        Log.d(TAG, "insert() called on thread:" + Thread.currentThread().getName());
        return null;
    }

    @Override
    public int delete(@NonNull Uri uri, @Nullable String selection, @Nullable String[] selectionArgs) {
        Log.d(TAG, "delete() called on thread:" + Thread.currentThread().getName());
        return 0;
    }

    @Override
    public int update(@NonNull Uri uri, @Nullable ContentValues values, @Nullable String selection,
                      @Nullable String[] selectionArgs) {
        Log.d(TAG, "update() called on thread:" + Thread.currentThread().getName());
        return 0;
    }
}
