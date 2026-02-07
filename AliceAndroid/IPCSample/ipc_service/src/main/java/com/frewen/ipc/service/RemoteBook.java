package com.frewen.ipc.service;

import android.os.Parcel;
import android.os.Parcelable;

public class RemoteBook implements Parcelable {
    public int bookId;
    public String bookName;

    /**
     * 含有参数的构造函数
     *
     * @param bookId
     * @param bookName
     */
    public RemoteBook(int bookId, String bookName) {
        this.bookId = bookId;
        this.bookName = bookName;
    }

    protected RemoteBook(Parcel in) {
        this.bookId = in.readInt();
        this.bookName = in.readString();
    }

    /**
     * 序列化重写机制
     *
     * @param dest
     * @param flags
     */
    @Override
    public void writeToParcel(Parcel dest, int flags) {
        dest.writeInt(bookId);
        dest.writeString(bookName);
    }

    /**
     * 一般不修改，直接默认返回0
     *
     * @return
     */
    @Override
    public int describeContents() {
        return 0;
    }

    public static final Creator<RemoteBook> CREATOR = new Creator<RemoteBook>() {
        @Override
        public RemoteBook createFromParcel(Parcel in) {
            return new RemoteBook(in);
        }

        @Override
        public RemoteBook[] newArray(int size) {
            return new RemoteBook[size];
        }
    };

    @Override
    public String toString() {
        return "RemoteBook{" +
                "bookId=" + bookId +
                ", bookName='" + bookName + '\'' +
                '}';
    }
}
