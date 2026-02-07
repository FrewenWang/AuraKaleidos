package com.frewen.ipc.service.sample;


import android.os.Parcel;
import android.os.Parcelable;

/**
 * Book  AIDL中的实体类 实现了Parcelable接口
 *
 * @author Created By frewen
 * @version 版本号：
 * @date 创建时间：2018/6/12
 * @description
 */
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

    /**
     * 直接从Parcel的序列化里面写入
     *
     * @param in
     */
    protected RemoteBook(Parcel in) {
        this.bookId = in.readInt();
        this.bookName = in.readString();
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

    /**
     * 一般不修改，直接默认返回0
     *
     * @return
     */
    @Override
    public int describeContents() {
        return 0;
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

    @Override
    public String toString() {
        return "RemoteBook{" +
                "bookId=" + bookId +
                ", bookName='" + bookName + '\'' +
                '}';
    }
}
