package com.frewen.android.github.samples.ipc.remote.aidl;

import android.os.Parcel;
import android.os.Parcelable;

import java.util.HashMap;
import java.util.Map;

/**
 * @filename: RemoteTicket
 * @introduction:
 * @author: Frewen.Wong
 * @time: 2019/9/4 0004 下午12:59
 * Copyright ©2019 Frewen.Wong. All Rights Reserved.
 */
public class RemoteTicket implements Parcelable {
    private int ticketId;
    private String ticketName;
    private double ticketPrice;
    private Map<String, String> ticketPlaces;

    /**
     * 含有参数构造函数
     *
     * @param ticketId
     * @param ticketName
     * @param ticketPrice
     * @param ticketPlaces
     */
    public RemoteTicket(int ticketId, String ticketName, double ticketPrice, Map<String, String> ticketPlaces) {
        this.ticketId = ticketId;
        this.ticketName = ticketName;
        this.ticketPrice = ticketPrice;
        this.ticketPlaces = ticketPlaces;
    }

    /**
     * 直接从Parcel的序列化里面写入
     *
     * @param in
     */
    protected RemoteTicket(Parcel in) {
        ticketId = in.readInt();
        ticketName = in.readString();
        ticketPrice = in.readDouble();
        ticketPlaces = in.readHashMap(HashMap.class.getClassLoader());
    }

    public static final Creator<RemoteTicket> CREATOR = new Creator<RemoteTicket>() {
        @Override
        public RemoteTicket createFromParcel(Parcel in) {
            return new RemoteTicket(in);
        }

        @Override
        public RemoteTicket[] newArray(int size) {
            return new RemoteTicket[size];
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
        dest.writeInt(ticketId);
        dest.writeString(ticketName);
        dest.writeDouble(ticketPrice);
        dest.writeMap(ticketPlaces);
    }

    @Override
    public String toString() {
        return "RemoteTicket{" +
                "ticketId=" + ticketId +
                ", ticketName='" + ticketName + '\'' +
                ", ticketPrice=" + ticketPrice +
                ", ticketPlaces=" + ticketPlaces +
                '}';
    }
}
