// IRemoteServiceInterface.aidl
package com.frewen.android.github.samples.ipc.remote.aidl;
// Declare any non-default types here with import statements
import com.frewen.android.github.samples.ipc.remote.aidl.RemoteTicket;
import com.frewen.android.github.samples.ipc.remote.aidl.IOnNewTicketArrivedListener;

interface IRemoteServiceInterface {
   /**
    * 获取远程服务的pid
    **/
    int getPid();

    List<RemoteTicket> getTicketList();

    void addTicket(in RemoteTicket ticket);

    void registerListener(IOnNewTicketArrivedListener listener);

    void unregisterListener(IOnNewTicketArrivedListener listener);
    /**
     * Demonstrates some basic types that you can use as parameters
     * and return values in AIDL.
     */
    void basicTypes(int anInt, long aLong, boolean aBoolean, float aFloat,
            double aDouble, String aString);
}
