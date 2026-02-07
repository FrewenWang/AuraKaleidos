// IRemoteBookManager.aidl
package com.frewen.ipc.service.sample;

// Declare any non-default types here with import statements
import com.frewen.ipc.service.sample.RemoteBook;
import com.frewen.ipc.service.sample.IOnNewBookArrivedListener;


interface IRemoteBookManager {

    List<RemoteBook> getBookList();

    void addBook(in RemoteBook book);

    void registerListener(IOnNewBookArrivedListener listener);

    void unregisterListener(IOnNewBookArrivedListener listener);

}
