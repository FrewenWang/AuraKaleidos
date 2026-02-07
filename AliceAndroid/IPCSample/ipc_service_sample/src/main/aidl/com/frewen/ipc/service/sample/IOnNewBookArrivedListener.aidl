// IOnNewBookArrivedListener.aidl
package com.frewen.ipc.service.sample;

// Declare any non-default types here with import statements
import com.frewen.ipc.service.sample.RemoteBook;
import com.frewen.ipc.service.sample.IOnNewBookArrivedSuccessListener;

interface IOnNewBookArrivedListener {

    void onNewBookArrived(in RemoteBook newBook, IOnNewBookArrivedSuccessListener listener);

}
