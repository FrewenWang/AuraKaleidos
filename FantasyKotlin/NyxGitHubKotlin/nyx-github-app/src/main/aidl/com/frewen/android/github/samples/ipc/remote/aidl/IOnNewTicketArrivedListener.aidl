// IOnNewBookArrivedListener.aidl
package com.frewen.android.github.samples.ipc.remote.aidl;

// Declare any non-default types here with import statements
import com.frewen.android.github.samples.ipc.remote.aidl.RemoteTicket;

interface IOnNewTicketArrivedListener {
    void onNewTicketArrived(in RemoteTicket newTicket);
}
