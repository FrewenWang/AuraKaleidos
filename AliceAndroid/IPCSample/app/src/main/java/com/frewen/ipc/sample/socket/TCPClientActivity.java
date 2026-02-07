package com.frewen.ipc.sample.socket;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import com.frewen.ipc.sample.R;

public class TCPClientActivity extends AppCompatActivity implements View.OnClickListener {

    private Button mSendButton;
    private TextView mMessageTextView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_tcp_client);

        mMessageTextView = findViewById(R.id.msg_container);
        mSendButton = findViewById(R.id.send);
        mSendButton.setOnClickListener(this);
    }

    @Override
    public void onClick(View view) {

    }
}
