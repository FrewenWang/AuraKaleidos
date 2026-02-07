package com.frewen.android.github.samples.bluetooth;

import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothGatt;
import android.bluetooth.BluetoothGattCallback;
import android.bluetooth.BluetoothGattCharacteristic;
import android.bluetooth.BluetoothGattService;
import android.bluetooth.BluetoothManager;
import android.content.Intent;
import android.os.Build;
import android.os.Handler;
import android.os.Bundle;
import android.text.TextUtils;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ListView;
import android.widget.ProgressBar;
import android.widget.TextView;

import com.frewen.android.github.R;
import com.frewen.android.github.utils.HexUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

import androidx.appcompat.app.AppCompatActivity;

import static android.bluetooth.BluetoothDevice.TRANSPORT_LE;

/**
 * @author Frewen.Wong
 */
public class BlueToothDemoActivity extends AppCompatActivity {

    private static final String TAG = "ble_tag";
    private static final String HEX = "0123456789abcdef";
    ProgressBar pbSearchBle;
    ImageView ivSerBleStatus;
    TextView tvSerBleStatus;
    TextView tvSerBindStatus;
    ListView bleListView;
    private LinearLayout operaView;
    private Button btnWrite;
    private Button btnRead;
    private EditText etWriteContent;
    private TextView tvResponse;
    private List<BluetoothDevice> mDatas;
    private List<Integer> mRssis;
    private BlueAdapter mAdapter;
    private BluetoothAdapter mBluetoothAdapter;
    private BluetoothManager mBluetoothManager;
    private boolean isScaning = false;
    private boolean isConnecting = false;
    private BluetoothGatt mBluetoothGatt;

    //服务和特征值
    private UUID writeUUIDService;
    private UUID writeUUIDChara;
    private UUID readUUIDService;
    private UUID readUUIDChara;
    private UUID notifyUUIDService;
    private UUID notifyUUIDChara;
    private UUID indicateUUIDService;
    private UUID indicateUUIDChara;
    private String hex = "7B46363941373237323532443741397D";



    private BluetoothGattCallback gattCallback = new BluetoothGattCallback() {
        @Override
        public void onConnectionStateChange(BluetoothGatt gatt, int status, int newState) {
            super.onConnectionStateChange(gatt, status, newState);
            Log.e(TAG, "onConnectionStateChange()");
            if (status == BluetoothGatt.GATT_SUCCESS) {
                //连接成功
                if (newState == BluetoothGatt.STATE_CONNECTED) {
                    Log.e(TAG, "连接成功");
                    //发现服务
                    gatt.discoverServices();
                }
            } else {
                //连接失败
                Log.e(TAG, "失败==" + status);
                mBluetoothGatt.close();
                isConnecting = false;
            }
        }
        @Override
        public void onServicesDiscovered(BluetoothGatt gatt, int status) {
            super.onServicesDiscovered(gatt, status);
            //直到这里才是真正建立了可通信的连接
            isConnecting = false;
            Log.e(TAG, "onServicesDiscovered()---建立连接");
            //获取初始化服务和特征值
            initServiceAndChara();
            //订阅通知
            mBluetoothGatt.setCharacteristicNotification(mBluetoothGatt
                    .getService(notifyUUIDService).getCharacteristic(notifyUUIDChara), true);
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    bleListView.setVisibility(View.GONE);
                    operaView.setVisibility(View.VISIBLE);
                    tvSerBindStatus.setText("已连接");
                }
            });
        }
        @Override
        public void onCharacteristicRead(BluetoothGatt gatt, BluetoothGattCharacteristic characteristic, int status) {
            super.onCharacteristicRead(gatt, characteristic, status);
        }
        @Override
        public void onCharacteristicWrite(BluetoothGatt gatt, BluetoothGattCharacteristic characteristic, int status) {
            super.onCharacteristicWrite(gatt, characteristic, status);
        }
        @Override
        public void onCharacteristicChanged(BluetoothGatt gatt, BluetoothGattCharacteristic characteristic) {
            super.onCharacteristicChanged(gatt, characteristic);
            Log.e(TAG, "onCharacteristicChanged()" + characteristic.getValue());
            final byte[] data = characteristic.getValue();
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    addText(tvResponse, bytes2hex(data));
                }
            });
        }
    };
    /**
     * 蓝牙搜索的的回调接口
     */
    private BluetoothAdapter.LeScanCallback scanCallback = new BluetoothAdapter.LeScanCallback() {
        @Override
        public void onLeScan(BluetoothDevice device, int rssi, byte[] scanRecord) {
            Log.e(TAG, "run: scanning...");
            if (!mDatas.contains(device)) {
                mDatas.add(device);
                mRssis.add(rssi);
                mAdapter.notifyDataSetChanged();
            }

        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_blue_tooth_demo);
        initView();
        initData();
        mBluetoothManager = (BluetoothManager) getSystemService(BLUETOOTH_SERVICE);
        mBluetoothAdapter = mBluetoothManager.getAdapter();
        if (mBluetoothAdapter == null || !mBluetoothAdapter.isEnabled()) {
            Intent intent = new Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE);
            startActivityForResult(intent, 0);
        }

    }

    private void initData() {
        mDatas = new ArrayList<>();
        mRssis = new ArrayList<>();
        mAdapter = new BlueAdapter(BlueToothDemoActivity.this, mDatas, mRssis);
        bleListView.setAdapter(mAdapter);
        mAdapter.notifyDataSetChanged();
    }

    private void initView() {
        pbSearchBle = findViewById(R.id.progress_ser_bluetooth);
        ivSerBleStatus = findViewById(R.id.iv_ser_ble_status);
        tvSerBindStatus = findViewById(R.id.tv_ser_bind_status);
        tvSerBleStatus = findViewById(R.id.tv_ser_ble_status);
        bleListView = findViewById(R.id.ble_list_view);
        operaView = findViewById(R.id.opera_view);
        btnWrite = findViewById(R.id.btnWrite);
        btnRead = findViewById(R.id.btnRead);
        etWriteContent = findViewById(R.id.et_write);
        tvResponse = findViewById(R.id.tv_response);
        btnRead.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                readData();
            }
        });

        btnWrite.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //执行写入操作
                writeData();
            }
        });


        ivSerBleStatus.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (isScaning) {
                    tvSerBindStatus.setText("停止搜索");
                    stopScanDevice();
                } else {
                    checkPermissions();
                }

            }
        });
        bleListView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                if (isScaning) {
                    stopScanDevice();
                }
                if (!isConnecting) {
                    isConnecting = true;
                    BluetoothDevice bluetoothDevice = mDatas.get(position);
                    //连接设备
                    tvSerBindStatus.setText("连接中");
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                        mBluetoothGatt = bluetoothDevice.connectGatt(BlueToothDemoActivity.this,
                                true, gattCallback, TRANSPORT_LE);
                    } else {
                        mBluetoothGatt = bluetoothDevice.connectGatt(BlueToothDemoActivity.this,
                                true, gattCallback);
                    }
                }

            }
        });


    }

    private void readData() {
        BluetoothGattCharacteristic characteristic = mBluetoothGatt.getService(readUUIDService)
                .getCharacteristic(readUUIDChara);
        mBluetoothGatt.readCharacteristic(characteristic);
    }


    /**
     * 开始扫描 10秒后自动停止
     */
    private void scanDevice() {
        tvSerBindStatus.setText("正在搜索");
        isScaning = true;
        pbSearchBle.setVisibility(View.VISIBLE);
        mBluetoothAdapter.startLeScan(scanCallback);
        new Handler().postDelayed(new Runnable() {
            @Override
            public void run() {
                //结束扫描
                mBluetoothAdapter.stopLeScan(scanCallback);
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        isScaning = false;
                        pbSearchBle.setVisibility(View.GONE);
                        tvSerBindStatus.setText("搜索已结束");
                    }
                });
            }
        }, 10000);
    }

    /**
     * 停止扫描
     */
    private void stopScanDevice() {
        isScaning = false;
        pbSearchBle.setVisibility(View.GONE);
        mBluetoothAdapter.stopLeScan(scanCallback);
    }
    /**
     * 检查权限
     */
    private void checkPermissions() {

    }


    private void initServiceAndChara() {
        List<BluetoothGattService> bluetoothGattServices = mBluetoothGatt.getServices();
        for (BluetoothGattService bluetoothGattService : bluetoothGattServices) {
            List<BluetoothGattCharacteristic> characteristics = bluetoothGattService.getCharacteristics();
            for (BluetoothGattCharacteristic characteristic : characteristics) {
                int charaProp = characteristic.getProperties();
                if ((charaProp & BluetoothGattCharacteristic.PROPERTY_READ) > 0) {
                    readUUIDChara = characteristic.getUuid();
                    readUUIDService = bluetoothGattService.getUuid();
                    Log.e(TAG, "read_chara=" + readUUIDChara + "----read_service=" + readUUIDService);
                }
                if ((charaProp & BluetoothGattCharacteristic.PROPERTY_WRITE) > 0) {
                    writeUUIDChara = characteristic.getUuid();
                    writeUUIDService = bluetoothGattService.getUuid();
                    Log.e(TAG, "write_chara=" + writeUUIDChara + "----write_service=" + writeUUIDService);
                }
                if ((charaProp & BluetoothGattCharacteristic.PROPERTY_WRITE_NO_RESPONSE) > 0) {
                    writeUUIDChara = characteristic.getUuid();
                    writeUUIDService = bluetoothGattService.getUuid();
                    Log.e(TAG, "write_chara=" + writeUUIDChara + "----write_service=" + writeUUIDService);

                }
                if ((charaProp & BluetoothGattCharacteristic.PROPERTY_NOTIFY) > 0) {
                    notifyUUIDChara = characteristic.getUuid();
                    notifyUUIDService = bluetoothGattService.getUuid();
                    Log.e(TAG, "notify_chara=" + notifyUUIDChara + "----notify_service=" + notifyUUIDService);
                }
                if ((charaProp & BluetoothGattCharacteristic.PROPERTY_INDICATE) > 0) {
                    indicateUUIDChara = characteristic.getUuid();
                    indicateUUIDService = bluetoothGattService.getUuid();
                    Log.e(TAG, "indicate_chara=" + indicateUUIDChara + "----indicate_service=" + indicateUUIDService);

                }
            }
        }
    }

    private void addText(TextView textView, String content) {
        textView.append(content);
        textView.append("\n");
        int offset = textView.getLineCount() * textView.getLineHeight();
        if (offset > textView.getHeight()) {
            textView.scrollTo(0, offset - textView.getHeight());
        }
    }

    private void writeData() {
        BluetoothGattService service = mBluetoothGatt.getService(writeUUIDService);
        BluetoothGattCharacteristic charaWrite = service.getCharacteristic(writeUUIDChara);
        byte[] data;
        String content = etWriteContent.getText().toString();
        if (!TextUtils.isEmpty(content)) {
            data = HexUtils.hexStringToBytes(content);
        } else {
            data = HexUtils.hexStringToBytes(hex);
        }
        //数据大于个字节 分批次写入
        if (data.length > 20) {
            Log.e(TAG, "writeData: length=" + data.length);
            int num = 0;
            if (data.length % 20 != 0) {
                num = data.length / 20 + 1;
            } else {
                num = data.length / 20;
            }
            for (int i = 0; i < num; i++) {
                byte[] tempArr;
                if (i == num - 1) {
                    tempArr = new byte[data.length - i * 20];
                    System.arraycopy(data, i * 20, tempArr, 0, data.length - i * 20);
                } else {
                    tempArr = new byte[20];
                    System.arraycopy(data, i * 20, tempArr, 0, 20);
                }
                charaWrite.setValue(tempArr);
                mBluetoothGatt.writeCharacteristic(charaWrite);
            }
        } else {
            charaWrite.setValue(data);
            mBluetoothGatt.writeCharacteristic(charaWrite);
        }
    }

    /**
     * bytes2hex
     * @param bytes
     * @return
     */
    public static String bytes2hex(byte[] bytes) {
        StringBuilder sb = new StringBuilder(bytes.length * 2);
        for (byte b : bytes) {
            // 取出这个字节的高4位，然后与0x0f与运算，得到一个0-15之间的数据，通过HEX.charAt(0-15)即为16进制数
            sb.append(HEX.charAt((b >> 4) & 0x0f));
            // 取出这个字节的低位，与0x0f与运算，得到一个0-15之间的数据，通过HEX.charAt(0-15)即为16进制数
            sb.append(HEX.charAt(b & 0x0f));
        }
        return sb.toString();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        mBluetoothGatt.disconnect();
    }
}
