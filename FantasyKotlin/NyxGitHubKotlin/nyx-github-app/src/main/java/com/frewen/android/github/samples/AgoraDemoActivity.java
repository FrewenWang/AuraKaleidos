package com.frewen.android.github.samples;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.PorterDuff;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.Toast;

import com.frewen.android.github.R;

import io.agora.rtc.IRtcEngineEventHandler;
import io.agora.rtc.RtcEngine;
import io.agora.rtc.video.VideoCanvas;
import io.agora.rtc.video.VideoEncoderConfiguration;

/**
 * 声网SDKDemo学习
 */
public class AgoraDemoActivity extends AppCompatActivity {
    private static final String TAG = "T:AgoraDemoActivity";

    private static final int PERMISSION_REQ_ID = 22;

    // permission WRITE_EXTERNAL_STORAGE is not mandatory for Agora RTC SDK,
    // just incase if you wanna save logs to external sdcard
    private static final String[] REQUESTED_PERMISSIONS = {Manifest.permission.RECORD_AUDIO,
            Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE};

    /**
     * 接口类包含应用程序调用的主要方法。
     */
    private RtcEngine mRtcEngine;
    /**
     * RtcEngine连接事件回调的Handler
     */
    private IRtcEngineEventHandler mRtcEventHandler = new IRtcEngineEventHandler() {
        @Override
        public void onFirstRemoteVideoDecoded(int uid, int width, int height, int elapsed) {
            super.onFirstRemoteVideoDecoded(uid, width, height, elapsed);
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    setupRemoteVideo(uid);
                }
            });
        }

        @Override
        public void onUserEnableVideo(int uid, boolean enabled) {
            super.onUserEnableVideo(uid, enabled);
        }

        @Override
        public void onUserJoined(int uid, int elapsed) {
            super.onUserJoined(uid, elapsed);
        }

        @Override
        public void onUserMuteVideo(int uid, boolean muted) {
            super.onUserMuteVideo(uid, muted);
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    onRemoteUserVideoMuted(uid, muted);
                }
            });
        }

        @Override
        public void onUserMuteAudio(int uid, boolean muted) {
            super.onUserMuteAudio(uid, muted);
            Log.d(TAG, "FMsg:onUserMuteAudio() called with: uid = [" + uid + "], muted = ["
                    + muted + "]");
        }

        @Override
        public void onUserOffline(int uid, int reason) {
            super.onUserOffline(uid, reason);
            Log.d(TAG, "FMsg:onUserOffline() called with: uid = [" + uid + "], reason = ["
                    + reason + "]");
            Log.d(TAG, "FMsg:onUserOffline() called with: uid = [" + uid + "], reason = ["
                    + reason + "]");
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    onRemoteUserLeft();
                }
            });
        }
    };

    /**
     * 设置远程的视频视图View
     *
     * @param uid
     */
    private void setupRemoteVideo(int uid) {
        FrameLayout container = findViewById(R.id.remote_video_view_container);
        if (container.getChildCount() >= 1) {
            return;
        }
        SurfaceView surfaceView = RtcEngine.CreateRendererView(getBaseContext());
        container.addView(surfaceView);
        mRtcEngine.setupRemoteVideo(new VideoCanvas(surfaceView, VideoCanvas.RENDER_MODE_FIT, uid));

        surfaceView.setTag(uid); // for mark purpose
        View tipMsg = findViewById(R.id.quick_tips_when_use_agora_sdk); // optional UI
        tipMsg.setVisibility(View.GONE);
    }

    /**
     * onRemoteUserVideoMuted
     *
     * @param uid
     * @param muted
     */
    private void onRemoteUserVideoMuted(int uid, boolean muted) {
        FrameLayout container = findViewById(R.id.remote_video_view_container);
        SurfaceView surfaceView = (SurfaceView) container.getChildAt(0);
        Object tag = surfaceView.getTag();
        if (tag != null && (Integer) tag == uid) {
            if (muted) {
                surfaceView.setVisibility(View.GONE);
            } else {
                surfaceView.setVisibility(View.VISIBLE);
            }
        }
    }

    /**
     * onRemoteUserLeft
     */
    private void onRemoteUserLeft() {
        FrameLayout container = findViewById(R.id.remote_video_view_container);
        container.removeAllViews();

        View tipMsg = findViewById(R.id.quick_tips_when_use_agora_sdk); // optional UI
        tipMsg.setVisibility(View.VISIBLE);
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_agora_demo);

        if (checkSelfPermission(REQUESTED_PERMISSIONS[0], PERMISSION_REQ_ID)
                && checkSelfPermission(REQUESTED_PERMISSIONS[1], PERMISSION_REQ_ID)
                && checkSelfPermission(REQUESTED_PERMISSIONS[2], PERMISSION_REQ_ID)) {
            initAgoraEngineAndJoinChannel();
        }

    }

    /**
     * 初始化声网的引擎。并且加以Channel
     */
    private void initAgoraEngineAndJoinChannel() {
        Log.d(TAG, "FMsg:initAgoraEngineAndJoinChannel() called");
        initializeAgoraEngine();

        setupVideoProfile();
        setupLocalVideo();

        joinChannel();
    }

    /**
     * joinChannel
     */
    private void joinChannel() {
        mRtcEngine.joinChannel(null, "test", "Extra Optional Data", 0);
        // if you do not specify the uid, we will generate the uid for you
    }

    /**
     * setupVideoProfile
     */
    private void setupVideoProfile() {
        // 启用视频模块
        mRtcEngine.enableVideo();

        //mRtcEngine.setVideoProfile(Constants.VIDEO_PROFILE_360P, false); // Earlier than 2.3.0
        // 设置视频编码配置。
        mRtcEngine.setVideoEncoderConfiguration(new VideoEncoderConfiguration(
                VideoEncoderConfiguration.VD_640x360,
                VideoEncoderConfiguration.FRAME_RATE.FRAME_RATE_FPS_15,
                VideoEncoderConfiguration.STANDARD_BITRATE,
                VideoEncoderConfiguration.ORIENTATION_MODE.ORIENTATION_MODE_FIXED_PORTRAIT));

        // 启用音频模块
        mRtcEngine.enableAudio();
    }

    /**
     * 设置本地视频
     */
    private void setupLocalVideo() {
        FrameLayout container = findViewById(R.id.local_video_view_container);
        SurfaceView surfaceView = RtcEngine.CreateRendererView(getBaseContext());
        surfaceView.setZOrderMediaOverlay(true);
        container.addView(surfaceView);
        mRtcEngine.setupLocalVideo(new VideoCanvas(surfaceView, VideoCanvas.RENDER_MODE_FIT, 0));
    }


    /**
     * 初始化声网引擎
     */
    private void initializeAgoraEngine() {
        try {
            mRtcEngine = RtcEngine.create(getBaseContext(), getString(R.string.agora_app_id),
                    mRtcEventHandler);
        } catch (Exception e) {
            Log.e(TAG, Log.getStackTraceString(e));

            throw new RuntimeException("NEED TO check rtc sdk init fatal error\n"
                    + Log.getStackTraceString(e));
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        leaveChannel();
        RtcEngine.destroy();
        mRtcEngine = null;
    }

    /**
     * leaveChannel
     */
    private void leaveChannel() {
        mRtcEngine.leaveChannel();
    }

    /**
     * Activity进行权限检查
     *
     * @param permission
     * @param requestCode
     * @return
     */
    public boolean checkSelfPermission(String permission, int requestCode) {
        Log.d(TAG, "FMsg:checkSelfPermission() called with: permission = [" + permission + "], "
                + "requestCode = [" + requestCode + "]");
        if (ContextCompat.checkSelfPermission(this, permission)
                != PackageManager.PERMISSION_GRANTED) {

            ActivityCompat.requestPermissions(this,
                    REQUESTED_PERMISSIONS,
                    requestCode);
            return false;
        }
        return true;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        Log.i(TAG, "onRequestPermissionsResult " + grantResults[0] + " " + requestCode);

        switch (requestCode) {
            case PERMISSION_REQ_ID:
                if (grantResults[0] != PackageManager.PERMISSION_GRANTED || grantResults[1]
                        != PackageManager.PERMISSION_GRANTED
                        || grantResults[2] != PackageManager.PERMISSION_GRANTED) {
                    showLongToast("Need permissions " + Manifest.permission.RECORD_AUDIO + "/"
                            + Manifest.permission.CAMERA + "/"
                            + Manifest.permission.WRITE_EXTERNAL_STORAGE);
                    finish();
                    break;
                }
                initAgoraEngineAndJoinChannel();
                break;
            default:
                break;
        }
    }

    /**
     * showLongToast
     *
     * @param msg
     */
    public final void showLongToast(final String msg) {
        this.runOnUiThread(new Runnable() {
            @Override
            public void run() {
                Toast.makeText(getApplicationContext(), msg, Toast.LENGTH_LONG).show();
            }
        });
    }


    /**
     * 本地禁麦
     *
     * @param view
     */
    public void onLocalVideoMuteClicked(View view) {
        ImageView iv = (ImageView) view;
        if (iv.isSelected()) {
            iv.setSelected(false);
            iv.clearColorFilter();
        } else {
            iv.setSelected(true);
            iv.setColorFilter(getResources().getColor(R.color.colorPrimary), PorterDuff.Mode.MULTIPLY);
        }
        // 开关本地音频发送。
        mRtcEngine.muteLocalVideoStream(iv.isSelected());

        FrameLayout container = (FrameLayout) findViewById(R.id.local_video_view_container);
        SurfaceView surfaceView = (SurfaceView) container.getChildAt(0);
        surfaceView.setZOrderMediaOverlay(!iv.isSelected());
        if (iv.isSelected()) {
            surfaceView.setVisibility(View.GONE);
        } else {
            surfaceView.setVisibility(View.VISIBLE);
        }
    }

    /**
     * onLocalAudioMuteClicked
     *
     * @param view
     */
    public void onLocalAudioMuteClicked(View view) {
        ImageView iv = (ImageView) view;
        if (iv.isSelected()) {
            iv.setSelected(false);
            iv.clearColorFilter();
        } else {
            iv.setSelected(true);
            iv.setColorFilter(getResources().getColor(R.color.colorPrimary), PorterDuff.Mode.MULTIPLY);
        }

        mRtcEngine.muteLocalAudioStream(iv.isSelected());
    }

    /**
     * onSwitchCameraClicked
     *
     * @param view
     */
    public void onSwitchCameraClicked(View view) {
        mRtcEngine.switchCamera();
    }

    /**
     * onEncCallClicked
     *
     * @param view
     */
    public void onEncCallClicked(View view) {
        finish();
    }
}
