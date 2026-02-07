package com.frewen.android.github.samples.network;

/**
 * @filename: ControlCommandReqBean
 * @introduction: 远程控制参数
 * @author: Frewen.Wong
 * @time: 2019/4/13 19:58
 * Copyright ©2018 Frewen.Wong. All Rights Reserved.
 */
public class ControlCommandReqBean extends BaseReqBean {

    private String command;
    private String filename;
    private String receiveName;

    public ControlCommandReqBean(String command, String filename, String receiveName) {
        this.command = command;
        this.filename = filename;
        this.receiveName = receiveName;
    }

    public String getCommand() {
        return command;
    }

    public void setCommand(String command) {
        this.command = command;
    }

    public String getFilename() {
        return filename;
    }

    public void setFilename(String filename) {
        this.filename = filename;
    }

    public String getReceiveName() {
        return receiveName;
    }

    public void setReceiveName(String receiveName) {
        this.receiveName = receiveName;
    }
}
