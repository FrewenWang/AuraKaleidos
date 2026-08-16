***重要：必须运行install.bat进行注册***

1、python环境安装，版本为3.7

2、目前只支持AVI格式，使用ffmpeg进行转换，命令如下：
ffmpeg -i filename.mp4 -vcodec copy -acodec copy filename.avi

3、在xml中配置相关要读取的视频集合

4、在命令行python control2.py运行
