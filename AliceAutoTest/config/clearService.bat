@echo off
cd %~dp0
set exeList=Monitor Phoenix AI PxAnswerService PxVideoProcessor LocalFile Sys RecordScreen Camera Player VideoCapture

for %%I in (%exeList%) do (
 echo kill %%I.exe
 TASKKILL /IM %%I.exe /F /T
)