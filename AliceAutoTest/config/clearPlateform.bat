@echo off
cd %~dp0
set exeList=TSClient TALLivePlatform VCamDemo

for %%I in (%exeList%) do (
 echo kill %%I.exe
 TASKKILL /IM %%I.exe /F /T
)