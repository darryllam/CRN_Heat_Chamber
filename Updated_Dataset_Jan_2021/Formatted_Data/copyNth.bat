:: copyNth.bat  interval  sourcePath  destinationPath
@echo off
setlocal
set /a n=0
for %%F in ("%~f2.\*") do 2>nul set /a "1/(n=(n+1)%%%1)" || copy "%%F" %3