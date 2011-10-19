REM		Remove old results (prior to the last run)
if exist .\RegressionResultsBackup rmdir /q /s .\RegressionResultsBackup

REM		Copy last runs results to backup location.
if exist .\RegressionResults move .\RegressionResults .\RegressionResultsBackup

mkdir .\RegressionResults

REM     RUN All collected panels

REM		ALTA DATA
set APP=.\src\Applications\CyberStitchFidTester\bin\x64\Release\CyberStitchFidTester.exe
set SIMDATA=\\msp\dfs\archive\colorSimBoards\RegressionTest2\
set CADDIR=\\msp\dfs\archive\colorSimBoards\PPM\
%APP% -s "%SIMDATA%SIMScenario.xml" -p "%CADDIR%PPMTestPanel-HighRes2Fids(diagonal).xml" -f "%CADDIR%PPMTestPanel-HighResAllFidsNoPads.xml" -o .\RegressionResults\2Fids.txt -n 23

REM		Test that the results are the same as last time
REM TODO .\src\Applications\DBCompare\bin\Release\DBCompare.exe -1 c:\SentryData\Sentry.yap -2 c:\SentryDataBackup\Sentry.yap -o .\DBCompareResults.txt



