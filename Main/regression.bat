REM - VARIABLES
set OUTPUTDIR=C:\CyberStitchRegressionResults\
set OUTPUTBACKUP=C:\CyberStitchRegressionResultsBackup\
set APP=.\src\Applications\CyberStitchFidTester\bin\x64\Release\CyberStitchFidTester.exe
set SIMDATA=\\msp\dfs\archive\colorSimBoards\RegressionTest2\
set CADDIR=\\msp\dfs\archive\colorSimBoards\PPM\

REM		Remove old results (prior to the last run)
if exist %OUTPUTBACKUP% rmdir /q /s %OUTPUTBACKUP%

REM		Copy last runs results to backup location.
if exist %OUTPUTDIR% move %OUTPUTDIR% %OUTPUTBACKUP%

REM		Create Output Directory
mkdir  %OUTPUTDIR%

REM     RUN All collected panels
%APP% -b -s "%SIMDATA%SIMScenario.xml" -p "%CADDIR%PPMTestPanel-HighRes2Fids(diagonal).xml" -f "%CADDIR%PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%2Fids.txt -n 6
%APP% -b -s "%SIMDATA%SIMScenario.xml" -p "%CADDIR%PPMTestPanel-HighRes3Fids(triangle).xml" -f "%CADDIR%PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%3Fids.txt -n 6
%APP% -b -s "%SIMDATA%SIMScenario.xml" -p "%CADDIR%PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%4Fids.txt -n 6
%APP% -b -w -s "%SIMDATA%SIMScenario.xml" -p "%CADDIR%PPMTestPanel-HighRes2Fids(diagonal).xml" -f "%CADDIR%PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%2FidsProjective.txt -n 6
%APP% -b -w -s "%SIMDATA%SIMScenario.xml" -p "%CADDIR%PPMTestPanel-HighRes3Fids(triangle).xml" -f "%CADDIR%PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%3FidsProjective.txt -n 6
%APP% -b -w -s "%SIMDATA%SIMScenario.xml" -p "%CADDIR%PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%4FidsProjective.txt -n 6

REM		Test that the results are the same as last time
REM TODO .\src\Applications\DBCompare\bin\Release\DBCompare.exe -1 c:\SentryData\Sentry.yap -2 c:\SentryDataBackup\Sentry.yap -o .\DBCompareResults.txt



