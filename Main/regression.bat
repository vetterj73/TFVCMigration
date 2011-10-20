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
%APP% -b -s "%SIMDATA%SIMScenario.xml" -p "%CADDIR%PPMTestPanel-HighRes2Fids(diagonal).xml" -f "%CADDIR%PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%2Fids.txt -l %OUTPUTBACKUP%2Fids.txt -u .\Results\UnitTest\ -n 6
%APP% -b -s "%SIMDATA%SIMScenario.xml" -p "%CADDIR%PPMTestPanel-HighRes3Fids(triangle).xml" -f "%CADDIR%PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%3Fids.txt -l %OUTPUTBACKUP%3Fids.txt -u .\Results\UnitTest\ -n 6
%APP% -b -s "%SIMDATA%SIMScenario.xml" -p "%CADDIR%PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%4Fids.txt -l %OUTPUTBACKUP%4Fids.txt -u .\Results\UnitTest\ -n 6
%APP% -b -w -s "%SIMDATA%SIMScenario.xml" -p "%CADDIR%PPMTestPanel-HighRes2Fids(diagonal).xml" -f "%CADDIR%PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%2FidsProjective.txt -l %OUTPUTBACKUP%2FidsProjective.txt -u .\Results\UnitTest\ -n 6
%APP% -b -w -s "%SIMDATA%SIMScenario.xml" -p "%CADDIR%PPMTestPanel-HighRes3Fids(triangle).xml" -f "%CADDIR%PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%3FidsProjective.txt -l %OUTPUTBACKUP%3FidsProjective.txt -u .\Results\UnitTest\ -n 6
%APP% -b -w -s "%SIMDATA%SIMScenario.xml" -p "%CADDIR%PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%4FidsProjective.txt -l %OUTPUTBACKUP%4FidsProjective.txt -u .\Results\UnitTest\ -n 6



