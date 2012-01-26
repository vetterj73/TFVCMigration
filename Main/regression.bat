REM - VARIABLES
set OUTPUTDIR=C:\CyberStitchRegressionResults
set OUTPUTBACKUP=C:\CyberStitchRegressionResultsBackup
set APP=.\src\Applications\CyberStitchFidTester\bin\x64\Release\CyberStitchFidTester.exe
set SIMDATA=\\msp\dfs\archive\CyberStitchRegressionData\RegressionTest2\PPM(colorSim)\RegressionTest2Data(1micronPerCycle)
set CADDIR=\\msp\dfs\archive\CyberStitchRegressionData\RegressionTest2\PPM(colorSim)\RegressionTest2CAD

REM		Remove old results (prior to the last run)
if exist %OUTPUTBACKUP% rmdir /q /s %OUTPUTBACKUP%

REM		Copy last runs results to backup location.
if exist %OUTPUTDIR% move %OUTPUTDIR% %OUTPUTBACKUP%

REM		Create Output Directory
mkdir  %OUTPUTDIR%

REM     RUN All collected panels
%APP% -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes2Fids(diagonal).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\2Fids.txt -l %OUTPUTBACKUP%\2Fids.txt -u .\Results\UnitTest\ -n 6
%APP% -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes3Fids(triangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\3Fids.txt -l %OUTPUTBACKUP%\3Fids.txt -u .\Results\UnitTest\ -n 6
%APP% -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\4Fids.txt -l %OUTPUTBACKUP%\4Fids.txt -u .\Results\UnitTest\ -n 6
%APP% -b -w -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes2Fids(diagonal).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\2FidsProjective.txt -l %OUTPUTBACKUP%\2FidsProjective.txt -u .\Results\UnitTest\ -n 6
%APP% -b -w -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes3Fids(triangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\3FidsProjective.txt -l %OUTPUTBACKUP%\3FidsProjective.txt -u .\Results\UnitTest\ -n 6
%APP% -b -w -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\4FidsProjective.txt -l %OUTPUTBACKUP%\4FidsProjective.txt -u .\Results\UnitTest\ -n 6

set SIMDATA=\\msp\dfs\archive\CyberStitchRegressionData\RegressionTest2\PPM(colorSim)\RegressionTest2Data(40micronPerCycle)

REM     RUN All collected panels
%APP% -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes2Fids(diagonal).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\2Fids40microns.txt -l %OUTPUTBACKUP%\2Fids40microns.txt -u .\Results\UnitTest\ -n 10
%APP% -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes3Fids(triangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\3Fids40microns.txt -l %OUTPUTBACKUP%\3Fids40microns.txt -u .\Results\UnitTest\ -n 10
%APP% -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\4Fids40microns.txt -l %OUTPUTBACKUP%\4Fids40microns.txt -u .\Results\UnitTest\ -n 10
%APP% -b -w -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes2Fids(diagonal).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\2FidsProjective40microns.txt -l %OUTPUTBACKUP%\2FidsProjective40microns.txt -u .\Results\UnitTest\ -n 10
%APP% -b -w -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes3Fids(triangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\3FidsProjective40microns.txt -l %OUTPUTBACKUP%\3FidsProjective40microns.txt -u .\Results\UnitTest\ -n 10
%APP% -b -w -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\4FidsProjective40microns.txt -l %OUTPUTBACKUP%\4FidsProjective40microns.txt -u .\Results\UnitTest\ -n 10



set SIMDATA=\\msp\dfs\archive\CyberStitchRegressionData\RegressionTest4\PPM(colorSim)\RegressionTest4Data(40micronPerCycle)
set CADDIR=\\msp\dfs\archive\CyberStitchRegressionData\RegressionTest4\PPM(colorSim)\RegressionTest4CAD

REM     RUN All collected panels
%APP% -b -s "%SIMDATA%\NoInsertion\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\NoInsertion4Fids40microns.txt -l %OUTPUTBACKUP%\NoInsertion4Fids40microns.txt -u .\Results\UnitTest\ -n 10
%APP% -b -s "%SIMDATA%\1Insertion\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\1Insertion4Fids40microns.txt -l %OUTPUTBACKUP%\1Insertion4Fids40microns.txt -u .\Results\UnitTest\ -n 10
%APP% -b -s "%SIMDATA%\2Insertion\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\2Insertion4Fids40microns.txt -l %OUTPUTBACKUP%\2Insertion4Fids40microns.txt -u .\Results\UnitTest\ -n 10
%APP% -b -w -s "%SIMDATA%\NoInsertion\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\NoInsertion4FidsProjective40microns.txt -l %OUTPUTBACKUP%\NoInsertion4FidsProjective40microns.txt -u .\Results\UnitTest\ -n 10
%APP% -b -w -s "%SIMDATA%\1Insertion\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml"  -o  %OUTPUTDIR%\1Insertion4FidsProjective40microns.txt -l %OUTPUTBACKUP%\1Insertion4FidsProjective40microns.txt -u .\Results\UnitTest\ -n 10
%APP% -b -w -s "%SIMDATA%\2Insertion\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\2Insertion4FidsProjective40microns.txt -l %OUTPUTBACKUP%\2Insertion4FidsProjective40microns.txt -u .\Results\UnitTest\ -n 10
 


set SIMDATA=\\msp\dfs\archive\CyberStitchRegressionData\RegressionTest5\PPM(colorSim)\RegressionTest5Data(Z-4mmPerCycle)
set CADDIR=\\msp\dfs\archive\CyberStitchRegressionData\RegressionTest5\PPM(colorSim)\RegressionTest5CAD

REM     RUN All collected panels
%APP% -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\4FidsZ-4mms.txt -l %OUTPUTBACKUP%\4FidsZ-4mms.txt -u .\Results\UnitTest\ -n 10
%APP% -b -w -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\4FidsZ-4mmsProjective.txt -l %OUTPUTBACKUP%\4FidsZ-4mmsProjective.txt -u .\Results\UnitTest\ -n 10