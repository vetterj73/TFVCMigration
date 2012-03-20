rem '-w' flag should not matter if '-cammod' flag is set
set ROOTDIR=C:\CyberStitchRegression
set OUTPUTDIR=%ROOTDIR%\Results
set OUTPUTBACKUP=%ROOTDIR%\LastResults
set APP=.\src\Applications\CyberStitchFidTester\bin\x64\Release\CyberStitchFidTester.exe
set SIMDATA=\\msp\dfs\archive\CyberStitchRegressionData\RegressionTest2\PPM(colorSim)\RegressionTest2Data(1micronPerCycle)
set CADDIR=\\msp\dfs\archive\CyberStitchRegressionData\RegressionTest2\PPM(colorSim)\RegressionTest2CAD

REM		Remove old results (prior to the last run)
if exist %OUTPUTBACKUP% rmdir /q /s %OUTPUTBACKUP%

REM		Copy last runs results to backup location.
if exist %OUTPUTDIR% move %OUTPUTDIR% %OUTPUTBACKUP%

REM		Create Output Directory
mkdir  %OUTPUTDIR%

REM First run camera model version

REM     RUN All collected panels
%APP%  -cammod -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes2Fids(diagonal).xml"  -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\2FidsCameraModel.txt  -l %OUTPUTBACKUP%\2FidsCameraModel.txt    -u .\Results\UnitTest\ -n 6
%APP%  -cammod -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes3Fids(triangle).xml"  -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\3FidsCameraModel.txt  -l %OUTPUTBACKUP%\3FidsCameraModel.txt    -u .\Results\UnitTest\ -n 6
%APP%  -cammod -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\4FidsCameraModel.txt  -l %OUTPUTBACKUP%\4FidsCameraModel.txt    -u .\Results\UnitTest\ -n 6
%APP%  -cammod -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighResAllFids.xml"          -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\AllFidsCameraModel.txt -l %OUTPUTBACKUP%\AllFidsCameraModel.txt -u .\Results\UnitTest\ -n 6

%APP%  -iter   -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes2Fids(diagonal).xml"  -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\2FidsIterative.txt  -l %OUTPUTBACKUP%\2FidsIterative.txt    -u .\Results\UnitTest\ -n 6
%APP%  -iter   -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes3Fids(triangle).xml"  -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\3FidsIterative.txt  -l %OUTPUTBACKUP%\3FidsIterative.txt    -u .\Results\UnitTest\ -n 6
%APP%  -iter   -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\4FidsIterative.txt  -l %OUTPUTBACKUP%\4FidsIterative.txt    -u .\Results\UnitTest\ -n 6
%APP%  -iter   -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighResAllFids.xml"          -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\AllFidsIterative.txt -l %OUTPUTBACKUP%\AllFidsIterative.txt -u .\Results\UnitTest\ -n 6

set SIMDATA=\\msp\dfs\archive\CyberStitchRegressionData\RegressionTest2\PPM(colorSim)\RegressionTest2Data(40micronPerCycle)

REM     RUN All collected panels
%APP%  -cammod -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes2Fids(diagonal).xml"  -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\2Fids40micronsCameraModel.txt -l %OUTPUTBACKUP%\2Fids40micronsCameraModel.txt -u .\Results\UnitTest\ -n 10
%APP%  -cammod -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes3Fids(triangle).xml"  -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\3Fids40micronsCameraModel.txt -l %OUTPUTBACKUP%\3Fids40micronsCameraModel.txt -u .\Results\UnitTest\ -n 10
%APP%  -cammod -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\4Fids40micronsCameraModel.txt -l %OUTPUTBACKUP%\4Fids40micronsCameraModel.txt -u .\Results\UnitTest\ -n 10

%APP%  -iter   -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes2Fids(diagonal).xml"  -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\2Fids40micronsIterative.txt -l %OUTPUTBACKUP%\2Fids40micronsIterative.txt -u .\Results\UnitTest\ -n 10
%APP%  -iter   -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes3Fids(triangle).xml"  -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\3Fids40micronsIterative.txt -l %OUTPUTBACKUP%\3Fids40micronsIterative.txt -u .\Results\UnitTest\ -n 10
%APP%  -iter   -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\4Fids40micronsIterative.txt -l %OUTPUTBACKUP%\4Fids40micronsIterative.txt -u .\Results\UnitTest\ -n 10



set SIMDATA=\\msp\dfs\archive\CyberStitchRegressionData\RegressionTest4\PPM(colorSim)\RegressionTest4Data(40micronPerCycle)
set CADDIR=\\msp\dfs\archive\CyberStitchRegressionData\RegressionTest4\PPM(colorSim)\RegressionTest4CAD

REM     RUN All collected panels
%APP%  -cammod -b -s "%SIMDATA%\NoInsertion\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\NoInsertion4Fids40micronsCameraModel.txt -l %OUTPUTBACKUP%\NoInsertion4Fids40micronsCameraModel.txt -u .\Results\UnitTest\ -n 10
%APP%  -cammod -b -s "%SIMDATA%\1Insertion\SIMScenario.xml"  -p "%CADDIR%\PPMTestPanel-HighRes4fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\1Insertion4Fids40micronsCameraModel.txt  -l %OUTPUTBACKUP%\1Insertion4Fids40micronsCameraModel.txt  -u .\Results\UnitTest\ -n 10
%APP%  -cammod -b -s "%SIMDATA%\2Insertion\SIMScenario.xml"  -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\2Insertion4Fids40micronsCameraModel.txt  -l %OUTPUTBACKUP%\2Insertion4Fids40micronsCameraModel.txt  -u .\Results\UnitTest\ -n 10
 
%APP%  -iter   -b -s "%SIMDATA%\NoInsertion\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\NoInsertion4Fids40micronsIterative.txt -l %OUTPUTBACKUP%\NoInsertion4Fids40micronsIterative.txt -u .\Results\UnitTest\ -n 10
%APP%  -iter   -b -s "%SIMDATA%\1Insertion\SIMScenario.xml"  -p "%CADDIR%\PPMTestPanel-HighRes4fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\1Insertion4Fids40micronsIterative.txt  -l %OUTPUTBACKUP%\1Insertion4Fids40micronsIterative.txt  -u .\Results\UnitTest\ -n 10
%APP%  -iter   -b -s "%SIMDATA%\2Insertion\SIMScenario.xml"  -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\2Insertion4Fids40micronsIterative.txt  -l %OUTPUTBACKUP%\2Insertion4Fids40micronsIterative.txt  -u .\Results\UnitTest\ -n 10
 


set SIMDATA=\\msp\dfs\archive\CyberStitchRegressionData\RegressionTest5\PPM(colorSim)\RegressionTest5Data(Z-4mmPerCycle)
set CADDIR=\\msp\dfs\archive\CyberStitchRegressionData\RegressionTest5\PPM(colorSim)\RegressionTest5CAD

REM     RUN All collected panels
%APP%  -cammod -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\4FidsZ-4mmsCameraModel.txt -l %OUTPUTBACKUP%\4FidsZ-4mmsCameraModel.txt -u .\Results\UnitTest\ -n 10
%APP%  -iter   -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\4FidsZ-4mmsIterative.txt -l %OUTPUTBACKUP%\4FidsZ-4mmsIterative.txt -u .\Results\UnitTest\ -n 10


REM Next run stitch classic version
set SIMDATA=\\msp\dfs\archive\CyberStitchRegressionData\RegressionTest2\PPM(colorSim)\RegressionTest2Data(1micronPerCycle)
set CADDIR=\\msp\dfs\archive\CyberStitchRegressionData\RegressionTest2\PPM(colorSim)\RegressionTest2CAD

REM     RUN All collected panels
%APP% -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes2Fids(diagonal).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\2Fids.txt -l %OUTPUTBACKUP%\2Fids.txt -u .\Results\UnitTest\ -n 6
%APP% -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes3Fids(triangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\3Fids.txt -l %OUTPUTBACKUP%\3Fids.txt -u .\Results\UnitTest\ -n 6
%APP% -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\4Fids.txt -l %OUTPUTBACKUP%\4Fids.txt -u .\Results\UnitTest\ -n 6
%APP% -b -w -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes2Fids(diagonal).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\2FidsProjective.txt -l %OUTPUTBACKUP%\2FidsProjective.txt -u .\Results\UnitTest\ -n 6
%APP% -b -w -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes3Fids(triangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\3FidsProjective.txt -l %OUTPUTBACKUP%\3FidsProjective.txt -u .\Results\UnitTest\ -n 6
%APP% -b -w -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\4FidsProjective.txt -l %OUTPUTBACKUP%\4FidsProjective.txt -u .\Results\UnitTest\ -n 6
%APP% -b -w -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighResAllFids.xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\AllFidsProjective.txt -l %OUTPUTBACKUP%\AllFidsProjective.txt -u .\Results\UnitTest\ -n 6
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
%APP% -de -b -w -s "%SIMDATA%\NoInsertion\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\NoInsertion4FidsProjectiveEdge40microns.txt -l %OUTPUTBACKUP%\NoInsertion4FidsProjectiveEdge40microns.txt -u .\Results\UnitTest\ -n 10
%APP% -de -b -w -s "%SIMDATA%\1Insertion\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml"  -o  %OUTPUTDIR%\1Insertion4FidsProjectiveEdge40microns.txt -l %OUTPUTBACKUP%\1Insertion4FidsProjectiveEdge40microns.txt -u .\Results\UnitTest\ -n 10
%APP% -de -b -w -s "%SIMDATA%\2Insertion\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\2Insertion4FidsProjectiveEdge40microns.txt -l %OUTPUTBACKUP%\2Insertion4FidsProjectiveEdge40microns.txt -u .\Results\UnitTest\ -n 10
 


set SIMDATA=\\msp\dfs\archive\CyberStitchRegressionData\RegressionTest5\PPM(colorSim)\RegressionTest5Data(Z-4mmPerCycle)
set CADDIR=\\msp\dfs\archive\CyberStitchRegressionData\RegressionTest5\PPM(colorSim)\RegressionTest5CAD

REM     RUN All collected panels
%APP% -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\4FidsZ-4mms.txt -l %OUTPUTBACKUP%\4FidsZ-4mms.txt -u .\Results\UnitTest\ -n 10
%APP% -b -w -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\4FidsZ-4mmsProjective.txt -l %OUTPUTBACKUP%\4FidsZ-4mmsProjective.txt -u .\Results\UnitTest\ -n 10
%APP% -de -b -w -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\4FidsZ-4mmsProjectiveEdge.txt -l %OUTPUTBACKUP%\4FidsZ-4mmsProjectiveEdge.txt -u .\Results\UnitTest\ -n 10


set SIMDATA=\\msp\dfs\archive\CyberStitchRegressionData\RegressionTest7
set CADDIR=\\msp\dfs\archive\CyberStitchRegressionData\RegressionTest7

REM     RUN All collected panels
%APP% -b -s "%SIMDATA%\PPM\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\FFR_LtoR.txt -l %OUTPUTBACKUP%\FFR_LtoR.txt -u .\Results\UnitTest\ -n 5
%APP% -b -s "%SIMDATA%\PPM_FRR\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\FRR_LtoR.txt -l %OUTPUTBACKUP%\FRR_LtoR.txt -u .\Results\UnitTest\ -n 5
%APP% -b -s "%SIMDATA%\PPM_RtoL\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\FFR_RtoL.txt -l %OUTPUTBACKUP%\FFR_RtoL.txt -u .\Results\UnitTest\ -n 5
%APP% -b -s "%SIMDATA%\PPM_FRR_RtoL\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\FRR_RtoL.txt -l %OUTPUTBACKUP%\FRR_RtoL.txt -u .\Results\UnitTest\ -n 5

set SIMDATA=\\msp\dfs\archive\CyberStitchRegressionData\RegressionTest8
set CADDIR=\\msp\dfs\archive\CyberStitchRegressionData\RegressionTest8

%APP% -de -b -w -s "%SIMDATA%\SentryData3(FRRandRtoL)\Raw\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\EdgeFRR_RtoL.txt -l %OUTPUTBACKUP%\EdgeFRR_RtoL.txt -u .\Results\UnitTest\ 
%APP% -de -b -w -s "%SIMDATA%\SentryData4(FRR)\Raw\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\EdgeFRR_LtoR.txt -l %OUTPUTBACKUP%\EdgeFRR_LtoR.txt -u .\Results\UnitTest\ 
%APP% -de -b -w -s "%SIMDATA%\SentryData5(RtoL)\Raw\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\EdgeFFR_RtoL.txt -l %OUTPUTBACKUP%\EdgeFFR_RtoL.txt -u .\Results\UnitTest\ 
%APP% -b -w -s "%SIMDATA%\SentryData3(FRRandRtoL)\Raw\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\NoEdgeFRR_RtoL.txt -l %OUTPUTBACKUP%\NoEdgeFRR_RtoL.txt -u .\Results\UnitTest\ 
%APP% -b -w -s "%SIMDATA%\SentryData4(FRR)\Raw\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\NoEdgeFRR_LtoR.txt -l %OUTPUTBACKUP%\NoEdgeFRR_LtoR.txt -u .\Results\UnitTest\ 
%APP% -b -w -s "%SIMDATA%\SentryData5(RtoL)\Raw\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\NoEdgeFFR_RtoL.txt -l %OUTPUTBACKUP%\NoEdgeFFR_RtoL.txt -u .\Results\UnitTest\ 


 
REM Get the current date and time in YYYY-MM-DD-HH-MM-SS format
SET THEDATE=%date:~10,4%-%date:~4,2%-%date:~7,2%-%time:~0,2%-%time:~3,2%-%time:~6,2%
REM Convert blanks to zeros...
set THEDATE=%THEDATE: =0%
mkdir %ROOTDIR%\%THEDATE%
xcopy /S /E /Y /Q %OUTPUTDIR%\* %ROOTDIR%\%THEDATE%\
