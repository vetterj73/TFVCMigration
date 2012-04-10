rem '-w' flag should not matter if '-cammod' flag is set
set ROOTDIR=C:\CyberStitchRegression
set OUTPUTDIR=%ROOTDIR%\Results
set OUTPUTBACKUP=%ROOTDIR%\LastResults
set APP=.\src\Applications\CyberStitchFidTester\bin\x64\Release\CyberStitchFidTester.exe
ROBOCOPY "\\msp\dfs\archive\CyberStitchRegressionData"  "E:\CyberStitchRegressionData" /s
set SIMDATA=E:\CyberStitchRegressionData\RegressionTest2\PPM(colorSim)\RegressionTest2Data(1micronPerCycle)
set CADDIR=E:\CyberStitchRegressionData\RegressionTest2\PPM(colorSim)\RegressionTest2CAD

REM		Remove old results (prior to the last run)
if exist %OUTPUTBACKUP% rmdir /q /s %OUTPUTBACKUP%

REM		Copy last runs results to backup location.
if exist %OUTPUTDIR% move %OUTPUTDIR% %OUTPUTBACKUP%

REM		Create Output Directory
mkdir  %OUTPUTDIR%

REM First run camera model version

REM     RUN All collected panels
%APP%  -cammod -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes2Fids(diagonal).xml"  -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\2FidsCameraModel.txt  -l %OUTPUTBACKUP%\2FidsCameraModel.txt    -u .\Results\UnitTest\ -n 5
%APP%  -cammod -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes3Fids(triangle).xml"  -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\3FidsCameraModel.txt  -l %OUTPUTBACKUP%\3FidsCameraModel.txt    -u .\Results\UnitTest\ -n 5
%APP%  -cammod -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\4FidsCameraModel.txt  -l %OUTPUTBACKUP%\4FidsCameraModel.txt    -u .\Results\UnitTest\ -n 5
%APP%  -cammod -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighResAllFids.xml"          -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\AllFidsCameraModel.txt -l %OUTPUTBACKUP%\AllFidsCameraModel.txt -u .\Results\UnitTest\ -n 5

%APP%  -iter   -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes2Fids(diagonal).xml"  -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\2FidsIterative.txt  -l %OUTPUTBACKUP%\2FidsIterative.txt    -u .\Results\UnitTest\ -n 5
%APP%  -iter   -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes3Fids(triangle).xml"  -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\3FidsIterative.txt  -l %OUTPUTBACKUP%\3FidsIterative.txt    -u .\Results\UnitTest\ -n 5
%APP%  -iter   -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\4FidsIterative.txt  -l %OUTPUTBACKUP%\4FidsIterative.txt    -u .\Results\UnitTest\ -n 5
%APP%  -iter   -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighResAllFids.xml"          -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\AllFidsIterative.txt -l %OUTPUTBACKUP%\AllFidsIterative.txt -u .\Results\UnitTest\ -n 5

set SIMDATA=E:\CyberStitchRegressionData\RegressionTest2\PPM(colorSim)\RegressionTest2Data(40micronPerCycle)

REM     RUN All collected panels
%APP%  -cammod -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes2Fids(diagonal).xml"  -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\2Fids40micronsCameraModel.txt -l %OUTPUTBACKUP%\2Fids40micronsCameraModel.txt -u .\Results\UnitTest\ -n 5
%APP%  -cammod -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes3Fids(triangle).xml"  -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\3Fids40micronsCameraModel.txt -l %OUTPUTBACKUP%\3Fids40micronsCameraModel.txt -u .\Results\UnitTest\ -n 5
%APP%  -cammod -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\4Fids40micronsCameraModel.txt -l %OUTPUTBACKUP%\4Fids40micronsCameraModel.txt -u .\Results\UnitTest\ -n 5

%APP%  -iter   -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes2Fids(diagonal).xml"  -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\2Fids40micronsIterative.txt -l %OUTPUTBACKUP%\2Fids40micronsIterative.txt -u .\Results\UnitTest\ -n 5
%APP%  -iter   -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes3Fids(triangle).xml"  -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\3Fids40micronsIterative.txt -l %OUTPUTBACKUP%\3Fids40micronsIterative.txt -u .\Results\UnitTest\ -n 5
%APP%  -iter   -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\4Fids40micronsIterative.txt -l %OUTPUTBACKUP%\4Fids40micronsIterative.txt -u .\Results\UnitTest\ -n 5



set SIMDATA=E:\CyberStitchRegressionData\RegressionTest4\PPM(colorSim)\RegressionTest4Data(40micronPerCycle)
set CADDIR=E:\CyberStitchRegressionData\RegressionTest4\PPM(colorSim)\RegressionTest4CAD

REM     RUN All collected panels
%APP%  -cammod -b -s "%SIMDATA%\NoInsertion\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\NoInsertion4Fids40micronsCameraModel.txt -l %OUTPUTBACKUP%\NoInsertion4Fids40micronsCameraModel.txt -u .\Results\UnitTest\ -n 5
%APP%  -cammod -b -s "%SIMDATA%\1Insertion\SIMScenario.xml"  -p "%CADDIR%\PPMTestPanel-HighRes4fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\1Insertion4Fids40micronsCameraModel.txt  -l %OUTPUTBACKUP%\1Insertion4Fids40micronsCameraModel.txt  -u .\Results\UnitTest\ -n 5
%APP%  -cammod -b -s "%SIMDATA%\2Insertion\SIMScenario.xml"  -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\2Insertion4Fids40micronsCameraModel.txt  -l %OUTPUTBACKUP%\2Insertion4Fids40micronsCameraModel.txt  -u .\Results\UnitTest\ -n 5
 
%APP%  -iter   -b -s "%SIMDATA%\NoInsertion\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\NoInsertion4Fids40micronsIterative.txt -l %OUTPUTBACKUP%\NoInsertion4Fids40micronsIterative.txt -u .\Results\UnitTest\ -n 5
%APP%  -iter   -b -s "%SIMDATA%\1Insertion\SIMScenario.xml"  -p "%CADDIR%\PPMTestPanel-HighRes4fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\1Insertion4Fids40micronsIterative.txt  -l %OUTPUTBACKUP%\1Insertion4Fids40micronsIterative.txt  -u .\Results\UnitTest\ -n 5
%APP%  -iter   -b -s "%SIMDATA%\2Insertion\SIMScenario.xml"  -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\2Insertion4Fids40micronsIterative.txt  -l %OUTPUTBACKUP%\2Insertion4Fids40micronsIterative.txt  -u .\Results\UnitTest\ -n 5
 


set SIMDATA=E:\CyberStitchRegressionData\RegressionTest5\PPM(colorSim)\RegressionTest5Data(Z-4mmPerCycle)
set CADDIR=E:\CyberStitchRegressionData\RegressionTest5\PPM(colorSim)\RegressionTest5CAD

REM     RUN All collected panels
%APP%  -cammod -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\4FidsZ-4mmsCameraModel.txt -l %OUTPUTBACKUP%\4FidsZ-4mmsCameraModel.txt -u .\Results\UnitTest\ -n 10
%APP%  -iter   -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\4FidsZ-4mmsIterative.txt -l %OUTPUTBACKUP%\4FidsZ-4mmsIterative.txt -u .\Results\UnitTest\ -n 10


REM Next run stitch classic version
set SIMDATA=E:\CyberStitchRegressionData\RegressionTest2\PPM(colorSim)\RegressionTest2Data(1micronPerCycle)
set CADDIR=E:\CyberStitchRegressionData\RegressionTest2\PPM(colorSim)\RegressionTest2CAD

REM     RUN All collected panels
%APP% -b -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes2Fids(diagonal).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\2Fids.txt -l %OUTPUTBACKUP%\2Fids.txt -u .\Results\UnitTest\ -n 5
%APP% -b -w -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes2Fids(diagonal).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\2FidsProjective.txt -l %OUTPUTBACKUP%\2FidsProjective.txt -u .\Results\UnitTest\ -n 5
%APP% -b -w -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes3Fids(triangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\3FidsProjective.txt -l %OUTPUTBACKUP%\3FidsProjective.txt -u .\Results\UnitTest\ -n 5
%APP% -b -w -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\4FidsProjective.txt -l %OUTPUTBACKUP%\4FidsProjective.txt -u .\Results\UnitTest\ -n 5
%APP% -b -w -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighResAllFids.xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\AllFidsProjective.txt -l %OUTPUTBACKUP%\AllFidsProjective.txt -u .\Results\UnitTest\ -n 5
set SIMDATA=E:\CyberStitchRegressionData\RegressionTest2\PPM(colorSim)\RegressionTest2Data(40micronPerCycle)

REM     RUN All collected panels
%APP% -b -w -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes2Fids(diagonal).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\2FidsProjective40microns.txt -l %OUTPUTBACKUP%\2FidsProjective40microns.txt -u .\Results\UnitTest\ -n 5
%APP% -b -w -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes3Fids(triangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\3FidsProjective40microns.txt -l %OUTPUTBACKUP%\3FidsProjective40microns.txt -u .\Results\UnitTest\ -n 5
%APP% -b -w -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\4FidsProjective40microns.txt -l %OUTPUTBACKUP%\4FidsProjective40microns.txt -u .\Results\UnitTest\ -n 5



set SIMDATA=E:\CyberStitchRegressionData\RegressionTest4\PPM(colorSim)\RegressionTest4Data(40micronPerCycle)
set CADDIR=E:\CyberStitchRegressionData\RegressionTest4\PPM(colorSim)\RegressionTest4CAD

REM     RUN All collected panels
%APP% -b -w -s "%SIMDATA%\NoInsertion\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\NoInsertion4FidsProjective40microns.txt -l %OUTPUTBACKUP%\NoInsertion4FidsProjective40microns.txt -u .\Results\UnitTest\ -n 5
%APP% -b -w -s "%SIMDATA%\1Insertion\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml"  -o  %OUTPUTDIR%\1Insertion4FidsProjective40microns.txt -l %OUTPUTBACKUP%\1Insertion4FidsProjective40microns.txt -u .\Results\UnitTest\ -n 5
%APP% -b -w -s "%SIMDATA%\2Insertion\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\2Insertion4FidsProjective40microns.txt -l %OUTPUTBACKUP%\2Insertion4FidsProjective40microns.txt -u .\Results\UnitTest\ -n 5
%APP% -de -b -w -s "%SIMDATA%\NoInsertion\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\NoInsertion4FidsProjectiveEdge40microns.txt -l %OUTPUTBACKUP%\NoInsertion4FidsProjectiveEdge40microns.txt -u .\Results\UnitTest\ -n 5
%APP% -de -b -w -s "%SIMDATA%\1Insertion\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml"  -o  %OUTPUTDIR%\1Insertion4FidsProjectiveEdge40microns.txt -l %OUTPUTBACKUP%\1Insertion4FidsProjectiveEdge40microns.txt -u .\Results\UnitTest\ -n 5
%APP% -de -b -w -s "%SIMDATA%\2Insertion\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\2Insertion4FidsProjectiveEdge40microns.txt -l %OUTPUTBACKUP%\2Insertion4FidsProjectiveEdge40microns.txt -u .\Results\UnitTest\ -n 5
 


set SIMDATA=E:\CyberStitchRegressionData\RegressionTest5\PPM(colorSim)\RegressionTest5Data(Z-4mmPerCycle)
set CADDIR=E:\CyberStitchRegressionData\RegressionTest5\PPM(colorSim)\RegressionTest5CAD

REM     RUN All collected panels
%APP% -b -w -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\4FidsZ-4mmsProjective.txt -l %OUTPUTBACKUP%\4FidsZ-4mmsProjective.txt -u .\Results\UnitTest\ -n 10
%APP% -de -b -w -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\4FidsZ-4mmsProjectiveEdge.txt -l %OUTPUTBACKUP%\4FidsZ-4mmsProjectiveEdge.txt -u .\Results\UnitTest\ -n 10

REM  Populated JUKI board data set 
REM  this panel has a wide variety of components
set SIMDATA=E:\CyberStitchRegressionData\RegressionTest6\JukiSinglePWBfor2080(colorSim)\RegressionTest6Data(40micronPerCycle)
set CADDIR=E:\CyberStitchRegressionData\RegressionTest6\JukiSinglePWBfor2080(colorSim)\RegressionTest6CAD

%APP%     -b -w      -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\JukiSinglePWBfor2080(3Fids)Mitutoyo.xml" -f "%CADDIR%\JukiSinglePWBfor2080(AllFids_NoPad)Mitutoyo.xml" -o  %OUTPUTDIR%\JukiSinglePWBPopulatedProjective.txt      -l %OUTPUTBACKUP%\JukiSinglePWBPopulatedProjective.txt -u .\Results\UnitTest\ -n 4
%APP% -de -b -w      -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\JukiSinglePWBfor2080(3Fids)Mitutoyo.xml" -f "%CADDIR%\JukiSinglePWBfor2080(AllFids_NoPad)Mitutoyo.xml" -o  %OUTPUTDIR%\JukiSinglePWBPopulatedProjectiveEdge.txt  -l %OUTPUTBACKUP%\JukiSinglePWBPopulatedProjectiveEdge.txt -u .\Results\UnitTest\ -n 4
%APP%     -b -cammod -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\JukiSinglePWBfor2080(3Fids)Mitutoyo.xml" -f "%CADDIR%\JukiSinglePWBfor2080(AllFids_NoPad)Mitutoyo.xml" -o  %OUTPUTDIR%\JukiSinglePWBPopulatedCameraModel.txt     -l %OUTPUTBACKUP%\JukiSinglePWBPopulatedCameraModel.txt -u .\Results\UnitTest\ -n 4
%APP% -de -b -cammod -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\JukiSinglePWBfor2080(3Fids)Mitutoyo.xml" -f "%CADDIR%\JukiSinglePWBfor2080(AllFids_NoPad)Mitutoyo.xml" -o  %OUTPUTDIR%\JukiSinglePWBPopulatedCameraModelEdge.txt -l %OUTPUTBACKUP%\JukiSinglePWBPopulatedCameraModelEdge.txt -u .\Results\UnitTest\ -n 4
%APP%     -b -iter   -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\JukiSinglePWBfor2080(3Fids)Mitutoyo.xml" -f "%CADDIR%\JukiSinglePWBfor2080(AllFids_NoPad)Mitutoyo.xml" -o  %OUTPUTDIR%\JukiSinglePWBPopulatedIterative.txt       -l %OUTPUTBACKUP%\JukiSinglePWBPopulatedIterative.txt -u .\Results\UnitTest\ -n 4
%APP% -de -b -iter   -s "%SIMDATA%\SIMScenario.xml" -p "%CADDIR%\JukiSinglePWBfor2080(3Fids)Mitutoyo.xml" -f "%CADDIR%\JukiSinglePWBfor2080(AllFids_NoPad)Mitutoyo.xml" -o  %OUTPUTDIR%\JukiSinglePWBPopulatedIterativeEdge.txt   -l %OUTPUTBACKUP%\JukiSinglePWBPopulatedIterativeEdge.txt -u .\Results\UnitTest\ -n 4

REM
set SIMDATA=E:\CyberStitchRegressionData\RegressionTest7
set CADDIR=E:\CyberStitchRegressionData\RegressionTest7

REM     RUN All collected panels
%APP% -b -s "%SIMDATA%\PPM\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\FFR_LtoR.txt -l %OUTPUTBACKUP%\FFR_LtoR.txt -u .\Results\UnitTest\ -n 5
%APP% -b -s "%SIMDATA%\PPM_FRR\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\FRR_LtoR.txt -l %OUTPUTBACKUP%\FRR_LtoR.txt -u .\Results\UnitTest\ -n 5
%APP% -b -s "%SIMDATA%\PPM_RtoL\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\FFR_RtoL.txt -l %OUTPUTBACKUP%\FFR_RtoL.txt -u .\Results\UnitTest\ -n 5
%APP% -b -s "%SIMDATA%\PPM_FRR_RtoL\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\FRR_RtoL.txt -l %OUTPUTBACKUP%\FRR_RtoL.txt -u .\Results\UnitTest\ -n 5

set SIMDATA=E:\CyberStitchRegressionData\RegressionTest8
set CADDIR=E:\CyberStitchRegressionData\RegressionTest8

%APP% -de -b -w -s "%SIMDATA%\SentryData3(FRRandRtoL)\Raw\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\EdgeFRR_RtoL.txt -l %OUTPUTBACKUP%\EdgeFRR_RtoL.txt -u .\Results\UnitTest\ 
%APP% -de -b -w -s "%SIMDATA%\SentryData4(FRR)\Raw\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\EdgeFRR_LtoR.txt -l %OUTPUTBACKUP%\EdgeFRR_LtoR.txt -u .\Results\UnitTest\ 
%APP% -de -b -w -s "%SIMDATA%\SentryData5(RtoL)\Raw\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\EdgeFFR_RtoL.txt -l %OUTPUTBACKUP%\EdgeFFR_RtoL.txt -u .\Results\UnitTest\ 
%APP% -b -w -s "%SIMDATA%\SentryData3(FRRandRtoL)\Raw\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\NoEdgeFRR_RtoL.txt -l %OUTPUTBACKUP%\NoEdgeFRR_RtoL.txt -u .\Results\UnitTest\ 
%APP% -b -w -s "%SIMDATA%\SentryData4(FRR)\Raw\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\NoEdgeFRR_LtoR.txt -l %OUTPUTBACKUP%\NoEdgeFRR_LtoR.txt -u .\Results\UnitTest\ 
%APP% -b -w -s "%SIMDATA%\SentryData5(RtoL)\Raw\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\NoEdgeFFR_RtoL.txt -l %OUTPUTBACKUP%\NoEdgeFFR_RtoL.txt -u .\Results\UnitTest\ 
REM save as above but using camera model and iterative solvers
%APP% -de -b -cammod -s "%SIMDATA%\SentryData3(FRRandRtoL)\Raw\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\EdgeFRR_RtoLCameraModel.txt -l %OUTPUTBACKUP%\EdgeFRR_RtoLCameraModel.txt -u .\Results\UnitTest\ 
%APP% -de -b -cammod -s "%SIMDATA%\SentryData4(FRR)\Raw\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\EdgeFRR_LtoRCameraModel.txt -l %OUTPUTBACKUP%\EdgeFRR_LtoRCameraModel.txt -u .\Results\UnitTest\ 
%APP% -de -b -cammod -s "%SIMDATA%\SentryData5(RtoL)\Raw\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\EdgeFFR_RtoLCameraModel.txt -l %OUTPUTBACKUP%\EdgeFFR_RtoLCameraModel.txt -u .\Results\UnitTest\ 
%APP% -b -cammod -s "%SIMDATA%\SentryData3(FRRandRtoL)\Raw\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\NoEdgeFRR_RtoLCameraModel.txt -l %OUTPUTBACKUP%\NoEdgeFRR_RtoLCameraModel.txt -u .\Results\UnitTest\ 
%APP% -b -cammod -s "%SIMDATA%\SentryData4(FRR)\Raw\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\NoEdgeFRR_LtoRCameraModel.txt -l %OUTPUTBACKUP%\NoEdgeFRR_LtoRCameraModel.txt -u .\Results\UnitTest\ 
%APP% -b -cammod -s "%SIMDATA%\SentryData5(RtoL)\Raw\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\NoEdgeFFR_RtoLCameraModel.txt -l %OUTPUTBACKUP%\NoEdgeFFR_RtoLCameraModel.txt -u .\Results\UnitTest\ 

%APP% -de -b -iter -s "%SIMDATA%\SentryData3(FRRandRtoL)\Raw\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\EdgeFRR_RtoLIterative.txt -l %OUTPUTBACKUP%\EdgeFRR_RtoLIterative.txt -u .\Results\UnitTest\ 
%APP% -de -b -iter -s "%SIMDATA%\SentryData4(FRR)\Raw\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\EdgeFRR_LtoRIterative.txt -l %OUTPUTBACKUP%\EdgeFRR_LtoRIterative.txt -u .\Results\UnitTest\ 
%APP% -de -b -iter -s "%SIMDATA%\SentryData5(RtoL)\Raw\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\EdgeFFR_RtoLIterative.txt -l %OUTPUTBACKUP%\EdgeFFR_RtoLIterative.txt -u .\Results\UnitTest\ 
%APP% -b -iter -s "%SIMDATA%\SentryData3(FRRandRtoL)\Raw\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\NoEdgeFRR_RtoLIterative.txt -l %OUTPUTBACKUP%\NoEdgeFRR_RtoLIterative.txt -u .\Results\UnitTest\ 
%APP% -b -iter -s "%SIMDATA%\SentryData4(FRR)\Raw\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\NoEdgeFRR_LtoRIterative.txt -l %OUTPUTBACKUP%\NoEdgeFRR_LtoRIterative.txt -u .\Results\UnitTest\ 
%APP% -b -iter -s "%SIMDATA%\SentryData5(RtoL)\Raw\SIMScenario.xml" -p "%CADDIR%\PPMTestPanel-HighRes4Fids(rectangle).xml" -f "%CADDIR%\PPMTestPanel-HighResAllFidsNoPads.xml" -o  %OUTPUTDIR%\NoEdgeFFR_RtoLIterative.txt -l %OUTPUTBACKUP%\NoEdgeFFR_RtoLIterative.txt -u .\Results\UnitTest\ 


REM DEK-VG board, including cases of shifted calibration data
set SIMDATA=E:\CyberStitchRegressionData\RegressionTest9
set CADDIR= E:\CyberStitchRegressionData\RegressionTest9

REM DEK-VG board
%APP% -cammod -n 1 -s "%SIMDATA%\DEK_VG\SIMScenario.xml" -p "%CADDIR%\DEK_VG\DEK_VG.xml"       -f "%CADDIR%\DEK_VG\DEK_VG.xml" -o %OUTPUTDIR%\DEK-VG_4FidsCameraModel.txt -l %OUTPUTBACKUP%\DEK-VG_4FidsCameraModel.txt  -u .\Results\UnitTest\ 
%APP% -cammod -n 1 -s "%SIMDATA%\DEK_VG\SIMScenario.xml" -p "%CADDIR%\DEK_VG\DEK_VG_3Fids.xml" -f "%CADDIR%\DEK_VG\DEK_VG.xml" -o %OUTPUTDIR%\DEK-VG_3FidsCameraModel.txt -l %OUTPUTBACKUP%\DEK-VG_3FidsCameraModel.txt  -u .\Results\UnitTest\ 
%APP% -cammod -n 1 -s "%SIMDATA%\DEK_VG\SIMScenario.xml" -p "%CADDIR%\DEK_VG\DEK_VG_2Fids.xml" -f "%CADDIR%\DEK_VG\DEK_VG.xml" -o %OUTPUTDIR%\DEK-VG_2FidsCameraModel.txt -l %OUTPUTBACKUP%\DEK-VG_2FidsCameraModel.txt  -u .\Results\UnitTest\ 
%APP% -iter   -n 1 -s "%SIMDATA%\DEK_VG\SIMScenario.xml" -p "%CADDIR%\DEK_VG\DEK_VG.xml"       -f "%CADDIR%\DEK_VG\DEK_VG.xml" -o %OUTPUTDIR%\DEK-VG_4FidsIterative.txt   -l %OUTPUTBACKUP%\DEK-VG_4FidsIterative.txt  -u .\Results\UnitTest\ 
%APP% -iter   -n 1 -s "%SIMDATA%\DEK_VG\SIMScenario.xml" -p "%CADDIR%\DEK_VG\DEK_VG_3Fids.xml" -f "%CADDIR%\DEK_VG\DEK_VG.xml" -o %OUTPUTDIR%\DEK-VG_3FidsIterative.txt   -l %OUTPUTBACKUP%\DEK-VG_3FidsIterative.txt  -u .\Results\UnitTest\ 
%APP% -iter   -n 1 -s "%SIMDATA%\DEK_VG\SIMScenario.xml" -p "%CADDIR%\DEK_VG\DEK_VG_2Fids.xml" -f "%CADDIR%\DEK_VG\DEK_VG.xml" -o %OUTPUTDIR%\DEK-VG_2FidsIterative.txt   -l %OUTPUTBACKUP%\DEK-VG_2FidsIterative.txt  -u .\Results\UnitTest\ 
%APP% -w      -n 1 -s "%SIMDATA%\DEK_VG\SIMScenario.xml" -p "%CADDIR%\DEK_VG\DEK_VG.xml"       -f "%CADDIR%\DEK_VG\DEK_VG.xml" -o %OUTPUTDIR%\DEK-VG_4FidsProjective.txt -l %OUTPUTBACKUP%\DEK-VG_4FidsProjective.txt  -u .\Results\UnitTest\ 
%APP% -w      -n 1 -s "%SIMDATA%\DEK_VG\SIMScenario.xml" -p "%CADDIR%\DEK_VG\DEK_VG_3Fids.xml" -f "%CADDIR%\DEK_VG\DEK_VG.xml" -o %OUTPUTDIR%\DEK-VG_3FidsProjective.txt -l %OUTPUTBACKUP%\DEK-VG_3FidsProjective.txt  -u .\Results\UnitTest\ 
%APP% -w      -n 1 -s "%SIMDATA%\DEK_VG\SIMScenario.xml" -p "%CADDIR%\DEK_VG\DEK_VG_2Fids.xml" -f "%CADDIR%\DEK_VG\DEK_VG.xml" -o %OUTPUTDIR%\DEK-VG_2FidsProjective.txt -l %OUTPUTBACKUP%\DEK-VG_2FidsProjective.txt  -u .\Results\UnitTest\ 

REM DEK-VG board with skewed calibration
%APP% -cammod -n 1 -s "%SIMDATA%\DEK_VG\SIMScenario_offsetCal_01.xml" -p "%CADDIR%\DEK_VG\DEK_VG.xml"       -f "%CADDIR%\DEK_VG\DEK_VG.xml" -o %OUTPUTDIR%\DEK-VG_4FidsCameraModel_offsetCal.txt -l %OUTPUTBACKUP%\DEK-VG_4FidsCameraModel_offsetCal.txt  -u .\Results\UnitTest\ 
%APP% -cammod -n 1 -s "%SIMDATA%\DEK_VG\SIMScenario_offsetCal_01.xml" -p "%CADDIR%\DEK_VG\DEK_VG_3Fids.xml" -f "%CADDIR%\DEK_VG\DEK_VG.xml" -o %OUTPUTDIR%\DEK-VG_3FidsCameraModel_offsetCal.txt -l %OUTPUTBACKUP%\DEK-VG_3FidsCameraModel_offsetCal.txt  -u .\Results\UnitTest\ 
%APP% -cammod -n 1 -s "%SIMDATA%\DEK_VG\SIMScenario_offsetCal_01.xml" -p "%CADDIR%\DEK_VG\DEK_VG_2Fids.xml" -f "%CADDIR%\DEK_VG\DEK_VG.xml" -o %OUTPUTDIR%\DEK-VG_2FidsCameraModel_offsetCal.txt -l %OUTPUTBACKUP%\DEK-VG_2FidsCameraModel_offsetCal.txt  -u .\Results\UnitTest\ 
%APP% -iter   -n 1 -s "%SIMDATA%\DEK_VG\SIMScenario_offsetCal_01.xml" -p "%CADDIR%\DEK_VG\DEK_VG.xml"       -f "%CADDIR%\DEK_VG\DEK_VG.xml" -o %OUTPUTDIR%\DEK-VG_4FidsIterative_offsetCal.txt   -l %OUTPUTBACKUP%\DEK-VG_4FidsIterative_offsetCal.txt  -u .\Results\UnitTest\ 
%APP% -iter   -n 1 -s "%SIMDATA%\DEK_VG\SIMScenario_offsetCal_01.xml" -p "%CADDIR%\DEK_VG\DEK_VG_3Fids.xml" -f "%CADDIR%\DEK_VG\DEK_VG.xml" -o %OUTPUTDIR%\DEK-VG_3FidsIterative_offsetCal.txt   -l %OUTPUTBACKUP%\DEK-VG_3FidsIterative_offsetCal.txt  -u .\Results\UnitTest\ 
%APP% -iter   -n 1 -s "%SIMDATA%\DEK_VG\SIMScenario_offsetCal_01.xml" -p "%CADDIR%\DEK_VG\DEK_VG_2Fids.xml" -f "%CADDIR%\DEK_VG\DEK_VG.xml" -o %OUTPUTDIR%\DEK-VG_2FidsIterative_offsetCal.txt   -l %OUTPUTBACKUP%\DEK-VG_2FidsIterative_offsetCal.txt  -u .\Results\UnitTest\ 
%APP% -w      -n 1 -s "%SIMDATA%\DEK_VG\SIMScenario_offsetCal_01.xml" -p "%CADDIR%\DEK_VG\DEK_VG.xml"       -f "%CADDIR%\DEK_VG\DEK_VG.xml" -o %OUTPUTDIR%\DEK-VG_4FidsProjective_offsetCal.txt -l %OUTPUTBACKUP%\DEK-VG_4FidsProjective_offsetCal.txt  -u .\Results\UnitTest\ 
%APP% -w      -n 1 -s "%SIMDATA%\DEK_VG\SIMScenario_offsetCal_01.xml" -p "%CADDIR%\DEK_VG\DEK_VG_3Fids.xml" -f "%CADDIR%\DEK_VG\DEK_VG.xml" -o %OUTPUTDIR%\DEK-VG_3FidsProjective_offsetCal.txt -l %OUTPUTBACKUP%\DEK-VG_3FidsProjective_offsetCal.txt  -u .\Results\UnitTest\ 
%APP% -w      -n 1 -s "%SIMDATA%\DEK_VG\SIMScenario_offsetCal_01.xml" -p "%CADDIR%\DEK_VG\DEK_VG_2Fids.xml" -f "%CADDIR%\DEK_VG\DEK_VG.xml" -o %OUTPUTDIR%\DEK-VG_2FidsProjective_offsetCal.txt -l %OUTPUTBACKUP%\DEK-VG_2FidsProjective_offsetCal.txt  -u .\Results\UnitTest\ 



 
REM Get the current date and time in YYYY-MM-DD-HH-MM-SS format
SET THEDATE=%date:~10,4%-%date:~4,2%-%date:~7,2%-%time:~0,2%-%time:~3,2%-%time:~6,2%
REM Convert blanks to zeros...
set THEDATE=%THEDATE: =0%
mkdir %ROOTDIR%\%THEDATE%
xcopy /S /E /Y /Q %OUTPUTDIR%\* %ROOTDIR%\%THEDATE%\

REM Create run charts
REM Need to put the script in a better location
c:\python27\python.exe .\StitchPlotRegressionHistory.py