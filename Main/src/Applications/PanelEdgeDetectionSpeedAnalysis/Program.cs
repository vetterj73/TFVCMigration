using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;

namespace PanelEdgeDetectionSpeedAnalysis
{
    class Program
    {
        // Names of process stage
        private static string[] _sStageName = new string[]{
            "prepare image",
            "smooth",
            "edge detection",
            "dilation",
            "Hough"};


        static void Main(string[] args)
        {
            // Input
            string logFile = "";
            string outputFile = "";
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "-l" && i < args.Length - 1)
                {
                    logFile = args[i + 1];
                }
                else if (args[i] == "-o" && i < args.Length - 1)
                {
                    outputFile = args[i + 1];
                }
            }

            // Input validation check
            if (0 == outputFile.Length)
            {
                Console.WriteLine("Illegal output file!");
                return;
            }
  
            if (!File.Exists(logFile))
            {
                Console.WriteLine("Illegal log file!");
                return;
            }

            // Create key word string
            int iStageNum = _sStageName.Length;
            string[] sStageStatus = new string[iStageNum * 2];
            for (int i = 0; i < _sStageName.Length; i++)
            {
                sStageStatus[i * 2] = "Begin " + _sStageName[i];
                sStageStatus[i * 2 + 1] = "End " + _sStageName[i];
            }

            // Get production name and all lines with key words
            string[] lines = File.ReadAllLines(logFile);
            List<string> sKeyWordList = new List<string>();
            foreach (string line in lines)
            {
                // Collect all lines with key word 2
                foreach (string s in sStageStatus)
                {
                    if (line.Contains(s))
                    {
                        sKeyWordList.Add(line);
                        break;
                    }
                }
            }

            // If there is no key word found
            if (sKeyWordList.Count == 0)
            {
                Console.WriteLine(logFile + " Has no key word");
                return;
            }

            // Create time table
            try
            {
                TimeTableForLog(outputFile, sKeyWordList);
            }
            catch (Exception e)
            {
                Console.WriteLine("There is an exception: {0}", e);
            }

        }


        private static double GetTimeInSecond(string s)
        {
            string[] sTime = s.Split(new char[] { ' ', '.', ':' });
            double dTime = Convert.ToInt32(sTime[0]) * 60 * 60; // in Hours
            dTime += Convert.ToInt32(sTime[1]) * 60;
            dTime += Convert.ToInt32(sTime[2]);
            dTime += (double)Convert.ToInt64(sTime[3]) / 1e3;

            return (dTime);
        }

        private static void TimeTableForLog(
            string outputFile,
            List<string> sKeyWordList)
        {
            int iPanelCount = 0;
            int iStageNum = _sStageName.Length;
            double[] dBeginTimes = new double[iStageNum];
            double[] dEndTimes = new double[iStageNum];
            double[] dSumTimes = new double[iStageNum];
            bool[] bValid = new bool[iStageNum];
            for (int i = 0; i < iStageNum; i++)
            {
                dBeginTimes[i] = -1;
                dEndTimes[i] = -1;
                dSumTimes[i] = 0;
                bValid[i] = false;
            }

            StreamWriter writer = new StreamWriter(outputFile);

            string sLine = "";
            for (int i = 0; i < iStageNum; i++)
                sLine += "," + _sStageName[i];
            writer.WriteLine(sLine);

            foreach (string s in sKeyWordList)
            {
                for (int i = 0; i < iStageNum; i++)
                {
                    if (s.Contains(_sStageName[i]))
                    {
                        if (s.Contains("Begin"))
                        {
                            dBeginTimes[i] = GetTimeInSecond(s);
                            bValid[i] = false;
                        }
                        else if (s.Contains("End"))
                        {
                            dEndTimes[i] = GetTimeInSecond(s);
                            bValid[i] = true;
                        }
                    }                    
                }

                bool bRecordLine = true;
                for (int i = 0; i < iStageNum; i++)
                {
                    if (bValid[i] == false)
                    {
                        bRecordLine = false;
                        break;
                    }
                }

                if (bRecordLine)
                {
                    iPanelCount++;

                    sLine = "#" + iPanelCount +",";
                    for (int i = 0; i < iStageNum; i++)
                    {
                        double dTime = dEndTimes[i] - dBeginTimes[i];
                        dSumTimes[i] += dTime;
                        sLine += dTime.ToString() + ',';
                    }
                    writer.WriteLine(sLine);
                }
            }

            sLine = "Sum,";
            for (int i = 0; i < iStageNum; i++)
            {
                sLine += (dSumTimes[i]/iPanelCount).ToString() + ',';
            }
            writer.WriteLine(sLine);
            writer.Close();
        }
    }
}
