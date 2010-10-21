using System;
using MCyberStitch;

namespace CyberStitchTester
{
    class Program
    {
        /// <summary>
        /// Use SIM to load up an image set and run it through the stitcher....
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {
            ManagedMosaic mosaic = new ManagedMosaic(3, 3, 2);
            Console.WriteLine("This is a tester for CyberStitch...");
        }
    }
}
