using System;
using System.Drawing.Imaging;
using Cyber.ImageUtils;
using Cyber.MPanel;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CPanelIOUnitTest
{
    /// <summary>
    /// Summary description for UnitTest1
    /// </summary>
    [TestClass]
    public class CPanelIOTests
    {
        public CPanelIOTests()
        {
            //
            // TODO: Add constructor logic here
            //
        }

        private TestContext testContextInstance;

        /// <summary>
        ///Gets or sets the test context which provides
        ///information about and functionality for the current test run.
        ///</summary>
        public TestContext TestContext
        {
            get
            {
                return testContextInstance;
            }
            set
            {
                testContextInstance = value;
            }
        }

        #region Additional test attributes
        //
        // You can use the following additional attributes as you write your tests:
        //
        // Use ClassInitialize to run code before running the first test in the class
        // [ClassInitialize()]
        // public static void MyClassInitialize(TestContext testContext) { }
        //
        // Use ClassCleanup to run code after all tests in a class have run
        // [ClassCleanup()]
        // public static void MyClassCleanup() { }
        //
        // Use TestInitialize to run code before running each test 
        // [TestInitialize()]
        // public void MyTestInitialize() { }
        //
        // Use TestCleanup to run code after each test has run
        // [TestCleanup()]
        // public void MyTestCleanup() { }
        //
        #endregion
        private const double cPixelSizeInMeters = 1.69e-5;

        [TestMethod]
        public void TestXmlConvert()
        {
            CPanel panel = MPanelIO.XmlToPanel.CSIMPanelXmlToCPanel(
                "..\\..\\..\\..\\..\\testdata\\5-540-206-00 Rev B1-T1_OffsetApplied.xml",
                cPixelSizeInMeters, cPixelSizeInMeters);

            Assert.IsTrue(panel.GetNumPixelsInX() == 14165);
            Assert.IsTrue(panel.GetNumPixelsInY() == 12024);

            Assert.IsTrue(panel.NumberOfFeatures == 329);
            Assert.IsTrue(panel.NumberOfFiducials == 2);

            // Write out a cad image for manual verification
            ImageSaver.SaveToFile(panel.GetNumPixelsInY(), panel.GetNumPixelsInX(),
                                  panel.GetNumPixelsInY(), panel.GetCADBuffer(),
                                  "..\\..\\..\\..\\..\\testdata\\xmlTest.png",
                                  PixelFormat.Format8bppIndexed,
                                  ImageFormat.Png);

            IntPtr test = panel.GetAperatureBuffer();
/*            ImageSaver.SaveToFile(panel.GetNumPixelsInY(), panel.GetNumPixelsInX(),
                                  panel.GetNumPixelsInY(), panel.GetAperatureBuffer(),
                                  "..\\..\\..\\..\\..\\testdata\\xmlTest16.Tif",
                                  PixelFormat.Format16bppGrayScale,
                                  ImageFormat.Tiff);
            */
            // Write out a cad image for manual verification
        }

        [TestMethod]
        public void TestSrfConvert()
        {
            CPanel panel = MPanelIO.SRFToPanel.parseSRF(
                "..\\..\\..\\..\\..\\testdata\\DEK_VG.srf",
                cPixelSizeInMeters, cPixelSizeInMeters);

            int pixX = panel.GetNumPixelsInX();
            int pixY = panel.GetNumPixelsInY();
            Assert.IsTrue(panel.GetNumPixelsInX() == 12030);
            Assert.IsTrue(panel.GetNumPixelsInY() == 7509);

            Assert.IsTrue(panel.NumberOfFeatures == 3785);
            Assert.IsTrue(panel.NumberOfFiducials == 4);

            // Can't save it, but I can create an image in memory and see if it crashes...
            // Write out a cad image for manual verification
            ImageSaver.SaveToFile(panel.GetNumPixelsInY(), panel.GetNumPixelsInX(),
                                  panel.GetNumPixelsInY(), panel.GetCADBuffer(),
                                  "..\\..\\..\\..\\..\\testdata\\srfTest.png",
                                  PixelFormat.Format8bppIndexed,
                                  ImageFormat.Png);
        }
    }
}
