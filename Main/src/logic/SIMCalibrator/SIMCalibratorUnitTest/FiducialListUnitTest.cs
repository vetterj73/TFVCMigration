using Microsoft.VisualStudio.TestTools.UnitTesting;
using PanelAlignM;
using SIMCalibrator;

namespace SIMCalibratorUnitTest
{
    /// <summary>
    /// Testing the Fiducial List Functions... 
    /// Currently - these tests need to be run in 32 bit mode....
    /// </summary>
    [TestClass]
    public class FiducialListUnitTest
    {

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

        [TestMethod]
        public void TestEmptyList()
        {
            FiducialList fidList = new FiducialList();
            Assert.IsTrue(fidList.Count == 0);
            Assert.IsTrue(fidList.GetAverageXOffset() == 0.0);
            Assert.IsTrue(fidList.GetAverageYOffset() == 0.0);
            Assert.IsTrue(fidList.GetNominalToActualVelocityRatio() == 1.0);
            Assert.IsTrue(fidList.GetFidClosestToLeadingEdge() == null);
            Assert.IsTrue(fidList.GetFidFarthestFromLeadingEdge() == null);
            
            // Some basic tests for tolerances... tolerances are hardcoded for now.
            Assert.IsTrue(fidList.IsXInTolerance(.0004));
            Assert.IsTrue(fidList.IsYInTolerance(.0004));
            Assert.IsTrue(fidList.IsVelocityRatioInTolerance(1.001));
            Assert.IsFalse(fidList.IsXInTolerance(.006));
            Assert.IsFalse(fidList.IsYInTolerance(.006));
            Assert.IsFalse(fidList.IsVelocityRatioInTolerance(1.01));
        }

        [TestMethod]
        public void TestListOfOne()
        {
            FiducialList fidList = new FiducialList();

            // Testing that correlation scores need to be > .85
            ManagedFidInfo info = new ManagedFidInfo(.001, .001, 10.0, 10.0, .84, .3);
            fidList.Add(info);
            Assert.IsTrue(fidList.Count == 0);

            info = new ManagedFidInfo(.001, .001, 10.0, 10.0, .85, .3);
            fidList.Add(info);
            Assert.IsTrue(fidList.Count == 1);

            Assert.IsTrue(fidList.GetAverageXOffset() == info.GetXOffsetInMeters());
            Assert.IsTrue(fidList.GetAverageYOffset() == info.GetYOffsetInMeters());
            Assert.IsTrue(fidList.GetNominalToActualVelocityRatio() == 1.0);
            Assert.IsTrue(fidList.GetFidFarthestFromLeadingEdge() != null);
            Assert.IsTrue(fidList.GetFidClosestToLeadingEdge() != null);
            Assert.IsTrue(fidList.GetFidClosestToLeadingEdge() == fidList.GetFidFarthestFromLeadingEdge());
        }

        [TestMethod]
        public void TestListOfTwoTooCloseForSpeed()
        {
            FiducialList fidList = new FiducialList();

            ManagedFidInfo info1 = new ManagedFidInfo(.001, .001, 0.0, 0.0, .86, .3);
            ManagedFidInfo info2 = new ManagedFidInfo(.009, .001, 100.0, 100.0, .93, .3);
            fidList.Add(info1);
            fidList.Add(info2);
            Assert.IsTrue(fidList.Count == 2);

            Assert.IsTrue(fidList.GetAverageXOffset() == info2.GetXOffsetInMeters()/ 2.0);
            Assert.IsTrue(fidList.GetAverageYOffset() == info2.GetYOffsetInMeters()/2.0);

            // This is an indication that speed will not be calculated.
            Assert.IsTrue(fidList.GetNominalToActualVelocityRatio() == 1.0);
            Assert.IsTrue(fidList.GetFidFarthestFromLeadingEdge() != null);
            Assert.IsTrue(fidList.GetFidClosestToLeadingEdge() != null);
            Assert.IsTrue(fidList.GetFidClosestToLeadingEdge().GetNominalXPositionInMeters() <
                fidList.GetFidFarthestFromLeadingEdge().GetNominalXPositionInMeters());
        }

        [TestMethod]
        public void TestListOfTwoFarEnoughForSpeed()
        {
            FiducialList fidList = new FiducialList();

            ManagedFidInfo info2 = new ManagedFidInfo(.011, .001, 100.0, 100.0, .93, .2);
            ManagedFidInfo info1 = new ManagedFidInfo(.001, .001, 0.0, 0.0, .86, .2);
            fidList.Add(info1);
            fidList.Add(info2);
            Assert.IsTrue(fidList.Count == 2);
 
            // This is an indication that speed will be calculated.
            Assert.IsTrue(fidList.GetNominalToActualVelocityRatio() != 1.0);
            Assert.IsTrue(fidList.GetFidFarthestFromLeadingEdge() != null);
            Assert.IsTrue(fidList.GetFidClosestToLeadingEdge() != null);
            Assert.IsTrue(fidList.GetFidClosestToLeadingEdge().GetNominalXPositionInMeters() <
                fidList.GetFidFarthestFromLeadingEdge().GetNominalXPositionInMeters());
        }
    }
}
