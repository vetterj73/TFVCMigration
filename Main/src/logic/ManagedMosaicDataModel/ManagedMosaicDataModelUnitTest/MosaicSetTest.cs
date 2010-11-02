using MMosaicDM;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace ManagedCyberStitchUnitTest
{
    /// <summary>
    /// Summary description for UnitTest1
    /// </summary>
    [TestClass]
    public class MosaicSetTest
    {
        private TestContext _testContextInstance;

        /// <summary>
        ///Gets or sets the test context which provides
        ///information about and functionality for the current test run.
        ///</summary>
        public TestContext TestContext
        {
            get
            {
                return _testContextInstance;
            }
            set
            {
                _testContextInstance = value;
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
        public void TestCalFlags()
        {
            ManagedMosaicSet mSet = new ManagedMosaicSet
                (.2, .25, 3, .003, 4, .004, 2592, 1944, 2592, .000017, .000017);

            Assert.IsTrue(mSet.GetCorrelationSet(0, 0) == null);
            Assert.IsTrue(mSet.GetCorrelationSet(1, 1) == null);
            
            mSet.AddLayer(.003);
            Assert.IsTrue(mSet.GetCorrelationSet(0, 0) != null);
            Assert.IsTrue(mSet.GetCorrelationSet(1, 1) == null);

            ManagedCorrelationFlags mcf = mSet.GetCorrelationSet(0, 0);
            Assert.IsTrue(mcf.GetCameraToCamera());
            Assert.IsTrue(mcf.GetTriggerToTrigger());
            mcf.SetCameraToCamera(false);

            mcf = mSet.GetCorrelationSet(0, 0);
            Assert.IsTrue(mcf.GetCameraToCamera()==false);
            Assert.IsTrue(mcf.GetTriggerToTrigger());
            mcf.SetCameraToCamera(true);
            mcf.SetTriggerToTrigger(false);

            mcf = mSet.GetCorrelationSet(0, 0);
            Assert.IsTrue(mcf.GetCameraToCamera());
            Assert.IsTrue(mcf.GetTriggerToTrigger() == false);

            mSet.AddLayer(.006);

            /// CorrelationFlags 0,1 is same as 1,0
            mcf = mSet.GetCorrelationSet(0, 1);
            Assert.IsTrue(mcf.GetCameraToCamera());
            Assert.IsTrue(mcf.GetTriggerToTrigger());
            mcf.SetCameraToCamera(false);
            mcf = mSet.GetCorrelationSet(1, 0);
            Assert.IsTrue(mcf.GetCameraToCamera()==false);
            Assert.IsTrue(mcf.GetTriggerToTrigger());

            mSet.AddLayer(.015);
            mcf = mSet.GetCorrelationSet(1, 2);
            Assert.IsTrue(mcf.GetCameraToCamera());
            Assert.IsTrue(mcf.GetTriggerToTrigger());

        }

        [TestMethod]
        public void BasicMosaicSetTest()
        {
            ManagedMosaicSet mSet = new ManagedMosaicSet
                (.2, .25, 3, .003, 4, .004, 2592, 1944, 2592, .000017, .000017);
            Assert.IsTrue(mSet.GetLayer(0) == null);
            Assert.IsTrue(mSet.GetLayer(1) == null);
            mSet.AddLayer(3.0);
            Assert.IsTrue(mSet.GetLayer(0) != null);
            Assert.IsTrue(mSet.GetLayer(1) == null);
            mSet.AddLayer(6.0);
            Assert.IsTrue(mSet.GetLayer(0) != null);
            Assert.IsTrue(mSet.GetLayer(1) != null);

            for (int i = 0; i < 1; i++)
            {
                Assert.IsTrue(mSet.GetLayer(i).GetTile(0, 0) != null);
                Assert.IsTrue(mSet.GetLayer(i).GetTile(1, 1) != null);
                Assert.IsTrue(mSet.GetLayer(i).GetTile(2, 3) != null);

                Assert.IsTrue(mSet.GetLayer(i).GetTile(3, 3) == null);
                Assert.IsTrue(mSet.GetLayer(i).GetTile(2, 4) == null);
            }

        }
    }
}
