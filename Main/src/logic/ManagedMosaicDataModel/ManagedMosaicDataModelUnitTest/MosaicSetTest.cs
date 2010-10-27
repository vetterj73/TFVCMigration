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
        public void BasicMosaicSetTest()
        {
            ManagedMosaicSet mSet = new ManagedMosaicSet();
            mSet.Initialize(3, 4, 1000, 2000, 1000, 1, 4);
            mSet.AddLayer(3.0);
            Assert.IsTrue(mSet.GetLayer(0) != null);
            Assert.IsTrue(mSet.GetLayer(1) == null);
            mSet.AddLayer(6.0);
            Assert.IsTrue(mSet.GetLayer(0) != null);
            Assert.IsTrue(mSet.GetLayer(1) != null);
            mSet.Reset();
            Assert.IsTrue(mSet.GetLayer(0) == null);
            Assert.IsTrue(mSet.GetLayer(1) == null);
        }
    }
}
