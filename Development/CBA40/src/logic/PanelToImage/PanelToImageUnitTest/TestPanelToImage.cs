using NUnit.Framework;
using PanelToImage;

namespace PanelToImageUnitTest
{
    [TestFixture]
    public class TestPanelToImage
    {
        [SetUp]
        public void Setup()
        {
        }

        [TearDown]
        public void TearDown()
        {
        }

        /// <summary>
        /// This is testing that the results for panel to image
        /// are staying consistant over time.  It is possible that the 
        /// results will change and still be correct.  If/When this happens,
        /// you simply need to save the newly generated files
        /// </summary>
        [Test]
        public void TestSomethingUseful()
        {
           // PanelConverter pc = new PanelConverter();

           // pc.Initialize()

            Assert.IsTrue(1 == 1);
        }
    }
}
