using Cyber.SPIAPI;
using NUnit.Framework;

namespace PanelToImageUnitTest
{
    [TestFixture]
    public class PanelTest
    {
        /// <summary>
        /// Verify that add/delete is robust and speedy.
        /// </summary>
        [Test]
        public void AddDeleteTest()
        {
            const int cNumFeatures = 1000;
            CPanel panel = new CPanel(10.0f, 10.0f);
            Assert.IsTrue(panel.NumberOfFeatures == 0);

            for (int i = 0; i < cNumFeatures; i++)
            {
                CCross cross = new CCross(i, 5.00, 5.0, 0, 10.0, 10.0, 3, 4);
                Assert.IsTrue(panel.AddFeature(cross) == SPISTATUS.SPISTATUS_SUCCESS);
            }

            Assert.IsTrue(panel.NumberOfFeatures == cNumFeatures);

            for (int i = 0; i < cNumFeatures; i++)
            {
                panel.RemoveFeature(i);
            }
            Assert.IsTrue(panel.NumberOfFeatures == 0);
        }

        /// <summary>
        /// Verify that aperature value is decreasing as you add items.
        /// </summary>
        [Test]
        public void AperatureTest()
        {
            CPanel panel = new CPanel(10.0f, 10.0f);
            panel.AddFeature(new CCross(1, 5.00, 5.0, 0, 10.0, 10.0, 3, 4));
            panel.AddFeature(new CCross(2, 5.00, 5.0, 0, 10.0, 10.0, 3, 4));

            CFeature feature = panel.GetFirstFeature();
            uint test = feature.AperatureValue;
            feature = panel.GetNextFeature();

            Assert.IsTrue(test == feature.AperatureValue + 1);
        }

        /// <summary>
        /// can add features that are out of bounds.
        /// </summary>
        [Test]
        public void BasicPanelAddFeatureOutOfBounds()
        {
            CPanel panel = new CPanel(10.0f, 10.0f);
            Assert.IsTrue(panel.NumberOfFeatures == 0);
            CCross cross = new CCross(1234, 10.001, 10.0, 0, 10.0, 10.0, 3, 4);
            Assert.IsTrue(panel.AddFeature(cross) == SPISTATUS.SPISTATUS_FEATURE_OUT_OF_BOUNDS);
            Assert.IsTrue(panel.NumberOfFeatures == 0);
        }

        /// <summary>
        /// Checking that pads must be fully INSIDE the panel.
        /// </summary>
        [Test]
        public void BasicPanelAddFeaturePartialOutOfBounds()
        {
            CPanel panel = new CPanel(10.0f, 10.0f);
            Assert.IsTrue(panel.NumberOfFeatures == 0);

            // this should work, the rest will not...
            CRectangle rect = new CRectangle(1, 5.0, 5.0, 0, 10.0, 10.0);
            Assert.IsTrue(panel.AddFeature(rect) == SPISTATUS.SPISTATUS_SUCCESS);
            Assert.IsTrue(panel.NumberOfFeatures == 1);

            rect = new CRectangle(2, 5.0, 5.1, 0, 10.0, 10.0);
            Assert.IsTrue(panel.AddFeature(rect) == SPISTATUS.SPISTATUS_FEATURE_OUT_OF_BOUNDS);
            Assert.IsTrue(panel.NumberOfFeatures == 1);

            rect = new CRectangle(2, 5.1, 5.0, 0, 10.0, 10.0);
            Assert.IsTrue(panel.AddFeature(rect) == SPISTATUS.SPISTATUS_FEATURE_OUT_OF_BOUNDS);
            Assert.IsTrue(panel.NumberOfFeatures == 1);

            rect = new CRectangle(2, 4.9, 5.0, 0, 10.0, 10.0);
            Assert.IsTrue(panel.AddFeature(rect) == SPISTATUS.SPISTATUS_FEATURE_OUT_OF_BOUNDS);
            Assert.IsTrue(panel.NumberOfFeatures == 1);

            rect = new CRectangle(2, 5.0, 4.9, 0, 10.0, 10.0);
            Assert.IsTrue(panel.AddFeature(rect) == SPISTATUS.SPISTATUS_FEATURE_OUT_OF_BOUNDS);
            Assert.IsTrue(panel.NumberOfFeatures == 1);
        }

        /// <summary>
        /// You cannot have 2 features with the same id!
        /// </summary>
        [Test]
        public void BasicPanelAddFeatureWithSameID()
        {
            CPanel panel = new CPanel(20.0f, 30.0f);
            Assert.IsTrue(panel.NumberOfFeatures == 0);

            CCross cross = new CCross(1234, 5.0, 5.0, 0, 10.0, 10.0, 3, 4);
            CCross cross2 = new CCross(1234, 5.0, 5.0, 0, 10.0, 10.0, 3, 4);
            Assert.IsTrue(panel.AddFeature(cross) == SPISTATUS.SPISTATUS_SUCCESS);
            Assert.IsTrue(panel.AddFeature(cross2) == SPISTATUS.SPISTATUS_FEATURE_ID_IN_USE);
            Assert.IsTrue(panel.NumberOfFeatures == 1);
            panel.RemoveFeature(1234);
            Assert.IsTrue(panel.NumberOfFeatures == 0);
        }

        /// <summary>
        /// Basic adding and removing of features.
        /// </summary>
        [Test]
        public void BasicPanelAddingTests()
        {
            CPanel panel = new CPanel(20.0f, 30.0f);
            Assert.IsTrue(panel.NumberOfFeatures == 0);
            Assert.IsTrue(panel.NumberOfFiducials == 0);

            CCross cross = new CCross(1234, 5.0, 5.0, 0, 10.0, 10.0, 3, 4);

            panel.AddFeature(cross);
            Assert.IsTrue(panel.NumberOfFeatures == 1);
            Assert.IsTrue(panel.NumberOfFiducials == 0);

            panel.RemoveFeature(1234);
            Assert.IsTrue(panel.NumberOfFeatures == 0);

            panel.RemoveFeature(1234);
            panel.RemoveFiducial(1234);

            Assert.IsTrue(panel.NumberOfFeatures == 0);
            Assert.IsTrue(panel.NumberOfFiducials == 0);
        }

        /// <summary>
        /// Currently, 2 features can overlap... this may be something to revisit...
        /// </summary>
        [Test]
        public void BasicPanelFeatureOverlap()
        {
            CPanel panel = new CPanel(10.0f, 10.0f);
            Assert.IsTrue(panel.NumberOfFeatures == 0);
            CCross cross = new CCross(1234, 5.00, 5.0, 0, 10.0, 10.0, 3, 4);
            CCross cross2 = new CCross(1235, 5.00, 5.0, 0, 10.0, 10.0, 3, 4);
            panel.AddFeature(cross);
            panel.AddFeature(cross2);
            Assert.IsTrue(panel.NumberOfFeatures == 2);
        }

        /// <summary>
        /// Verify that we can iteratate through lists without infinite loops, etc...
        /// </summary>
        [Test]
        public void InterationTest()
        {
            CPanel panel = new CPanel(10.0f, 10.0f);
            Assert.IsTrue(panel.GetNextFeature() == null);

            panel.AddFeature(new CCross(1, 5.00, 5.0, 0, 10.0, 10.0, 3, 4));

            CFeature feature = panel.GetFirstFeature();
            Assert.IsTrue(feature.ReferenceID == 1);
            Assert.IsTrue(panel.GetNextFeature() == null);

            panel.AddFeature(new CCross(2, 5.00, 5.0, 0, 10.0, 10.0, 3, 4));

            feature = panel.GetFirstFeature();
            Assert.IsTrue(feature.ReferenceID == 1);
            feature = panel.GetNextFeature();
            Assert.IsTrue(feature.ReferenceID == 2);
            Assert.IsTrue(panel.GetNextFeature() == null);
        }
    }
}