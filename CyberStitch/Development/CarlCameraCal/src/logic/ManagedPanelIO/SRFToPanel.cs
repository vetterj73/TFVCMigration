using System.Collections;
using System.Drawing;
using SRF;
using CCross=Cyber.MPanel.CCross;
using CDiamond = Cyber.MPanel.CDiamond;
using CDisc = Cyber.MPanel.CDisc;
using CDonut = Cyber.MPanel.CDonut;
using CPanel = Cyber.MPanel.CPanel;
using CRectangle = Cyber.MPanel.CRectangle;
using CTriangle = Cyber.MPanel.CTriangle;

namespace MPanelIO
{
    public class SRFToPanel
    {
        enum FeatureType
        {
            FIDUCIAL,
            PAD,
        }

        public static CPanel parseSRF(string path, double pixelSizeX, double pixelSizeY)
        {
            SRFUtilities srfUtilities = new SRFUtilities();

            if (srfUtilities.Read(path) == false)
            {
                return null;
            }
            PointF srfPanelSize = srfUtilities.GetPanelSize(DistanceType.Meters);
            CPanel panel = new CPanel(srfPanelSize.X, srfPanelSize.Y, pixelSizeX, pixelSizeY);

            Hashtable fiducials = new Hashtable();
            srfUtilities.GetAllImageFiducials(ref fiducials, AngleType.Degrees, DistanceType.Meters);

            foreach (int fidId in fiducials.Keys)
            {
                AddFeatures(panel, FeatureType.FIDUCIAL, fidId, fiducials);
            }

            Hashtable features = new Hashtable();
            srfUtilities.GetAllFeatures(ref features, AngleType.Degrees, DistanceType.Meters);

            foreach (int featureId in features.Keys)
            {
                AddFeatures(panel, FeatureType.PAD, featureId, features);
            }

            return panel;
        }

        private static void AddFeatures(CPanel panel, FeatureType type, int id, IDictionary features)
        {
            if (features[id].GetType() == typeof(SRFUtilities.CrossFeatureInfo))
            {
                SRFUtilities.CrossFeatureInfo shape = (SRFUtilities.CrossFeatureInfo)features[id];


                CCross cross = new CCross(id, shape.X, shape.Y, shape.Theta,
                                          shape.shapeInfo.Base, shape.shapeInfo.Height,
                                          shape.shapeInfo.BaseLegWidth, shape.shapeInfo.HeightLegWidth);
                if (type == FeatureType.PAD)
                    panel.AddFeature(cross);
                else
                    panel.AddFiducial(cross);
            }
            else if (features[id].GetType() == typeof(SRFUtilities.DiamondFeatureInfo))
            {
                SRFUtilities.DiamondFeatureInfo shape = (SRFUtilities.DiamondFeatureInfo)features[id];

                CDiamond diamond = new CDiamond(id, shape.X, shape.Y, shape.Theta,
                                                shape.shapeInfo.Base, shape.shapeInfo.Height);

                if (type == FeatureType.PAD)
                    panel.AddFeature(diamond);
                else
                    panel.AddFiducial(diamond);
            }
            else if (features[id].GetType() == typeof(SRFUtilities.DiscFeatureInfo))
            {
                SRFUtilities.DiscFeatureInfo discInfo = (SRFUtilities.DiscFeatureInfo)features[id];

                CDisc disc = new CDisc(id, discInfo.X, discInfo.Y, discInfo.shapeInfo.Diameter);

                if (type == FeatureType.PAD)
                    panel.AddFeature(disc);
                else
                    panel.AddFiducial(disc);
            }
            else if (features[id].GetType() == typeof(SRFUtilities.DonutFeatureInfo))
            {
                SRFUtilities.DonutFeatureInfo shape = (SRFUtilities.DonutFeatureInfo)features[id];

                CDonut donut = new CDonut(id, shape.X, shape.Y, shape.shapeInfo.InnerDiameter, shape.shapeInfo.OuterDiameter);

                if (type == FeatureType.PAD)
                    panel.AddFeature(donut);
                else
                    panel.AddFiducial(donut);
            }
            else if (features[id].GetType() == typeof(SRFUtilities.RectangleFeatureInfo))
            {
                SRFUtilities.RectangleFeatureInfo rectInfo = (SRFUtilities.RectangleFeatureInfo)features[id];

                CRectangle rectangle = new CRectangle(id, rectInfo.X, rectInfo.Y, rectInfo.Theta,
                                                      rectInfo.shapeInfo.Base, rectInfo.shapeInfo.Height);
                if (type == FeatureType.PAD)
                    panel.AddFeature(rectangle);
                else
                    panel.AddFiducial(rectangle);
            }
            else if (features[id].GetType() == typeof(SRFUtilities.TriangleFeatureInfo))
            {
                SRFUtilities.TriangleFeatureInfo shape = (SRFUtilities.TriangleFeatureInfo)features[id];

                CTriangle triangle = new CTriangle(id, shape.X, shape.Y, shape.Theta,
                                                   shape.shapeInfo.Base, shape.shapeInfo.Height, shape.shapeInfo.Offset);

                if (type == FeatureType.PAD)
                    panel.AddFeature(triangle);
                else
                    panel.AddFiducial(triangle);
            }
            else if (features[id].GetType() == typeof(SRFUtilities.VendorShapeFeatureInfo))
            {
                SRFUtilities.VendorShapeFeatureInfo shape = (SRFUtilities.VendorShapeFeatureInfo)features[id];

                Cyber.MPanel.CCyberShape cyberShape = new Cyber.MPanel.CCyberShape(id, shape.X, shape.Y, shape.Theta);

                //string filename = "c:\\2D_SPI\\LogFiles\\FrmMain-CyberShapes.csv";
                //bool fileExists = File.Exists(filename);

                //StreamWriter output = new StreamWriter(filename, true);

                //if (!fileExists)
                //    output.WriteLine("Shape Name,Shape ID,Segment ID,Line,PenDown,ClockwiseArc,PositionX,PositionY,ArcX,ArcY");

                foreach (SRF.CCyberSegment segment in shape.shapeInfo.CyberSegments)
                {
                    //output.WriteLine("{0},{1},{2},{3},{4},{5},{6:0.######},{7:0.######},{8:0.######},{9:0.######}",
                    //                 shape.shapeInfo.CyberShapeName, shape.shapeInfo.CyberShapeId, segment.CyberSegmentId, segment.Line, segment.PenDown,
                    //                 segment.ClockwiseArc, segment.PositionX, segment.PositionY,
                    //                 segment.ArcX, segment.ArcY);

                    Cyber.MPanel.CCyberSegment cCyberSegment = new Cyber.MPanel.CCyberSegment(segment.Line,
                        segment.PenDown, segment.ClockwiseArc, segment.PositionX, segment.PositionY,
                        segment.ArcX, segment.ArcY);

                    cyberShape.AddSegment(cCyberSegment);
                }
                //output.Close();

                if (type == FeatureType.PAD)
                    panel.AddFeature(cyberShape);
                else
                    panel.AddFiducial(cyberShape);
            }
        }

    }
}
