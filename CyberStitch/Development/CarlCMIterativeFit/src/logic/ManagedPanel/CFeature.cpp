/*-------------------------------------------------------------------------------
         Copyright � 2010 CyberOptics Corporation.  All rights reserved.
---------------------------------------------------------------------------------
    THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY
    KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR
    PURPOSE.

Module Name:

    FeatureDescription.cpp

Abstract:

	FeatureDescription.cpp provides the implementation to setup the CFeature objects for
	2D SPI.

Environment:

    Managed C++ ('new' syntax)

---------------------------------------------------------------------------------
*/
#include "CFeature.h"

namespace Cyber
{
	namespace MPanel
	{
		CFeature::CFeature()
		{
			_pFeature = NULL;
		}

		CCross::CCross(int referenceID, double positionX, double positionY, double rotation,
			double sizeX, double sizeY, double legSizeX, double legSizeY )
		{
			_pFeature = new CrossFeature(referenceID, positionX, positionY, rotation,
										sizeX, sizeY, legSizeX, legSizeY );
		}

		CDiamond::CDiamond(int referenceID, double positionX, double positionY, double rotation,
			double sizeX, double sizeY ) 
		{
			_pFeature = new DiamondFeature(referenceID, positionX, positionY, rotation,
										sizeX, sizeY);
		}

		CDiamondFrame::CDiamondFrame(int referenceID, double positionX, double positionY, double rotation,
			double sizeX, double sizeY, double thickness ) 
		{
			_pFeature = new DiamondFrameFeature(referenceID, positionX, positionY, rotation,
										sizeX, sizeY, thickness);
		}

		CDisc::CDisc(int referenceID, double positionX, double positionY,
			double diameter ) 
		{
			_pFeature = new DiscFeature(referenceID, positionX, positionY, diameter);
		}

		CDonut::CDonut(int referenceID, double positionX, double positionY,
			double diameterInside, double diameterOutside ) 
		{
			_pFeature = new DonutFeature(referenceID, positionX, positionY,	diameterInside, diameterOutside);
		}

		CRectangle::CRectangle(int referenceID, double positionX, double positionY, double rotation,
			double sizeX, double sizeY, double sizeZ ) 
		{
			_pFeature = new RectangularFeature(referenceID, positionX, positionY, rotation,
											sizeX, sizeY, sizeZ);
		}

		CRectangle::CRectangle(int referenceID, double positionX, double positionY, double rotation,
			double sizeX, double sizeY) 
		{
			_pFeature = new RectangularFeature(referenceID, positionX, positionY, rotation,
											sizeX, sizeY);
		}

		CRectangleFrame::CRectangleFrame(int referenceID, double positionX, double positionY, double rotation,
			double sizeX, double sizeY, double thickness ) 
		{
			_pFeature = new RectangularFrameFeature(referenceID, positionX, positionY, rotation,
											sizeX, sizeY, thickness);
		}

		CTriangle::CTriangle(int referenceID, double positionX, double positionY, double rotation,
			double sizeX, double sizeY, double offsetX ) 
		{
			_pFeature = new TriangleFeature(referenceID, positionX, positionY, rotation,
											sizeX, sizeY, offsetX);
		}

		CEquilateralTriangleFrame::CEquilateralTriangleFrame(int referenceID, double positionX, double positionY, double rotation,
			double size, double thickness )
		{
			_pFeature = new EquilateralTriangleFrameFeature(referenceID, positionX, positionY, rotation,
											size, thickness);
		}

		CCheckerPattern::CCheckerPattern(int referenceID, double positionX, double positionY, double rotation,
			double sizeX, double sizeY ) 
		{
			_pFeature = new CheckerPatternFeature(referenceID, positionX, positionY, rotation,
											sizeX, sizeY);
		}

		CCyberSegment::CCyberSegment(bool line, bool penDown, bool clockwiseArc, double positionX,
					double positionY, double arcX, double arcY)
		{
			_pCyberSegment = new CyberSegment(line, penDown, clockwiseArc, 
											positionX, positionY, arcX, arcY);
		}

		CCyberShape::CCyberShape(int referenceID, double positionX, double positionY, double rotation) 
		{
			_pFeature = new CyberFeature(referenceID, positionX, positionY, rotation);
		}

		void CCyberShape::AddSegment(CCyberSegment^ segment)
		{
			CyberFeature* pCF = (CyberFeature*)_pFeature;
			pCF->AddSegment((CyberSegment*)(void*)segment->UnmanagedSegment);
		}

		CCyberSegment^ CCyberShape::GetFirstSegment()
		{
			if(_pFeature == NULL)
				return nullptr;

			CyberFeature *pCyberFeature = (CyberFeature*)_pFeature;
			CyberSegment *pCyberSegment = pCyberFeature->GetFirstSegment();
			if(pCyberSegment == NULL)
				return nullptr;
			
			return gcnew CCyberSegment(pCyberSegment);
		}

		CCyberSegment^ CCyberShape::GetNextSegment()
		{
			if(_pFeature == NULL)
				return nullptr;

			CyberFeature *pCyberFeature = (CyberFeature*)_pFeature;
			CyberSegment *pCyberSegment = pCyberFeature->GetNextSegment();
			if(pCyberSegment == NULL)
				return nullptr;
			
			return gcnew CCyberSegment(pCyberSegment);
		}

		int CCyberShape::GetNumberSegments()
		{
			if(_pFeature == NULL)
				return 0;

			CyberFeature *pCyberFeature = (CyberFeature*)_pFeature;

			return pCyberFeature->GetNumberSegments();
		}

	} // SPIAPI Namespace
} // Cyber Namespace