#include "OverlapManager.h"

OverlapManager::OverlapManager(
	MosaicImage* pMosaics, 
	CorrelationFlags** pFlags, 
	unsigned int iNumIllumination,
	Image* pCadImg, 
	DRect validRect)
{	
	_iMinOverlapSize = 100;
	
	_pMosaics = pMosaics;	
	_pFlags = pFlags;	
	_iNumIllumination = iNumIllumination;
	_pCadImg = pCadImg;
	_validRect = validRect;

	unsigned int i, j;
	_iNumImgX=0;
	_iNumImgY=0;
	for(i=0; i<_iNumIllumination; i++)
	{
		if (_iNumImgX < pMosaics[i].NumImInX())
			_iNumImgX = pMosaics[i].NumImInX();

		if (_iNumImgY < pMosaics[i].NumImInY())
			_iNumImgY = pMosaics[i].NumImInY();
	}

	// Create 3D arrays for storage of overlaps
	_fovFovOverlapLists = new list<FovFovOverlap>**[_iNumIllumination];
	_cadFovOverlapLists = new list<CadFovOverlap>**[_iNumIllumination];
	_fidFovOverlapLists = new list<FidFovOverlap>**[_iNumIllumination];
	for(i=0; i<_iNumIllumination; i++)
	{
		_fovFovOverlapLists[i] = new list<FovFovOverlap>*[_iNumImgY];
		_cadFovOverlapLists[i] = new list<CadFovOverlap>*[_iNumImgY];
		_fidFovOverlapLists[i] = new list<FidFovOverlap>*[_iNumImgY];

		for(j=0; j<_iNumImgY; j++)
		{
			_fovFovOverlapLists[i][j] = new list<FovFovOverlap>[_iNumImgX];
			_cadFovOverlapLists[i][j] = new list<CadFovOverlap>[_iNumImgX];
			_fidFovOverlapLists[i][j] = new list<FidFovOverlap>[_iNumImgX];
		}
	}

	CreateFovFovOverlaps();
	CreateCadFovOverlaps();
}


OverlapManager::~OverlapManager(void)
{
	// Release 3D arrays for storage
	unsigned int i, j;
	for(i=0; i<_iNumIllumination; i++)
	{
		for(j=0; j<_iNumImgY; j++)
		{
			delete [] _fovFovOverlapLists[i][j];
			delete [] _cadFovOverlapLists[i][j];
			delete [] _fidFovOverlapLists[i][j];
		}

		delete [] _fovFovOverlapLists[i];
		delete [] _cadFovOverlapLists[i];
		delete [] _fidFovOverlapLists[i];
	}
	delete [] _fovFovOverlapLists;
	delete [] _cadFovOverlapLists;
	delete [] _fidFovOverlapLists;
}

// Create Fov overlap for two illuminations
bool OverlapManager::CreateFovFovOverlapsForTwoIllum(unsigned int iIndex1, unsigned int iIndex2)
{
	CorrelationFlags flags = _pFlags[iIndex1][iIndex2];
	bool bColCol = flags.GetCameraToCamera();
	bool bRowRow = flags.GetTriggerToTrigger();
	bool bMask = flags.GetMaskNeeded();
	bool bCad	= false; //need modify

	unsigned int iNumX1 = _pMosaics[iIndex1].NumImInX();
	unsigned int iNumY1 = _pMosaics[iIndex1].NumImInX();
	double* pdCenX1 = new double[iNumX1];
	double* pdCenY1 = new double[iNumY1];
	_pMosaics[iIndex1].ImageLineCentersX(pdCenX1);
	_pMosaics[iIndex1].ImageLineCentersY(pdCenY1);
	double dSpanX = (pdCenX1[iNumX1-1] - pdCenX1[0])/(iNumX1-1);
	double dSpanY = (pdCenY1[iNumY1-1] - pdCenY1[0])/(iNumY1-1);

	unsigned int iNumX2 = _pMosaics[iIndex2].NumImInX();
	unsigned int iNumY2 = _pMosaics[iIndex2].NumImInX();
	double* pdCenX2 = new double[iNumX2];
	double* pdCenY2 = new double[iNumY2];
	_pMosaics[iIndex2].ImageLineCentersX(pdCenX2);
	_pMosaics[iIndex2].ImageLineCentersY(pdCenY2);

	unsigned int ix, iy, kx, ky;
	for(iy = 0; iy<iNumY1; iy++)
	{
		for(ix = 0; ix<iNumX1; ix++)
		{
			// Col by col overlap
			if(bColCol)
			{
				int iPosX1=-1, iPosX2=-1, iPosY=-1;				
				// The nearest horizontal line1
				for(ky=0; ky<iNumY2; ky++)
				{	
					double dis = fabs(pdCenY1[iy] -pdCenY2[ky]);
					double dMinDis = 2*dSpanY;
					if(dMinDis > dis)
					{
						dMinDis = dis;
						iPosY = ky;
					}
				}
				// The nearest left image in horizontal line with a distance bigger than 0.5 span
				for(kx = 0; kx<iNumX2; kx++)
				{		
					double dis = pdCenX1[ix] -pdCenX2[kx];
					if(dis>dSpanX*0.5)
						iPosX1 = kx;
				}
				// The nearest rightr image in horizontal line with a distance bigger than 0.5 span
				for(kx = iNumX2-1; kx>=0; kx++)
				{		
					double dis = pdCenX2[kx]-pdCenX1[ix];
					if(dis>dSpanX*0.5)
						iPosX2 = kx;
				}

				// Create left overlap and add to list
				if(iPosX1>0 && iPosY>0)
				{
					FovFovOverlap overlap(
						&_pMosaics[iIndex1], &_pMosaics[iIndex2],
						pair<unsigned int, unsigned int>(ix, iy),
						pair<unsigned int, unsigned int>(iPosX1, iPosY),
						_validRect, bMask);

					if(overlap.Columns()>_iMinOverlapSize || overlap.Rows()>_iMinOverlapSize)
					{
						_fovFovOverlapLists[iIndex1][iy][ix].push_back(overlap);
						_fovFovOverlapLists[iIndex2][iPosY][iPosX1].push_back(overlap);
					}
				}
				// Create right overlap and add to list
				if(iPosX2>0 && iPosY>0)
				{
					FovFovOverlap overlap(
						&_pMosaics[iIndex1], &_pMosaics[iIndex2],
						pair<unsigned int, unsigned int>(ix, iy),
						pair<unsigned int, unsigned int>(iPosX2, iPosY),
						_validRect, bMask);

					if(overlap.Columns()>_iMinOverlapSize || overlap.Rows()>_iMinOverlapSize)
					{
						_fovFovOverlapLists[iIndex1][iy][ix].push_back(overlap);
						_fovFovOverlapLists[iIndex2][iPosY][iPosX2].push_back(overlap);
					}
				}
			} //if(bColCol)

			//Row by Row overlap
			if(bRowRow)
			{
				// Find the nearest vertical line
				int iPosX = -1;
				for(kx=0; kx<iNumX2; kx++)
				{
					double dis = fabs(pdCenX1[ix] -pdCenX2[kx]);
					double dMinDis = 2*dSpanX;
					if(dMinDis>dis && dis < 0.5*dSpanX)
					{
						dMinDis = dis;
						iPosX = kx; 
					}
				}
				
				// Any image nearby in the nearest vertical line
				for(ky=0; ky<iNumY2; ky++)
				{
					double dis = fabs(pdCenX1[ix] -pdCenX2[kx]);
					if(dis < 0.8*dSpanY && iPosX>0)
					{
						FovFovOverlap overlap(
							&_pMosaics[iIndex1], &_pMosaics[iIndex2],
							pair<unsigned int, unsigned int>(ix, iy),
							pair<unsigned int, unsigned int>(iPosX, ky),
							_validRect, bMask);

						if(overlap.Columns()>_iMinOverlapSize || overlap.Rows()>_iMinOverlapSize)
						{
							_fovFovOverlapLists[iIndex1][iy][ix].push_back(overlap);
							_fovFovOverlapLists[iIndex2][ky][iPosX].push_back(overlap);
						}
					}
				}
			} // Row by Row
		}
	}

	return true;
}

// Create Fov and Fov overlaps
void OverlapManager::CreateFovFovOverlaps()
{
	unsigned int i, j;
	for(i=0; i<_iNumIllumination; i++)
	{
		for(j=i; j<_iNumIllumination; j++)
		{
			CreateFovFovOverlapsForTwoIllum(i, j);
		}
	}
}

// Create Cad and Fov overlaps
void OverlapManager::CreateCadFovOverlaps()
{
	unsigned int i, kx, ky;
	for(i=0; i<_iNumIllumination; i++)
	{
		if(!_pMosaics[i].UseCad()) // If not use Cad
			continue;

		// If use Cad
		unsigned int iNumImgX = _pMosaics[i].NumImInX();
		unsigned int iNumImgY = _pMosaics[i].NumImInY();
		for(ky=0; ky<iNumImgY; ky++)
		{
			for(kx=0; kx<iNumImgX; kx++)
			{
				CadFovOverlap overlap(
					&_pMosaics[i],
					pair<unsigned int, unsigned int>(kx, ky),
					_pCadImg,
					_validRect);

				if(overlap.Columns()>_iMinOverlapSize || overlap.Rows()>_iMinOverlapSize)
					_cadFovOverlapLists[i][ky][kx].push_back(overlap);
			}
		}
	}
}

// Create Fiducial and Fov overlaps
void OverlapManager::CreateFidFovOverlaps()
{
}
