#include "ImgTransformCamModel.h"

#include "rtypes.h"
#include "dot2d.h"
#include "lsqrpoly.h"
#include "morph.h"


// TODO TODO remove hard coded values
//#define NOM_CAM_ROWS	1944
//#define NOM_CAM_COLS	2592
//#define NOM_CAM_HEIGHT	0.370
TransformCamModel::TransformCamModel()
{
	Reset();
}

TransformCamModel::TransformCamModel(const TransformCamModel& orig)
{
	*this = orig; 
}
			
void TransformCamModel::operator=(const TransformCamModel& b)
{
	for (unsigned int i(0); i < 2; ++i)
		for (unsigned int j(0); j < MORPH_BASES; ++j)
			for (unsigned int k(0); k < MORPH_BASES; ++k)
			{
				_S[i][j][k]=b._S[i][j][k];
				_dSdz[i][j][k]=b._dSdz[i][j][k];
				_SInverse[i][j][k]=b._SInverse[i][j][k];
				_dSdzInverse[i][j][k]=b._dSdzInverse[i][j][k];
			}
	_xMin = b._xMin;
	_xMax = b._xMax;
	_yMin = b._yMin;
	_yMax = b._yMax;
	_uMin = b._uMin;
	_uMax = b._uMax;
	_vMin = b._vMin;
	_vMax = b._vMax;

	_linearTrans = 	b._linearTrans;
	_bCombinedCalibration = b._bCombinedCalibration;
}

void TransformCamModel::Reset()
{
	// just zero out everything, close to real reset values don't help anything
	for (unsigned int i(0); i < 2; ++i)
		for (unsigned int j(0); j < MORPH_BASES; ++j)
			for (unsigned int k(0); k < MORPH_BASES; ++k)
			{
				_S[i][j][k]=0;
				_dSdz[i][j][k]=0;
				_SInverse[i][j][k]=0;
				_dSdzInverse[i][j][k]=0;
			}
	_uMin = _vMin = _xMin = _yMin = 0;
	_uMax = _vMax = _xMax = _yMax = 0;

	_bCombinedCalibration = false;
}

bool TransformCamModel::SetS(int iIndex, float pfVal[16])	// iIndex: 0 for Y and 1 for X
{
	// Validation check
	if(iIndex<0 || iIndex>1)
		return(false);

	for(int i= 0; i<16; i++)
	{
		int i2 = i%4;
		int i3 = i - i2*4;
		_S[iIndex][i2][i3] = pfVal[i];
	}

	return(true);
}

bool TransformCamModel::SetdSdz(int iIndex, float pfVal[16])	// iIndex: 0 for Y and 1 for X
{
	// Validation check
	if(iIndex<0 || iIndex>1)
		return(false);

	for(int i= 0; i<16; i++)
	{
		int i2 = i%4;
		int i3 = i - i2*4;
		_dSdz[iIndex][i2][i3] = pfVal[i];
	}

	return(true);
}

void TransformCamModel::SetLinerCalibration(double* pdVal)
{
	double dT[9];
	for(int i=0; i<8; i++)
		dT[i] = pdVal[i];
	dT[8] = 1;

	_linearTrans.SetMatrix(dT);
	_bCombinedCalibration = true;
}

void TransformCamModel::CalculateInverse()
{
	// find the inverse of the 'm' arrays by point fitting
	int nxi = 11;  // set to 3 uses corner and centers of edges (along with image center)
	int nyi = 11;
	// method of calculating inverse:
	// find extents of region in xy ?? is this already done??
	// tranform pts on grid in image to points in xy (which are within extents)
	// scale xy location to 0,0 to m,n (image size)
	// fit the xy values to the uv values (model after MorphCofs)
	POINT2D_C  xy;
	POINTPIX uv, uvtemp;
	double *rhs, *sysmat, *sigma;
	float *dots;
	void *buf;
	int i, idot, ipt, j, k, l, ndot, npts, retval, syssize;

	int ncbases(MORPH_BASES);
	int nrbases(MORPH_BASES);
	int nrows = (int)(_vMax-_vMin);
	int ncols = (int)(_uMax-_uMin);  // It seems that [u,v]Min should always be 0

	ndot = ncbases*nrbases;
	npts = nxi*nyi;
	syssize = npts*ndot;
	buf = malloc((syssize+ndot)*sizeof(*sysmat)
			   + ndot         *sizeof(*dots)
			   + npts         *sizeof(*rhs)
			   );
	//if (!buf) return -1;
	sysmat = (double *) buf;
	rhs = sysmat + syssize;
	sigma = rhs + npts;
	dots = (float *) (sigma + ndot);

	for (k=0; k<ncbases; k++) {
	  for (l=0; l<nrbases; l++) {
		 dots[l+k*nrbases] = 0.;
	  }
	}
	// modify, take uv[], xy[], eval limits (_xMin,max etc) as input
	// make own function (it will be used when loading align fit to CAD)
	/* xwarp */
	// COPIED FROM MorphCofs()
	for (i=0, ipt=0; i<nyi; i++) {      /*uv.v value is image ROW which is actually parallel to CAD X */
	  uv.v = (nrows-1)*i/(nyi-1);
	  uvtemp.v = uv.v - 0.5*(nrows-1);
	  for (j=0; j<nxi; j++) {          /* uv.u value is image COL, parralel to CAD Y*/
		 uv.u = (ncols-1)*j/(nxi-1);
		 uvtemp.u = uv.u - 0.5*(ncols-1);
		 xy = SPix2XY(uvtemp);   // xy.x, xy.y are in CAD directions
		 //xy.x += 0.5*(ncols-1);
		 //xy.y += 0.5*(nrows-1);
		 xy.x -= _xMin;				//remove offset
		 xy.x *= 1/(_xMax - _xMin);	// scale to 0 - 1 range
		 xy.x *= nrows;				// scale to 
		 xy.y -= _yMin;
		 xy.y *= 1/(_yMax - _yMin);
		 xy.y *= ncols;
		 rhs[ipt] = xy.x;

		 for (k=0, idot=0; k<ncbases; k++) {
			for (l=0; l<nrbases; l++) {
			   dots[l+k*nrbases] = 1.;
			   sysmat[ipt + idot*npts] =
				  htcorrp(nrows, ncols,
						  xy.x, xy.y,    // MorphCofs uses xy.x, uv.y !! We're only doing forward version so we use xy.x, xy.y !!
						  nrbases, ncbases,
						  dots, nrbases);
			   dots[l+k*nrbases] = 0.;
			   idot++;
			}
		 }
		 ipt++;
	  }
	}

	retval = qrd(sysmat, sigma, npts, npts, ndot);
	// TODO add logging
	//if (retval) 
	//	G_LOG_1_ERROR("xwarp calc, qrd failure, returned %d", retval);
	retval = qrsolv(sysmat, sigma, rhs, npts, npts, ndot, 1);
	//if (retval) 
	//	G_LOG_1_ERROR("xwarp calc, qrsolv failure, returned %d", retval);
	

	for (i=0; i<MORPH_BASES; i++) {
		for(j=0; j<MORPH_BASES; j++) {
			_SInverse[1][i][j] = (float)rhs[i*MORPH_BASES+j];
		}
	}

	/* ywarp */

	for (i=0, ipt=0; i<nyi; i++) {      /* y */
	  uv.v = (nrows-1)*i/(nyi-1);
	  uvtemp.v = uv.v - 0.5*(nrows-1);
	  for (j=0; j<nxi; j++) {          /* x */
		 uv.u = (ncols-1)*j/(nxi-1);
		 uvtemp.u = uv.u - 0.5*(ncols-1);
		 xy = SPix2XY(uvtemp);
		 //xy.x += 0.5*(ncols-1);
		 //xy.y += 0.5*(nrows-1);
		 xy.x -= _xMin;				//remove offset
		 xy.x *= 1/(_xMax - _xMin);	// scale to 0 - 1 range
		 xy.x *= nrows;				// scale to pseudo image size
		 xy.y -= _yMin;
		 xy.y *= 1/(_yMax - _yMin);
		 xy.y *= ncols;
		 rhs[ipt] = xy.y;

		 for (k=0, idot=0; k<ncbases; k++) {
			for (l=0; l<nrbases; l++) {
			   dots[l+k*nrbases] = 1.;
			   sysmat[ipt + idot*npts] =
				  htcorrp(nrows, ncols,
						  xy.x, xy.y,
						  nrbases, ncbases,
						  dots, nrbases);
			   dots[l+k*nrbases] = 0.;
			   idot++;
			}
		 }
		 ipt++;
	  }
	}
	retval = qrd(sysmat, sigma, npts, npts, ndot);
	// TODO add logging
	/*if (retval) 
		G_LOG_1_ERROR("ywarp calc, qrd failure, returned %d", retval);*/
	retval = qrsolv(sysmat, sigma, rhs, npts, npts, ndot, 1);
	// TODO add logging
	/*if (retval) 
		G_LOG_1_ERROR("ywarp calc, qrsolv failure, returned %d", retval);
	*/

	for (i=0; i<MORPH_BASES; i++) {
		for(j=0; j<MORPH_BASES; j++) {
			_SInverse[0][i][j] = (float)rhs[i*MORPH_BASES+j];
		}
	}

	free(buf);
	//return 0;
}

// given a field of points uv, xy, (assumed to include corners?) calc uv2xy transform (m), u/v/x/y Min/Max
void TransformCamModel::CalcTransform(POINTPIX* uv, POINT2D_C* xy, unsigned int npts)
{
	// find min, max values
	_uMin = _vMin = _yMin = _xMin = 100000;
	_uMax = _vMax = _xMax = _yMax = -100000;
	for (unsigned int i(0); i < npts; ++i)
	{
		if (uv[i].u < _uMin) _uMin = uv[i].u;
		if (uv[i].u > _uMax) _uMax = uv[i].u;
		if (uv[i].v < _vMin) _vMin = uv[i].v;
		if (uv[i].v > _vMax) _vMax = uv[i].v;
		
		if (xy[i].x < _xMin) _xMin = xy[i].x;
		if (xy[i].x > _xMax) _xMax = xy[i].x;
		if (xy[i].y < _yMin) _yMin = xy[i].y;
		if (xy[i].y > _yMax) _yMax = xy[i].y;
	}
	int ncbases(MORPH_BASES);
	int nrbases(MORPH_BASES);
	int nrows = (int)(_vMax-_vMin);
	int ncols = (int)(_uMax-_uMin);
	double *rhs, *sysmat, *sigma;
	float *dots;
	void *buf;
	int  syssize, k, l, i, j, idot;
	int retval;
	
	int ndot = ncbases*nrbases;
	//npts = nxi*nyi;
	syssize = npts*ndot;
	buf = malloc((syssize+ndot)*sizeof(*sysmat)
			   + ndot         *sizeof(*dots)
			   + npts         *sizeof(*rhs)
			   );
	//if (!buf) return -1;
	sysmat = (double *) buf;
	rhs = sysmat + syssize;
	sigma = rhs + npts;
	dots = (float *) (sigma + ndot);

	for (k=0; k<ncbases; k++) {
	  for (l=0; l<nrbases; l++) {
		 dots[l+k*nrbases] = 0.;
	  }
	}
	double ztemp;
	POINT2D xyPseudo;		// pseudo pixel value (xy scaled from meters to points in an image like space
	for (i=0; i<(int)npts; i++) {      /*uv.v value is image ROW which is actually parallel to CAD X */
		
		rhs[i] = xy[i].x; 

		for (k=0, idot=0; k<ncbases; k++) {
			for (l=0; l<nrbases; l++) {
			   dots[l+k*nrbases] = 1.;
			   ztemp = 
				  htcorrp(nrows, ncols,
						  uv[i].u, uv[i].v,    // MorphCofs uses xy.x, uv.y !! We're only doing forward version so we use xy.x, xy.y !!
						  nrbases, ncbases,
						  dots, nrbases);
			   sysmat[i + idot*npts] = ztemp;
			   dots[l+k*nrbases] = 0.;
			   idot++;
			}
		}
	}

	retval = qrd(sysmat, sigma, npts, npts, ndot);
	// TODO add logging
	/*if (retval) 
		G_LOG_1_ERROR("xwarp calc, qrd failure, returned %d", retval);*/
	retval = qrsolv(sysmat, sigma, rhs, npts, npts, ndot, 1);
	// TODO add logging
	/*if (retval) 
		G_LOG_1_ERROR("xwarp calc, qrsolv failure, returned %d", retval);*/
	

	for (i=0; i<MORPH_BASES; i++) {
		for(j=0; j<MORPH_BASES; j++) {
			_S[1][i][j] = (float)rhs[i*MORPH_BASES+j];
		}
	}
	for (i=0; i<(int)npts; i++) {      /*uv.u value is image ROW which is actually parallel to CAD Y */
		
		rhs[i] = xy[i].y; 

		for (k=0, idot=0; k<ncbases; k++) {
			for (l=0; l<nrbases; l++) {
			   dots[l+k*nrbases] = 1.;
			   sysmat[i + idot*npts] =
				  htcorrp(nrows, ncols,
						  uv[i].u, uv[i].v,   
						  nrbases, ncbases,
						  dots, nrbases);
			   dots[l+k*nrbases] = 0.;
			   idot++;
			}
		}
	}

	retval = qrd(sysmat, sigma, npts, npts, ndot);
	// TODO add logging
	/*if (retval) 
		G_LOG_1_ERROR("xwarp calc, qrd failure, returned %d", retval);*/
	retval = qrsolv(sysmat, sigma, rhs, npts, npts, ndot, 1);
	// TODO add logging
	/*if (retval) 
		G_LOG_1_ERROR("xwarp calc, qrsolv failure, returned %d", retval);*/
	

	for (i=0; i<MORPH_BASES; i++) {
		for(j=0; j<MORPH_BASES; j++) {
			_S[0][i][j] = (float)rhs[i*MORPH_BASES+j];
		}
	}
	// Next populate the inverse direction....
	for (i=0; i<(int)npts; i++) {      /*uv.v value is image ROW which is actually parallel to CAD X */
		xyPseudo.x = xy[i].x;
		xyPseudo.y = xy[i].y;
		xyPseudo.x -= _xMin;				//remove offset
		xyPseudo.x *= 1/(_xMax - _xMin);	// scale to 0 - 1 range
		xyPseudo.x *= nrows;				// scale to 
		xyPseudo.y -= _yMin;
		xyPseudo.y *= 1/(_yMax - _yMin);
		xyPseudo.y *= ncols;
		rhs[i] = uv[i].v; 

		for (k=0, idot=0; k<ncbases; k++) {
			for (l=0; l<nrbases; l++) {
			   dots[l+k*nrbases] = 1.;
			   sysmat[i + idot*npts] =
				  htcorrp(nrows, ncols,
						  xyPseudo.y, xyPseudo.x,    // MorphCofs uses xy.x, uv.y !! We're only doing forward version so we use xy.x, xy.y !!
						  nrbases, ncbases,
						  dots, nrbases);
			   dots[l+k*nrbases] = 0.;
			   idot++;
			}
		}
	}

	retval = qrd(sysmat, sigma, npts, npts, ndot);
	// TODO add logging
	/*if (retval) 
		G_LOG_1_ERROR("xwarp calc, qrd failure, returned %d", retval);*/
	retval = qrsolv(sysmat, sigma, rhs, npts, npts, ndot, 1);
	// TODO add logging
	/*if (retval) 
		G_LOG_1_ERROR("xwarp calc, qrsolv failure, returned %d", retval);*/
	

	for (i=0; i<MORPH_BASES; i++) {
		for(j=0; j<MORPH_BASES; j++) {
			_SInverse[1][i][j] = (float)rhs[i*MORPH_BASES+j];
		}
	}
	for (i=0; i<(int)npts; i++) {      /*uv.u value is image ROW which is actually parallel to CAD Y */
		xyPseudo.x = xy[i].x;
		xyPseudo.y = xy[i].y;
		xyPseudo.x -= _xMin;				//remove offset
		xyPseudo.x *= 1/(_xMax - _xMin);	// scale to 0 - 1 range
		xyPseudo.x *= nrows;				// scale to 
		xyPseudo.y -= _yMin;
		xyPseudo.y *= 1/(_yMax - _yMin);
		xyPseudo.y *= ncols;
		rhs[i] = uv[i].u; 

		for (k=0, idot=0; k<ncbases; k++) {
			for (l=0; l<nrbases; l++) {
			   dots[l+k*nrbases] = 1.;
			   sysmat[i + idot*npts] =
				  htcorrp(nrows, ncols,
						  xyPseudo.y, xyPseudo.x,   
						  nrbases, ncbases,
						  dots, nrbases);
			   dots[l+k*nrbases] = 0.;
			   idot++;
			}
		}
	}

	retval = qrd(sysmat, sigma, npts, npts, ndot);
	// TODO add logging
	/*if (retval) 
		G_LOG_1_ERROR("xwarp calc, qrd failure, returned %d", retval);*/
	retval = qrsolv(sysmat, sigma, rhs, npts, npts, ndot, 1);
	// TODO add logging
	/*if (retval) 
		G_LOG_1_ERROR("xwarp calc, qrsolv failure, returned %d", retval);*/
	

	for (i=0; i<MORPH_BASES; i++) {
		for(j=0; j<MORPH_BASES; j++) {
			_SInverse[0][i][j] = (float)rhs[i*MORPH_BASES+j];
		}
	}
	free(buf);
}

// S for (col, row)->(y,x) in world space
POINT2D_C TransformCamModel::SPix2XY(POINTPIX uv)
{
	float *xwarp;
	float *ywarp;
	xwarp = (float*)_S[1];		// X in _S[1]
	ywarp = (float*)_S[0];		// Y in _S[0]
		
	// transfrom a point at uv from pixel coord to xy coords.  Assume Z=0
	POINT2D_C xy;
	xy.x = htcorrp((int)(_vMax-_vMin), (int)(_uMax-_uMin),		// u is col and v is row
					uv.u,uv.v,
					MORPH_BASES, MORPH_BASES, 
					xwarp, 
					MORPH_BASES);
	xy.y = htcorrp((int)(_vMax-_vMin), (int)(_uMax-_uMin), 
					uv.u,uv.v,
					MORPH_BASES, MORPH_BASES, 
					ywarp, 
					MORPH_BASES);

	// If calibration is combined
	if(_bCombinedCalibration)
	{
		double dLinearX, dLinearY;
		_linearTrans.Map(uv.v, uv.u, &dLinearX, &dLinearY);

		xy.x += dLinearX;
		xy.y += dLinearY;
	}

	return xy;
}

void TransformCamModel::SPix2XY(double u, double v, double* px, double* py)
{
	POINTPIX uv(u,v);
	POINT2D_C xy = SPix2XY(uv);
	*px = xy.x;
	*py = xy.y;
}

// dSdZ for (col, row)->(y,x) in world space
POINT2D_C TransformCamModel::dSPix2XY(POINTPIX uv)
{
	float *xwarp;
	float *ywarp;
	xwarp = (float*)_dSdz[1];		// X in _dSdz[1]
	ywarp = (float*)_dSdz[0];		// Y in _dSdz[0]
		
	// transfrom a point at uv from pixel coord to xy coords.  Assume Z=0
	POINT2D_C xy;
	xy.x = htcorrp((int)(_vMax-_vMin), (int)(_uMax-_uMin),		// u is col and v is row
					uv.u,uv.v,
					MORPH_BASES, MORPH_BASES, 
					xwarp, 
					MORPH_BASES);
	xy.y = htcorrp((int)(_vMax-_vMin), (int)(_uMax-_uMin), 
					uv.u,uv.v,
					MORPH_BASES, MORPH_BASES, 
					ywarp, 
					MORPH_BASES);
	return xy;
}

void TransformCamModel::dSPix2XY(double u, double v, double* px, double* py)
{
	POINTPIX uv(u,v);
	POINT2D_C xy = dSPix2XY(uv);
	*px = xy.x;
	*py = xy.y;
}




