#include "ImgTransformCamModel.h"

// TODO TODO remove hard coded values
//#define NOM_CAM_ROWS	1944
//#define NOM_CAM_COLS	2592
//#define NOM_CAM_HEIGHT	0.370
TransformCamModel::TransformCamModel()
{
	Reset();
}

// TODO !!  TODO !!  TODO !!  TODO !!  TODO !!  TODO !!  TODO !!  TODO !!  TODO !!  TODO !!  TODO !!  TODO !!  
// add operations on dmdz and dmdzInverse


void TransformCamModel::Reset()
{
	// just zero out everything, close to real reset values don't help anything
	for (unsigned int i(0); i < 2; ++i)
		for (unsigned int j(0); j < MORPH_BASES; ++j)
			for (unsigned int k(0); k < MORPH_BASES; ++k)
			{
				S[i][j][k]=0;
				dSdz[i][j][k]=0;
				SInverse[i][j][k]=0;
				dSdzInverse[i][j][k]=0;
			}
	uMin = vMin = xMin = yMin = 0;
	uMax = vMax = xMax = yMax = 0;
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
	POINT2D  xy;
	POINTPIX uv, uvtemp;
	double *rhs, *sysmat, *sigma;
	float *dots;
	void *buf;
	int i, idot, ipt, j, k, l, ndot, npts, retval, syssize;

	int ncbases(MORPH_BASES);
	int nrbases(MORPH_BASES);
	int nrows = (int)(vMax-vMin);
	int ncols = (int)(uMax-uMin);  // It seems that [u,v]Min should always be 0

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
	// modify, take uv[], xy[], eval limits (xMin,max etc) as input
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
		 xy.x -= xMin;				//remove offset
		 xy.x *= 1/(xMax - xMin);	// scale to 0 - 1 range
		 xy.x *= nrows;				// scale to 
		 xy.y -= yMin;
		 xy.y *= 1/(yMax - yMin);
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
			SInverse[1][i][j] = (float)rhs[i*MORPH_BASES+j];
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
		 xy.x -= xMin;				//remove offset
		 xy.x *= 1/(xMax - xMin);	// scale to 0 - 1 range
		 xy.x *= nrows;				// scale to pseudo image size
		 xy.y -= yMin;
		 xy.y *= 1/(yMax - yMin);
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
			SInverse[0][i][j] = (float)rhs[i*MORPH_BASES+j];
		}
	}


	free(buf);
	//return 0;



}
// given a field of points uv, xy, (assumed to include corners?) calc uv2xy transform (m), u/v/x/y Min/Max
void TransformCamModel::CalcTransform(POINTPIX* uv, POINT2D* xy, unsigned int npts)
{
	// find min, max values
	uMin = vMin = yMin = xMin = 100000;
	uMax = vMax = xMax = yMax = -100000;
	for (unsigned int i(0); i < npts; ++i)
	{
		if (uv[i].u < uMin) uMin = uv[i].u;
		if (uv[i].u > uMax) uMax = uv[i].u;
		if (uv[i].v < vMin) vMin = uv[i].v;
		if (uv[i].v > vMax) vMax = uv[i].v;
		
		if (xy[i].x < xMin) xMin = xy[i].x;
		if (xy[i].x > xMax) xMax = xy[i].x;
		if (xy[i].y < yMin) yMin = xy[i].y;
		if (xy[i].y > yMax) yMax = xy[i].y;
	}
	int ncbases(MORPH_BASES);
	int nrbases(MORPH_BASES);
	int nrows = (int)(vMax-vMin);
	int ncols = (int)(uMax-uMin);
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
			S[1][i][j] = (float)rhs[i*MORPH_BASES+j];
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
			S[0][i][j] = (float)rhs[i*MORPH_BASES+j];
		}
	}
	// Next populate the inverse direction....
	for (i=0; i<(int)npts; i++) {      /*uv.v value is image ROW which is actually parallel to CAD X */
		xyPseudo.x = xy[i].x;
		xyPseudo.y = xy[i].y;
		xyPseudo.x -= xMin;				//remove offset
		xyPseudo.x *= 1/(xMax - xMin);	// scale to 0 - 1 range
		xyPseudo.x *= nrows;				// scale to 
		xyPseudo.y -= yMin;
		xyPseudo.y *= 1/(yMax - yMin);
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
			SInverse[1][i][j] = (float)rhs[i*MORPH_BASES+j];
		}
	}
	for (i=0; i<(int)npts; i++) {      /*uv.u value is image ROW which is actually parallel to CAD Y */
		xyPseudo.x = xy[i].x;
		xyPseudo.y = xy[i].y;
		xyPseudo.x -= xMin;				//remove offset
		xyPseudo.x *= 1/(xMax - xMin);	// scale to 0 - 1 range
		xyPseudo.x *= nrows;				// scale to 
		xyPseudo.y -= yMin;
		xyPseudo.y *= 1/(yMax - yMin);
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
			SInverse[0][i][j] = (float)rhs[i*MORPH_BASES+j];
		}
	}
	free(buf);


}

// S for (col, row)->(y,x) in world space
POINT2D TransformCamModel::SPix2XY(POINTPIX uv)
{
	float *xwarp;
	float *ywarp;
	xwarp = (float*)S[1];		// X in S[1]
	ywarp = (float*)S[0];		// Y in S[0]
		
	// transfrom a point at uv from pixel coord to xy coords.  Assume Z=0
	POINT2D xy;
	xy.x = htcorrp((int)(vMax-vMin), (int)(uMax-uMin),		// u is col and v is row
					uv.u,uv.v,
					MORPH_BASES, MORPH_BASES, 
					xwarp, 
					MORPH_BASES);
	xy.y = htcorrp((int)(vMax-vMin), (int)(uMax-uMin), 
					uv.u,uv.v,
					MORPH_BASES, MORPH_BASES, 
					ywarp, 
					MORPH_BASES);
	return xy;

}

// dSdZ for (col, row)->(y,x) in world space
POINT2D TransformCamModel::dSPix2XY(POINTPIX uv)
{
	float *xwarp;
	float *ywarp;
	xwarp = (float*)dSdz[1];		// X in S[1]
	ywarp = (float*)dSdz[0];		// Y in S[0]
		
	// transfrom a point at uv from pixel coord to xy coords.  Assume Z=0
	POINT2D xy;
	xy.x = htcorrp((int)(vMax-vMin), (int)(uMax-uMin),		// u is col and v is row
					uv.u,uv.v,
					MORPH_BASES, MORPH_BASES, 
					xwarp, 
					MORPH_BASES);
	xy.y = htcorrp((int)(vMax-vMin), (int)(uMax-uMin), 
					uv.u,uv.v,
					MORPH_BASES, MORPH_BASES, 
					ywarp, 
					MORPH_BASES);
	return xy;
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
				S[i][j][k]=b.S[i][j][k];
				dSdz[i][j][k]=b.dSdz[i][j][k];
				SInverse[i][j][k]=b.SInverse[i][j][k];
				dSdzInverse[i][j][k]=b.dSdzInverse[i][j][k];
			}
	xMin = b.xMin;
	xMax = b.xMax;
	yMin = b.yMin;
	yMax = b.yMax;
	uMin = b.uMin;
	uMax = b.uMax;
	vMin = b.vMin;
	vMax = b.vMax;
}

