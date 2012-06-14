/*
 * Originally:
 *
 * Concave Polygon Scan Conversion
 * by Paul Heckbert
 * from "Graphics Gems", Academic Press, 1990
 *
 * concave: scan convert nvert-sided concave non-simple polygon with
 * vertices at (point[i].x, point[i].y) for i in [0..nvert-1] within
 * the window win by calling spanproc for each visible span of pixels.
 * Polygon can be clockwise or counterclockwise.  Algorithm does
 * uniform point sampling at pixel centers.  Inside-outside test done
 * by Jordan's rule: a point is considered inside if an emanating ray
 * intersects the polygon an odd number of times.
 *
 *  Paul Heckbert 30 June 81, 18 Dec 89 
 */
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include "shape.h"

#ifndef min
template<class T> T min(T a, T b) { return a<b ? a : b; }
#endif
#ifndef max
template<class T> T max(T a, T b) { return a>b ? a : b; }
#endif

inline int loclip(int lo, double v) { return max(lo, int(ceil(v-.5))); }
inline int hiclip(int hi, double v) { return min(hi-1, int(floor(v-.5))); }

struct edge // a polygon edge 
{
	double x; // x coordinate of edge's intersection with current scanline 
	double dx; // change in x with respect to y 
	int i; // edge number: edge i goes from pt[i] to pt[i+1] 
	edge (double X, double DX, int I) : x(X), dx(DX), i(I) {}
};

struct idx
{
	int i; // vertex number
	double y; // vertex y value
	idx(int I, double Y) : i(I), y(Y) {}
};

static void cdelete(xlist<edge> &aet, int i) // remove edge i from active list 
{
	for(listiter<edge> a(aet); a.on(); a++)
		if(a->i==i)
		{
			a.del();
			return;
		}
}

static void cinsert(xlist<edge> &aet, const xlist<vector3> &v, int i, int y)
{
	double dx, x;
	int j=(i<v.count()-1)? i+1 : 0;
	const vector3 *p=&v[i], *q=&v[j];

	dx = (q->x - p->x)/(q->y - p->y); // slope
	if(p->y >= q->y) p = q; // choose the lower vertex
	x = dx*(y+.5 - p->y) + p->x;
	
	aet.push(edge(x, dx, i));
}

static int cmpaet(edge &a, edge &b, void *) { return a.x<b.x? -1 : 0; }
static int cmpind(idx &a, idx &b, void *) { return a.y<b.y? -1 : 0; }

void polygon::scan(const rect &b, spanproc *span, void *d) const
{
	int i=0, j, y, ymin, ymax, xl, xr;
	xlist<edge> aet;
	xlist<idx> ind;
	
	// create y-sorted array of indices ind[k] into vertex list 
	for (listiter<vector3> a(p); a.on(); a++, i++) 
		ind.push(idx(i, a->y));

	ind.sort(cmpind);
	listiter<idx> k(ind);

	ymin = loclip(b.y1, ind[0].y);
	ymax = hiclip(b.y2, ind[-1].y);
	
	for(y=ymin; y<=ymax; y++)
	{
		// check vertices between previous scanline and current one 
		for(; k.on() && k->y <= y+.5; k++) 
		{
			i = k->i;
			
			j = i>0 ? i-1 : p.count()-1; // previous vertex
			if(p[j].y <= y-.5) cdelete(aet, j);
			else if(p[j].y > y+.5) cinsert(aet, p, j, y);
			
			j = i<p.count()-1 ? i+1 : 0; // next vertex
			if(p[j].y <= y-.5) cdelete(aet, i);
			else if(p[j].y > y+.5) cinsert(aet, p, i, y);
		}
		
		aet.sort(cmpaet);

		for (listiter<edge> e(aet); e.on(); e++) 
		{	
			xl = loclip(b.x1, e->x);
			e->x += e->dx; // increment edge coords 

			e++;
			if(!e.on()) break;

			xr = hiclip(b.x2, e->x);
			e->x += e->dx;
			
			if(xl<=xr) span(y, xl, xr, d);
		}
	}
}

void ellipse::scan(const rect &b, spanproc *s, void *d) const
{
	double y, w;
	int ymin=loclip(b.y1, c.y-r.y), ymax=hiclip(b.y2, c.y+r.y);

	for(int l=ymin; l<ymax; l++)
	{
		y=l-c.y+.5;
		w=r.x * sqrt(1-y*y/(r.y*r.y));
		s(l, loclip(b.x1, c.x-w), hiclip(b.x2, c.x+w), d);
	}
}

void donut::scan(const rect &b, spanproc *s, void *d) const
{
	double y, w;
	double iy,iw;
	vector3 c=outer.centroid();
	vector3 r=outer.radius();
	vector3 ic=inner.centroid();
	vector3 ir=inner.radius();

	int ymin=loclip(b.y1, c.y-r.y), ymax=hiclip(b.y2, c.y+r.y);

	for(int l=ymin; l<ymax; l++)
	{
		y=l-c.y+.5;
		w=r.x * sqrt(1-y*y/(r.y*r.y));

		if((l<=ic.y-ir.y) || (l>=ic.y+ir.y))
			s(l, loclip(b.x1, c.x-w), hiclip(b.x2, c.x+w), d);
		else
		{
			iy=l-ic.y+.5;
			iw=ir.x * sqrt(1-iy*iy/(ir.y*ir.y));

			s(l, loclip(b.x1, c.x-w), hiclip(b.x2, ic.x-iw), d);
			s(l, loclip(b.x1, ic.x+iw), hiclip(b.x2, c.x+w), d);
		}
	}
}

struct tracedata
{
	shape::spanproc *s;
	void *d;
	int xl, xr, y; // values from last call
};

static void inline 
edgetrace(int y, int xo, int xn, shape::spanproc *s, void *d)
{
		if(xo == xn)     s(y, xo,   xn,   d);
		else if(xo < xn) s(y, xo+1, xn,   d);
		else             s(y, xn,   xo-1, d);
}

// this version only handles convex - no two spans on the same line
static void traceproc(int y, int xl, int xr, void *td)
{
	tracedata *t=(tracedata *)td;

	if(t->y==INT_MAX) t->s(y, xl, xr, t->d); // top
	else
	{
		edgetrace(y, t->xl, xl, t->s, t->d);
		edgetrace(y, t->xr, xr, t->s, t->d);
	}

	t->xl=xl;
	t->xr=xr;
	t->y=y;
}

void shape::trace(const rect &b, spanproc *s, void *d) const
{
	tracedata t={ s, d, 0, 0, INT_MAX };

	scan(b, traceproc, &t);
	if(t.y != INT_MAX && t.xl<t.xr-1) s(t.y, t.xl+1, t.xr-1, d); // bottom
}



