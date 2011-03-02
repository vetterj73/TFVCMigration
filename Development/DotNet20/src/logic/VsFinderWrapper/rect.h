#ifndef __RECT_H__
#define __RECT_H__

class rect
{
#ifndef min
 private:
	int min(int a, int b) const { return a<b?a:b; }
	int max(int a, int b) const { return a>b?a:b; }
#endif	

 public:
	int x1,y1,x2,y2; // dimensions

	rect() : x1(0), y1(0), x2(0), y2(0) {}
	rect(int width, int height) : x1(0), y1(0), x2(width), y2(height) {}
	rect(int X1, int Y1, int X2, int Y2) : x1(X1), y1(Y1), x2(X2), y2(Y2) {}
	
	operator int() const { return (x1!=x2&&y1!=y2); }
	int width() const {return x2-x1;}
	int height() const {return y2-y1;}
	int area() const {return width()*height();}
	int perimeter() const {return 2*(width()+height());}
	
	rect operator+(const rect &R) const 
		{ return rect(x1+R.x1, y1+R.y1, x2+R.x2, y2+R.y2); }
	rect operator-(const rect &R) const 
		{ return rect(x1-R.x1, y1-R.y1, x2-R.x2, y2-R.y2); }
	int operator==(const rect &R)
	{return x1==R.x1 && x2==R.x2 && y1==R.y1 && y2==R.y2;}

	rect inflate(int x, int y) const { return *this+rect(-x,-y,x,y); }
	rect inflate(int v) const { return inflate(v,v); }
	rect deflate(int x, int y) const { return inflate(-x,-y); }
	rect deflate(int v) const { return deflate(v,v); }
	rect offset(int x, int y) const { return *this+rect(x,y,x,y); }

	rect center(const rect &R) const
		{ return offset((R.x2-R.x1-x2+x1)/2,(R.y2-R.y1-y2+y1)/2); }

	rect operator |(const rect &R) const // union
		{ 
			return rect(min(x1,R.x1),min(y1,R.y1),
				    max(x2,R.x2),max(y2,R.y2)); 
		}

	rect operator &(const rect &R) const // intersection
		{ 
			rect T(max(x1,R.x1),max(y1,R.y1),
			       min(x2,R.x2),min(y2,R.y2));
			if(T.x1>T.x2||T.y1>T.y2)
				return rect();
			return T;
		}

	rect operator *(int a) const { return rect(x1*a, y1*a, x2*a, y2*a); }
};



#endif
