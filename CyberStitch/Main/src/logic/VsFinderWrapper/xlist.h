#ifndef __LIST_H__
#define __LIST_H__

#include <stdio.h>

#pragma warning(disable: 4355)	// 'this' used in base member initialization list (ok to use here)

template<class T> class listiter;

class __link
{
public:
	__link *prev, *next;
	__link(__link *p, __link *n) : prev(p), next(n) {} 
	__link() : prev(this), next(this) {} 
};

template<class T>
class __Tlink : public __link
{
public:
	T e;
	__Tlink(const T &E, __link *p, __link *n) : e(E), __link(p, n) {} 
};

template<class T>
class xlist
{
private:
	typedef listiter<T> iter;
	typedef int cmpfunc(T &a, T &b, void *data);
	friend class listiter<T>; // MS C is broken

	__link h; 
	int c;
	
	T &get(int n) const
	{
		__link *l; 
	
		if(n>=0) 
			for(l=head(); n>0 && l->next!=&h; n--) l=l->next; 
		else    
			for(l=tail(); n<-1 && l->prev!=&h; n++) l=l->prev; 
	
		return ((__Tlink<T> *)l)->e; 
	}
	__link *head() const { return this?h.next:0; }
	__link *tail() const { return this?h.prev:0; }
	static T &e(__link *l) { return ((__Tlink<T> *)l)->e; }
	static void qsort(iter l, iter r, cmpfunc *cmp, void *data)
	{
		if(l==r) return;
	
		iter i(l), p(l);
		
		for(i++; i!=r; i++) 
			if(cmp(*i, *l, data)<0) 
				i.swap(++p);
	
		l.swap(p);
		qsort(l, p, cmp, data);
		qsort(++p, r, cmp, data);
	}
	void dec() { while(count()) popd(); }
	void ins(const T& t, __link *prev, __link *next)
	{
		__link *l=new __Tlink<T>(t, prev, next);
		prev->next=l;
		next->prev=l;
		c++;
	}
	void del(__link *l)
	{
		c--;
		l->prev->next=l->next;
		l->next->prev=l->prev;
		delete (__Tlink<T> *)l;
	}
public:
	xlist() : c(0) {}
	xlist(const xlist &L) : c(0) { *this=L;}
	~xlist() { dec();}
	
	xlist &operator=(const xlist &l)
	{
		if(!(*this==l))
		{
			dec();
			for(iter i((xlist &)l); i.on(); i++) push(*i); 
		}

		return *this; 
	}
	int operator==(const xlist &L) const { return c==L.c; }

	int count() const { return this?c:0; }
	operator int() { return count(); }
	
	void push(const T& t) { ins(t,tail(),&h); }
	void unshift(const T& t) { ins(t,&h,head()); }
	void popd() { del(tail()); }
	void shiftd() { del(head()); }
	void clear() { dec(); }
	T pop() { T t=get(-1); del(tail()); return t; }
	T shift() { T t=get(0); del(head()); return t; }

	T &operator[](int i) { return get(i); }
	const T &operator[](int i) const { return get(i); }

	iter begin() { return iter(this, head()); }
	iter end() { return iter(this, &h); }

	void sort(cmpfunc *c, void *d=0) { qsort(begin(), end(), c, d); }
};				      

template<class T>
class listiter
{
private:
	typedef listiter<T> self;

	::xlist<T> *l;
	__link *c;
public:
	listiter() {}
	listiter(::xlist<T> *L, __link *C) : l(L), c(C) {}
	listiter(::xlist<T> *L) : l(L), c(L->head()) {}
	listiter(::xlist<T> &L) : l(&L), c(L.head()) {}
	listiter(const ::xlist<T> &L) : l((xlist<T>*)&L), c(L.head()) {}
	
	int operator==(const self& x) const { return c == x.c; }
	int operator!=(const self& x) const { return c != x.c; }
	T &operator*() const { return ((__Tlink<T> *)c)->e; }
	T *operator->() const { return &(operator*()); }
	void ins(const T& t) { l->ins(t, c->prev, c); c=c->prev; }
	void del() { __link *t=c->next; l->del(c); c=t; }
	void swap(self &i) { T t=*(*this); *(*this)=*i; *i=t; }

	self& operator++() { c = c->next; return *this; }
	self operator++(int) { self tmp = *this; ++*this; return tmp; }
	self& operator--() { c = c->prev; return *this; }
	self operator--(int) { self tmp = *this; --*this; return tmp; }
	int on() const { return l && c!=&l->h; }
};

/*
template<class T>
T &::list<T>::get(int n) const	
{ 
	__link *l; 

	if(n>=0) 
		for(l=head(); n>0 && l->next!=&h; n--) l=l->next; 
	else    
		for(l=tail(); n<-1 && l->prev!=&h; n++) l=l->prev; 

	return ((__Tlink<T> *)l)->e; 
}
*/

/*
template<class T>
::list<T> &::list<T>::operator=(const list &l) 
{ 
	if(!(*this==l))
	{
		dec();
		for(iter i((list &)l); i.on(); i++) push(*i); 
	}

	return *this; 
}
*/

/*
template<class T>
void ::list<T>::del(__link *l)
{
	c--;
	l->prev->next=l->next;
	l->next->prev=l->prev;
	delete (__Tlink<T> *)l;
}
*/

/*
template<class T>
void ::list<T>::ins(T const &t, __link *prev, __link *next)
{
	__link *l=new __Tlink<T>(t, prev, next);
	prev->next=l;
	next->prev=l;
	c++;
}
*/

// This function is so cool.
/*
template<class T>
void ::list<T>::qsort(iter l, iter r, cmpfunc *cmp, void *data)
{
	if(l==r) return;

	iter i(l), p(l);
	
	for(i++; i!=r; i++) 
		if(cmp(*i, *l, data)<0) 
			i.swap(++p);

	l.swap(p);
	qsort(l, p, cmp, data);
	qsort(++p, r, cmp, data);
}
*/

#endif
