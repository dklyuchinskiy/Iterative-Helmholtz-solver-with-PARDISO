#pragma once
//Header file of Complex.cpp

double ZABS(REAL, REAL);
void ZSQRT(REAL, REAL, REAL *, REAL *);
void ZEXP(REAL, REAL, REAL *, REAL *);
void ZMLT(REAL, REAL, REAL, REAL, REAL *, REAL *);
void ZDIV(REAL, REAL, REAL, REAL, REAL *, REAL *);
void ZLOG(REAL, REAL, REAL *, REAL *, int *);

double DMAX(REAL, REAL);
double DMIN(REAL, REAL);

int IMAX(int,int);
int IMIN(int,int);

double D1MACH(int);
long I1MACH(int);
