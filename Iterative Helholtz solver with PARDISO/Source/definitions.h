#pragma once

/*****************************
Preprocessor definitions and
declaration of used structures
*****************************/

// C
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>

// C++
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <complex>

typedef double rtype;
typedef std::complex<double> dtype;
#define MKL_Complex16 dtype

#include "C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2018.2.185\windows\mkl\include\mkl.h"
#include "C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2018.2.185\windows\mkl\include\mkl_dfti.h"

//#define DEBUG

#define EPS 0.00000001

#define min(a,b) ((a) <= (b)) ? (a) : (b)

struct MatrixCSRReal {

	int *ia = NULL;
	int *ja = NULL;
	double *values = NULL;
};

typedef struct MatrixCSRReal dcsr;

struct MatrixCSRComplex {

	int *ia = NULL;
	int *ja = NULL;
	dtype *values = NULL;
	int non_zeros = 0;
};

typedef struct MatrixCSRComplex ccsr;


struct size_m {
	double l;
	int n;
	double h;
	int pml_pts;
};

struct point {
	double x;
	double y;
	double z;
};

struct BinaryMatrixTreeNode {

	int p = 0;
	double *U = NULL;
	double *VT = NULL;
	double *A = NULL;
	struct BinaryMatrixTreeNode *left;
	struct BinaryMatrixTreeNode *right;
};

typedef struct BinaryMatrixTreeNode mnode;

struct ComplexBinaryMatrixTreeNode {

	int p = 0;
	dtype *U = NULL;
	dtype *VT = NULL;
	dtype *A = NULL;
	struct ComplexBinaryMatrixTreeNode *left;
	struct ComplexBinaryMatrixTreeNode *right;
};

typedef struct ComplexBinaryMatrixTreeNode cmnode;

struct list {
	mnode* node;
	struct list* next;
};

struct my_queue {
	struct list *first, *last;
};

typedef struct list qlist;

#define PI 3.141592653589793238462643

//#define HELMHOLTZ

#ifdef HELMHOLTZ
#define LENGTH 900
#else
#define LENGTH 1
#endif

#define OUTPUT
#define GNUPLOT

#ifdef HELMHOLTZ
#define omega 4.0
#define c_z 1280
/*--------------*/
#define ky 1.8
#define beta_eq 2.3
#define kk (2.0 * (PI) * (omega) / (c_z))

//#define kk ((omega) / (c_z))
#else
#define omega 4.0
#define ky 0
#define beta_eq 1
#define c_z 0
#define kk 0
#endif


#define STRUCT_CSR

#ifdef STRUCT_CSR
#define ONLINE
#endif

#define FULL_SVD

//#define COL_UPDATE
//#define COL_ADD

// Функция выделения памяти под массив

template<typename T>
T* alloc_arr(int n)
{
	T *f = (T*)malloc(n * sizeof(T));

#pragma omp parallel for schedule(static)
	for (int i = 0; i < n; i++)
		f[i] = 0.0;

	return f;
}

template<typename T>
void free_arr(T* &arr)
{
	free(arr);
}






