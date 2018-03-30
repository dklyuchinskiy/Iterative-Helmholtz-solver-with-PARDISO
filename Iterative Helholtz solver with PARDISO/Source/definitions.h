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

typedef std::complex<double> dtype;
#define MKL_Complex16 dtype

#include "C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2018.1.156\windows\mkl\include\mkl.h"
#include "mkl_dfti.h"

//#define DEBUG

#define EPS 0.00000001

#define min(a,b) ((a) <= (b)) ? (a) : (b)

struct size_m {
	int l;
	int n;
	double h;
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

struct MatrixCSR {

	int *ia = NULL;
	int *ja = NULL;
	dtype *values = NULL;
};

typedef struct MatrixCSR dcsr;

struct list {
	mnode* node;
	struct list* next;
};

struct my_queue {
	struct list *first, *last;
};

typedef struct list qlist;

#define STRUCT_CSR

#ifdef STRUCT_CSR
#define ONLINE
#endif

#define FULL_SVD

//#define COL_UPDATE
//#define COL_ADD


#define PI 3.141592653589793238462643

// Функция выделения памяти под массив

template<typename T>
T* alloc_arr(int n)
{
	T *f = (T*)malloc(n * sizeof(T));

#pragma omp parallel for simd schedule(simd:static)
	for (int i = 0; i < n; i++)
		f[i] = 0.0;

	return f;
}

template<typename T>
void free_arr(T* &arr)
{
	free(arr);
}






