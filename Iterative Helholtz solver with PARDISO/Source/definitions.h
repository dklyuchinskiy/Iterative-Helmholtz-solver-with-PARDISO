#pragma once

/*****************************
Preprocessor definitions and
declaration of used structures
*****************************/
#include "libraries.h"

typedef double rtype;
typedef std::complex<double> dtype;
#define MKL_Complex16 dtype

#if defined(_WIN32) || defined(WIN32)
#include "C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2018\windows\mkl\include\mkl.h"
#include "C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2018\windows\mkl\include\mkl_dfti.h"
#include "C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2018\windows\mkl\include\mkl_rci.h"
#else
#include "mkl.h"
#include "mkl_dfti.h"
#include "mkl_rci.h"
#endif

//#define DEBUG

#define EPS 0.00001

struct MatrixCSRReal {

	int *ia = NULL;
	int *ja = NULL;
	double *values = NULL;
};

typedef struct MatrixCSRReal dcsr;

struct MatrixCSRComplex 
{
	int *ia = NULL;
	int *ja = NULL;
	dtype *values = NULL;
	int non_zeros = 0;
	int solve = 0;
};

typedef struct MatrixCSRComplex ccsr;


struct size_m 
{
	double l;
	int n;
	double h;
	int pml_pts;
	int spg_pts;
	int n_nopml;
};

struct point 
{
	double x;
	double y;
	double z;
};

struct BinaryMatrixTreeNode 
{
	int p = 0;
	double *U = NULL;
	double *VT = NULL;
	double *A = NULL;
	struct BinaryMatrixTreeNode *left;
	struct BinaryMatrixTreeNode *right;
};

typedef struct BinaryMatrixTreeNode mnode;

struct ComplexBinaryMatrixTreeNode 
{
	int p = 0;
	dtype *U = NULL;
	dtype *VT = NULL;
	dtype *A = NULL;
	struct ComplexBinaryMatrixTreeNode *left;
	struct ComplexBinaryMatrixTreeNode *right;
};

typedef struct ComplexBinaryMatrixTreeNode cmnode;

struct list 
{
	mnode* node;
	struct list* next;
};

struct my_queue 
{
	struct list *first, *last;
};

typedef struct list qlist;

struct matrix
{
	int i;
	int j;
	dtype val;
};

enum class DIAG13
{	
	m_six = 1,
	m_five,
	m_four,
	m_three,
	m_two,
	m_one,
	zero,
	one,
	two,
	three,
	four,
	five,
	six
};

#define PI 3.141592653589793238462643

#define HELMHOLTZ
#define PML
#define GMRES_SIZE 128

#ifdef HELMHOLTZ
#ifdef PML
//#define LENGTH 900
//#define LENGTH 1200
#define LENGTH 3200

//#define LENGTH 8
#define PERF


#if 0
#define LENGTH_X 5120
#define LENGTH_Y 5120
#define LENGTH_Z 2560
#else
#define LENGTH_X LENGTH
#define LENGTH_Y LENGTH
#define LENGTH_Z LENGTH
#endif

#if 0
#define LENGTH_X 3200
#define LENGTH_Y 3200
#define LENGTH_Z 3200
#endif
//#define LENGTH 200
#else
#define LENGTH 1500
#endif
#else
#define LENGTH 1
#endif

//#define GEN_BLOCK_CSR
//#define SOLVE_3D_PROBLEM

//#define OUTPUT
//#define GNUPLOT
//#define GEN_3D_MATRIX


//#define TEST_HELM_1D

#ifdef HELMHOLTZ
#define nu 2
#define c_z 1280.0
/*--------------*/
#define ky 1.8
//#define beta_eq 0.005
#define beta_eq 0.2

#define omega 2.0 * (PI) * (nu)
#define kk ((omega) / (c_z))

//#define kk ((omega) / (c_z))
#else
#define omega 4.0
#define ky 0
#define beta_eq 1
#define c_z 0
#define kk 0
#endif


#define EPS_ZERO 0.00000001

#define STRUCT_CSR

#ifdef STRUCT_CSR
#define ONLINE
#endif

#define FULL_SVD

//#define COL_UPDATE
//#define COL_ADD

// Функция выделения памяти под массив

template<typename T>
T* alloc_arr(long long int n)
{
//	printf("%lli\n", n);
	T *f = (T*)malloc(n * sizeof(T));

#pragma omp parallel for schedule(static)
	for (long long int i = 0; i < n; i++)
		f[i] = 0.0;

	return f;
}

template<typename T>
void free_arr(T* &arr)
{
	free(arr);
}

template<typename T>
void MultVectorConst(int n, T* v1, T alpha, T* v2)
{
#pragma omp parallel for simd schedule(simd:static)
	for (int i = 0; i < n; i++)
		v2[i] = v1[i] * alpha;
}





