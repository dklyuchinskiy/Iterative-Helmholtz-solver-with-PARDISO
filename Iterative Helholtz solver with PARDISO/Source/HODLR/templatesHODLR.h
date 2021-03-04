#pragma once

/****************************
Prototypes for all functions.
****************************/
#include "../definitions.h"

using namespace std;

// functions.cpp
// HSS

void LowRankApprox(int n2, int n1, double *A , int lda, double *V , int ldv, int&p, double eps, char *method);
void DiagMult(int n, double *A, int lda, double *d, int small_size);

// Solver

void GenMatrixandRHSandSolution(const int n1, const int n2, const int n3, double *D, int ldd, double *B, double *x1, double *f);
void Block3DSPDSolveFast(int n1, int n2, int n3, double *D, int ldd, double *B, double *f, double thresh, int smallsize, int ItRef, char *bench,
	 double *G, int ldg, double *x, int &success, double &RelRes, int &itcount);
void DirFactFastDiag(int n1, int n2, int n3, double *D, int ldd, double *B, double *G, int ldg, double eps, int smallsize, char *bench);
void DirSolveFastDiag(int n1, int n2, int n3, double *G, int ldg, double *B, double *f, double *x, double eps, int smallsize);
void GenMatrixandRHSandSolution2(size_m x, size_m y, size_m z, double *D, int ldd, double *B, double *x1, double *f, double thresh);
void Test_NonZeroElementsInFactors(size_m x, size_m y, cmnode **Gstr, dtype* B, double thresh, int smallsize);

// Support

double random(double min, double max);
double F_ex(double x, double y, double z);
double u_ex(double x, double y, double z);
double F_ex_2D(size_m xx, size_m yy, double x, double y);
double u_ex_2D(size_m xx, size_m yy, double x, double y);
double rel_error(int n, int k, double *Hrec, double *Hinit, int ldh, double eps);
double c0(double x, double y);
dtype alph(size_m size, int xl, int xr, int i);
void GenRHSandSolution2D_Syntetic(size_m x, size_m y, zcsr *Dcsr, /* output */ dtype *u, dtype *f);
void Clear(int m, int n, dtype* DD, int lddd);

void Mat_Trans(int m, int n, dtype *H, int ldh, dtype *Hcomp_tr, int ldhtr);
void Hilbert(int m, int n, dtype *H, int ldh);
void op_mat(int n1, int n, double *Y11, double *Y12, int ldy, char sign);
void Add_dense(int m, int n, dtype alpha, dtype *A, int lda, dtype beta, dtype *B, int ldb, dtype *C, int ldc);
void Resid(int n1, int n2, int n3, double *D, int ldd, double *B, double *x, double *f, double *g, double &RelRes);
void print_map(const map <vector<int>, dtype>& SD);
void Eye(int n, dtype *H, int ldh);
void Diag(int n, dtype *H, int ldh, double value);
void Add_dense_vect(int n, double alpha, double *a, double beta, double *b, double *c);
void GenSolVector(int size, dtype *x1);
void DenseDiagMult(int n, dtype *diag, dtype *v, dtype *f);
void Mult_Au(int n1, int n2, int n3, double *D, int ldd, double *B, double *u, double *Au /*output*/);
void print(int m, int n, dtype *u, int ldu, char *mess);
void print_vec_mat(int m, int n, double *u, int ldu, double *vec, char *mess);
void print_vec(int size, double *vec1, double *vec2, char *name);
void print_vec(int size, int *vec1, double *vec2, char *name);

void print_vec_complex(int size, dtype *p2, char *name);

int compare_str(int n, char *s1, char *s2);

map<vector<int>, double> dense_to_sparse(int m, int n, double *DD, int ldd, int *i_ind, int *j_ind, double *d);
map<vector<int>, double> block3diag_to_CSR(int n1, int n2, int blocks, double *BL, int ldbl, double *A, int lda, double *BR, int ldbr, zcsr* Acsr);
map<vector<int>, double> concat_maps(const map<vector<int>, double>& map1, const map<vector<int>, double>& map2);

// BinaryTrees.cpp

int TreeSize(cmnode* root);
int MaxDepth(cmnode* Node);
void PrintRanks(mnode* root);
void PrintRanksInWidth(cmnode *root);
void CopyStruct(int n, cmnode* Gstr, cmnode* &TD1str, int smallsize);
void FreeNodes(int n, cmnode* &Astr, int smallsize);
void alloc_dense_node(int n, cmnode* &Cstr);
void PrintStruct(int n, cmnode *root);
int CountElementsInMatrixTree(int n, cmnode* root);

// BinaryTrees.cpp

// Solver
// Complex
void LowRankApproxStruct(int n2, int n1, dtype *A, int lda, cmnode* &Astr, double eps, char *method);
void SymRecCompressStruct(int n, dtype *A, const int lda, cmnode* &ACstr, const int smallsize, double eps, char *method);
void DiagMultStruct(int n, cmnode* Astr, dtype *d, int small_size);
void RecMultLStruct(int n, int m, cmnode* Astr, dtype *X, int ldx, dtype *Y, int ldy, int smallsize);
void RecMultLStructWork(int n, int m, cmnode* Astr, dtype *X, int ldx, dtype *Y, int ldy, dtype *work1, int lwork1, dtype *work2, int lwork2, int smallsize);
void AddStruct(int n, dtype alpha, cmnode* Astr, dtype beta, cmnode* Bstr, cmnode* &Cstr, int smallsize, double eps, char *method);
void SymUpdate4Subroutine(int n2, int n1, dtype alpha, cmnode* Astr, const dtype *Y, int ldy, cmnode* &Bstr, int smallsize, double eps, char* method);
void SymCompUpdate2Struct(int n, int k, cmnode* Astr, dtype alpha, dtype *Y, int ldy, dtype *V, int ldv, cmnode* &Bstr, int smallsize, double eps, char* method);
void SymCompUpdate4LowRankStruct(int n, int k1, int k2, cumnode* Astr, dtype alpha, dtype *Y, int ldy, dtype *V1, int ldv1, dtype *V2, int ldv2, cumnode* &Bstr, int smallsize, double eps, char* method);
void SymCompRecInvStruct(int n, cmnode* Astr, cmnode* &Bstr, int smallsize, double eps, char *method);
void SymResRestoreStruct(int n, cmnode* H1str, dtype *H2, int ldh, int smallsize);
void SymLUfactLowRankStruct(int n, cumnode* Astr, int *ipiv, int smallsize, double eps, char* method);
double rel_error_complex(int n, int k, dtype *Hrec, dtype *Hinit, int ldh, double eps);
void Hilbert2(int m, int n, dtype *H, int ldh);
void Hilbert3(int m, int n, dtype *H, int ldh);
void Hilbert4(int m, int n, dtype *H, int ldh);
void Hilbert5(int m, int n, dtype *H, int ldh);
void Hilbert6(int m, int n, dtype *H, int ldh);
void Hilbert7LowRank(int m, int n, dtype *H, int ldh);
void alloc_dense_simple_node(int n, cmnode* &Cstr);

// Unsymm
void UnsymmLUfact(int n, cumnode* Astr, int *ipiv, int smallsize, double eps, char* method);
void UnsymmRecMultLStruct(int n, int m, cumnode* Astr, dtype *X, int ldx, dtype *Y, int ldy, int smallsize);
void UnsymmRecMultRStruct(int n, int m, cumnode* Astr, dtype *X, int ldx, dtype *Y, int ldy, int smallsize);
void UnsymmUpdate2Subroutine(int n2, int n1, int k, cmnode* Astr, dtype alpha, dtype *Y, int ldy, dtype *V1, int ldv1, dtype *V2, int ldv2, cmnode* &Bstr, int smallsize, double eps, char* method);
void UnsymmCompUpdate2Struct(int n, int k, cumnode* Astr, dtype alpha, dtype *Y, int ldy, dtype *V, int ldv, cumnode* &Bstr, int smallsize, double eps, char* method);
void UnsymmCompRecInvStruct(int n, cumnode* Astr, cumnode* &Bstr, int smallsize, double eps, char *method);
void UnsymmRecCompressStruct(int n, dtype *A, const int lda, cumnode* &ACstr, const int smallsize, double eps, char *method);
void UnsymmResRestoreStruct(int n, cumnode* H1str, dtype *H2, int ldh, int smallsize);
void UnsymmAddStruct(int n, dtype alpha, cumnode* Astr, dtype beta, cumnode* Bstr, cumnode* &Cstr, int smallsize, double eps, char *method);
void UnsymmAddSubroutine(int n2, int n1, dtype alpha, cmnode* Astr, dtype beta, cmnode* Bstr, cmnode* &Cstr, int smallsize, double eps, char *method);
void UnsymmUpdate3Subroutine(int n2, int n1, int k1, int k2, cmnode* Astr, dtype alpha, dtype *Y, int ldy, dtype *V1, int ldv1, dtype* V2, int ldv2, cmnode* &Bstr, int smallsize, double eps, char* method);
void UnsymmCompUpdate3Struct(int n, int k1, int k2, cumnode* Astr, dtype alpha, dtype *Y, int ldy, dtype *V1, int ldv1, dtype *V2, int ldv2, cumnode* &Bstr, int smallsize, double eps, char* method);
void UnsymmCopyStruct(int n, cumnode* Astr, cumnode* Bstr, int smallsize);
void CopyLfactor(int n, cumnode* Astr, cumnode* &Lstr, int smallsize);
void CopyRfactor(int n, cumnode* Astr, cumnode* &Rstr, int smallsize);
void CopyUnsymmStruct(int n, cumnode* Astr, cumnode* &Bstr, int smallsize);
void ApplyToA21(int n, cmnode* Astr, cumnode* R, int smallsize, double eps, char *method);
void ApplyToA12(int n, cmnode* Astr, cumnode* L, int smallsize, double eps, char *method);
void ApplyToA21Ver2(int p, int n, dtype* VT, int ldvt, cumnode* R, int smallsize, double eps, char *method);
void ApplyToA12Ver2(int n, int p, dtype* U, int ldu, cumnode* L, int smallsize, double eps, char *method);
void MyLURec(int n, dtype *Hinit, int ldh, int *ipiv, int smallsize);
void UnsymmCompRecInvLowerTriangStruct(int n, cumnode* Lstr, cumnode* &Bstr, int smallsize, double eps, char *method);
void UnsymmCompRecInvUpperTriangStruct(int n, cumnode* Ustr, cumnode* &Bstr, int smallsize, double eps, char *method);
void UnsymmRecMultUpperRStruct(int n, int m, cumnode* Astr, dtype *X, int ldx, dtype *Y, int ldy, int smallsize);
void UnsymmRecMultUpperLStruct(int n, int m, cumnode* Astr, dtype *X, int ldx, dtype *Y, int ldy, int smallsize);
void UnsymmRecMultLowerLStruct(int n, int m, cumnode* Astr, dtype *X, int ldx, dtype *Y, int ldy, int smallsize);
void UnsymmRecMultLowerRStruct(int n, int m, cumnode* Astr, dtype *X, int ldx, dtype *Y, int ldy, int smallsize);
void LowRankToUnsymmHSS(int n, int r, dtype *U, int ldu, dtype *VT, int ldvt, cumnode *&Aout, int smallsize);
void LowRankToSymmHSS(int n, int r, dtype *U, int ldu, dtype *VT, int ldvt, cmnode *&Aout, int smallsize);
void LowRankCholeskyFact(int n, cmnode* Astr, dtype* work, int smallsize, double eps, char* method);
void AddSymmHSSDiag(int n, cmnode *Aout, dtype *Diag, int smallsize);
void SolveTriangSystemA21(int p, int n, dtype* VT, int ldvt, cmnode* R, int smallsize, double eps, char *method);
void SymCompUpdate5LowRankStruct(int n, int k, cmnode* Astr, dtype alpha, dtype *Y, int ldy, dtype *V, int ldv, int smallsize, double eps, char* method);
void LowRankToSymmHSS(int n, int r, dtype *U, int ldu, cmnode *&Aout, int smallsize);

void alloc_dense_unsymm_node(int n, cumnode* &Cstr);
void FreeUnsymmNodes(int n, cumnode* &Astr, int smallsize);
void UnsymmClearStruct(int n, cumnode* Astr, int smallsize);
int my_log(int a, int b);
int GetNumberOfLeaves(cumnode *root);
void GetDistances(cumnode *root, int *dist, int &count);
void Hilbert8Unique(int m, int n, dtype *H, int ldh);
void MakeFullDenseSymMatrix(char part, int n, dtype *A, int lda);
void PrintMat(int m, int n, dtype *A, int lda);
void CholeskyFact(int n, dtype* A, int lda, int smallsize, double eps, char* method);
void SymUpdate5Subroutine(int n2, int n1, dtype alpha, cmnode* Astr, const dtype *Y, int ldy, dtype *V, int ldv, cmnode* &Bstr, int smallsize, double eps, char* method);


// Solver
void Block3DSPDSolveFastStruct(size_m x, size_m y, dtype *D, int ldd, dtype *B, dtype *f, zcsr* Dcsr, double thresh, int smallsize, int ItRef, char *bench,
	cmnode** &Gstr, dtype *x_sol, int &success, double &RelRes, int &itcount, double beta_eq);
void DirFactFastDiagStructOnline(size_m x, size_m y, cmnode** &Gstr, dtype *B, dtype *sound2D, double kww, double beta_eq, dtype *work, int lwork, double eps, int smallsize);
void DirSolveFastDiagStruct(int n1, int n2, cmnode* *Gstr, dtype *B, const dtype *f, dtype *x, dtype *work, int lwork, double eps, int smallsize);
void GenerateDiagonal1DBlockHODLR(int j, size_m x, size_m y, dtype *DD, int lddd, dtype *sound2D, double kww, double beta_eq);

void GenerateDiagonal2DBlock(int part_of_field, size_m x, size_m y, size_m z, dtype *DD, int lddd);

void DirFactFastDiagStruct(int n1, int n2, int n3, double *D, int ldd, double *B, mnode** &Gstr, 
	double eps, int smallsize, char *bench);

void DiagVec(int n, dtype *H, int ldh, dtype *value);
void ResidCSR(int n1, int n2, zcsr* Dcsr, dtype* x_sol, dtype *f, dtype* g, double &RelRes);
void GenSparseMatrix(size_m x, size_m y, size_m z, double *BL, int ldbl, double *A, int lda, double *BR, int ldbr, zcsr* Acsr);
void GenerateDiagonal1DBlock(int part_of_field, size_m x, size_m y, dtype *DD, int lddd, dtype *alpX, dtype* alpY);
void GenRHSandSolution2D(size_m x, size_m y, /* output */ dtype *u, dtype *f);
void GenRHSandSolution3D(size_m x, size_m y, size_m z, dtype* B, dtype *u, dtype *f);
void GenSparseMatrixOnline(size_m x, size_m y, size_m z, dtype *BL, int ldbl, dtype *A, int lda, dtype *BR, int ldbr, zcsr* Acsr);
void GenSparseMatrixOnline2D(size_m x, size_m y, dtype *B, dtype *BL, int ldbl, dtype *A, int lda, dtype *BR, int ldbr, zcsr* Acsr);
map<vector<int>, dtype> Block1DRowMat_to_CSR(int blk, int n1, int n2, dtype *BL, int ldbl, dtype *A, int lda, dtype *BR, int ldbr, zcsr* Acsr, int& non_zeros_on_prev_level);
map<vector<int>, dtype> dense_to_CSR(int m, int n, dtype *A, int lda, int *ia, int *ja, dtype *values);
map<vector<int>, dtype> BlockRowMat_to_CSR(int blk, int n1, int n2, int n3, dtype *BL, int ldbl, dtype *A, int lda, dtype *BR, int ldbr, zcsr* Acsr, int& non_zeros_on_prev_level);
void construct_block_row(int m, int n, dtype* BL, int ldbl, dtype *A, int lda, dtype *BR, int ldbr, dtype* AR, int ldar);
void shift_values(int rows, int *ia, int shift_non_zeros, int non_zeros, int *ja, int shift_columns);
void SetPml(int blk, size_m x, size_m y, int n, dtype* alpX, dtype* alpY);
void MyLU(int n, dtype *Hinit, int ldh, int *ipiv);
void Hilbert5(int m, int n, dtype *H, int ldh);
void GenerateSubdiagonalB(size_m x, size_m y, dtype *B);

void count_dense_elements(int m, int n, double *A, int lda, int& non_zeros);
void compare_vec(int size, dtype* v1, dtype* v2);

// Queue
void init(struct my_queue* &q);
bool my_empty(struct my_queue* q);
void push(struct my_queue* &q, cmnode* node);
void pop(struct my_queue* &q);
cmnode* front(struct my_queue* q);
void PrintRanksInWidthList(cmnode *root);
void print_queue(struct my_queue* q);

void init(struct my_queue2* &q);
bool my_empty(struct my_queue2* q);
void push(struct my_queue2* &q, cumnode* node);
void pop(struct my_queue2* &q);
cumnode* front(struct my_queue2* q);
void UnsymmPrintRanksInWidthList(cumnode *root);
void UnsymmPrintRanksInWidth(cumnode *root);







