#pragma once

/****************************
Prototypes for all functions.
****************************/

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


// Support

double random(double min, double max);
double F_ex(size_m xx, size_m yy, size_m zz, double x, double y, double z);
double u_ex(size_m xx, size_m yy, size_m zz, double x, double y, double z);
//double rel_error(int n, int k, double *Hrec, double *Hinit, int ldh, double eps);

void Mat_Trans(int m, int n, dtype *H, int ldh, dtype *Hcomp_tr, int ldhtr);
void Hilbert(int n, dtype *H, int ldh);
void op_mat(int n1, int n, double *Y11, double *Y12, int ldy, char sign);
void Add_dense(int m, int n, dtype alpha, dtype *A, int lda, dtype beta, dtype *B, int ldb, dtype *C, int ldc);
void Resid(int n1, int n2, int n3, double *D, int ldd, double *B, double *x, double *f, double *g, double &RelRes);
void print_map(const map<vector<int>, double>& SD);
void Eye(int n, dtype *H, int ldh);
//void Diag(int n, dtype *H, int ldh, rtype value);
void Add_dense_vect(int n, double alpha, double *a, double beta, double *b, double *c);
void DenseDiagMult(int n, dtype *diag, dtype *v, dtype *f);
void Mult_Au(int n1, int n2, int n3, double *D, int ldd, double *B, double *u, double *Au /*output*/);
void print(int m, int n, dtype *u, int ldu, char *mess);
void print_vec_mat(int m, int n, double *u, int ldu, double *vec, char *mess);
void print_vec(int size, double *vec1, double *vec2, char *name);
void print_vec(int size, int *vec1, double *vec2, char *name);

void print_map(const map<vector<int>, dtype>& SD);
void print_csr(int n, dcsr* A);
int compare_str(int n, char *s1, char *s2);
int ind(int j, int n);

map<vector<int>, double> dense_to_sparse(int m, int n, double *DD, int ldd, int *i_ind, int *j_ind, double *d);
map<vector<int>, double> block3diag_to_CSR(int n1, int n2, int blocks, double *BL, int ldbl, double *A, int lda, double *BR, int ldbr, dcsr* Acsr);
map<vector<int>, double> concat_maps(const map<vector<int>, double>& map1, const map<vector<int>, double>& map2);

// BinaryTrees.cpp

int TreeSize(mnode* root);
int MaxDepth(mnode* Node);
void PrintRanks(mnode* root);
void PrintRanksInWidth(mnode *root);
void CopyStruct(int n, cmnode* Gstr, cmnode* &TD1str, int smallsize);
void FreeNodes(int n, cmnode* &Astr, int smallsize);
void alloc_dense_node(int n, cmnode* &Cstr);
void PrintStruct(int n, mnode *root);

// BinaryTrees.cpp

// Solver
// Complex
void LowRankApproxStruct(int n2, int n1, dtype *A, int lda, cmnode* &Astr, double eps, char *method);
void SymRecCompressStruct(int n, dtype *A, const int lda, cmnode* &ACstr, const int smallsize, double eps, char *method);
void DiagMultStruct(int n, cmnode* Astr, dtype *d, int small_size);
void RecMultLStruct(int n, int m, cmnode* Astr, dtype *X, int ldx, dtype *Y, int ldy, int smallsize);
void AddStruct(int n, dtype alpha, cmnode* Astr, dtype beta, cmnode* Bstr, cmnode* &Cstr, int smallsize, double eps, char *method);
void SymCompUpdate2Struct(int n, int k, cmnode* Astr, dtype alpha, dtype *Y, int ldy, dtype *V, int ldv, cmnode* &Bstr, int smallsize, double eps, char* method);
void SymCompRecInvStruct(int n, cmnode* Astr, cmnode* &Bstr, int smallsize, double eps, char *method);
void SymResRestoreStruct(int n, cmnode* H1str, dtype *H2, int ldh, int smallsize);
//double rel_error_complex(int n, int k, dtype *Hrec, dtype *Hinit, int ldh, double eps);

// Solver
void Block3DSPDSolveFastStruct(size_m x, size_m y, size_m z, dtype *D, int ldd, dtype *B, dtype *f, dcsr* Dcsr, double thresh, int smallsize, int ItRef, char *bench,
	cmnode** &Gstr, dtype *x_sol, int &success, double &RelRes, int &itcount);
void DirFactFastDiagStructOnline(size_m x, size_m y, size_m z, cmnode** &Gstr, dtype *B, double thresh, int smallsize, char *bench);
void DirSolveFastDiagStruct(int n1, int n2, int n3, cmnode* *Gstr, dtype *B, dtype *f, dtype *x, double eps, int smallsize);

void GenerateDiagonal2DBlock(int part_of_field, size_m x, size_m y, size_m z, dtype *DD, int lddd);
void GenerateDiagonal1DBlock(double w, int part_of_field, size_m y, size_m z, dtype *DD, int lddd);

void DirFactFastDiagStruct(int n1, int n2, int n3, double *D, int ldd, double *B, mnode** &Gstr, 
	double eps, int smallsize, char *bench);


void ResidCSR(int n1, int n2, int n3, ccsr* Dcsr, dtype* x_sol, dtype *f, dtype* g, double &RelRes);
void GenSparseMatrix(size_m x, size_m y, size_m z, double *BL, int ldbl, double *A, int lda, double *BR, int ldbr, dcsr* Acsr);

void GenRHSandSolution(size_m x, size_m y, size_m z, dtype* B, dtype *u, dtype *f);
void GenSparseMatrixOnline2D(int w, size_m y, size_m z, dtype *BL, int ldbl, dtype *A, int lda, dtype *BR, int ldbr, ccsr* Acsr);
void GenSparseMatrixOnline3D(size_m x, size_m y, size_m z, dtype *BL, int ldbl, dtype *A, int lda, dtype *BR, int ldbr, ccsr* Acsr);
map<vector<int>, dtype> Block1DRowMat_to_CSR(int blk, int n1, int n2, dtype *BL, int ldbl, dtype *A, int lda, dtype *BR, int ldbr, ccsr* Acsr, int& non_zeros_on_prev_level);
void GenRhs2D(int w, size_m x, size_m y, size_m z, dtype* f, dtype* f2D);
void Clear(int m, int n, dtype* A, int lda);
void GenSol1DBackward(int w, size_m x, size_m y, size_m z, dtype* x_sol_prd, dtype *u1D);

//map<vector<int>, dtype> dense_to_CSR(int m, int n, dtype *A, int lda, int *ia, int *ja, dtype *values);
map<vector<int>, dtype> BlockRowMat_to_CSR(int blk, int n1, int n2, int n3, dtype *BL, int ldbl, dtype *A, int lda, dtype *BR, int ldbr, ccsr* Acsr, int& non_zeros_on_prev_level);
//void construct_block_row(int m, int n, dtype* BL, int ldbl, dtype *A, int lda, dtype *BR, int ldbr, dtype* AR, int ldar);
void shift_values(int rows, int *ia, int shift_non_zeros, int non_zeros, int *ja, int shift_columns);

void count_dense_elements(int m, int n, double *A, int lda, int& non_zeros);
void SolvePardiso3D(size_m x, size_m y, size_m z, ccsr* Dcsr, dtype* x_pard, dtype* f, double thresh);
dtype my_exp(double val);

// Queue
void init(struct my_queue* &q);
bool my_empty(struct my_queue* q);
void push(struct my_queue* &q, mnode* node);
void pop(struct my_queue* &q);
mnode* front(struct my_queue* q);
void PrintRanksInWidthList(mnode *root);
void print_queue(struct my_queue* q);

//FFT
void MyFFT1D_ForwardReal(int n, double *f, dtype*f_MYFFT);
void MyFFT1D_ForwardComplex(int N, dtype* f, dtype *f_MYFFT);
void MyFFT1D_BackwardReal(int N, dtype *f_MYFFT, double* f);
void MyFFT1D_BackwardComplex(int N, dtype *f_MYFFT, dtype* f);
void MyFFT1D_BackwardComplexSin(int N, dtype* f, dtype *f_MYFFT);
void MyFFT1D_ForwardComplexSin(int N, dtype* f, dtype *f_MYFFT);

// FT

void MyFT1D_ForwardReal(int N, double h, double* f, dtype *f_MYFFT);
void MyFT1D_BackwardReal(int N, double h, dtype *f_MYFFT, double* f);
void MyFT1D_BackwardComplex(int N, double h, dtype *f_MYFFT, dtype* f);
void MyFT1D_ForwardComplex(int N, double h, dtype* f, dtype *f_MYFFT);


//
void GenSolVector(int size, dtype *vector);
void GenRHSandSolution2D_Syntetic(size_m x, size_m y, ccsr *Dcsr, dtype *u, dtype *f);








