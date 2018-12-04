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
void AddDenseVectors(int n, double alpha, double *a, double beta, double *b, double *c);
void AddDenseVectorsComplex(int n, dtype alpha, dtype *a, dtype beta, dtype *b, dtype *c);
void MultVectorConst(int n, dtype* v1, dtype alpha, dtype* v2);
void Resid(int n1, int n2, int n3, double *D, int ldd, double *B, double *x, double *f, double *g, double &RelRes);
void print_map(const map<vector<int>, double>& SD);
void Eye(int n, dtype *H, int ldh);
//void Diag(int n, dtype *H, int ldh, rtype value);
void DenseDiagMult(int n, dtype *diag, dtype *v, dtype *f);
void Mult_Au(int n1, int n2, int n3, double *D, int ldd, double *B, double *u, double *Au /*output*/);
void print(int m, int n, dtype *u, int ldu, char *mess);
void print_vec_mat(int m, int n, double *u, int ldu, double *vec, char *mess);
void print_vec(int size, double *vec1, double *vec2, char *name);
void print_vec(int size, int *vec1, double *vec2, char *name);
void NormalizeVector(int size, dtype* v, dtype* out, double& norm);
void GenRHSandSolutionViaSound3D(size_m x, size_m y, size_m z, /* output */ dtype *u, dtype *f, point source);
dtype u_ex_complex_sound3D(size_m xx, size_m yy, size_m zz, double x, double y, double z, point source);
void FGMRES(size_m x, size_m y, size_m z, int m, const point source, dtype *x_sol, const dtype *f, double thresh);
void check_norm_circle(size_m x, size_m y, size_m z, dtype* x_orig_nopml, dtype* x_sol_nopml, point source, double thresh);
void print_2Dcsr_mat(size_m x, size_m y, ccsr* D2csr);
void print_2Dcsr_mat2(size_m x, size_m y, ccsr* D2csr);
void check_test_3Dsolution_in1D(int n1, int n2, int n3, dtype* u_sol, dtype *u_ex, double thresh);
void SetRHS3DForTest(size_m xx, size_m yy, size_m zz, dtype* f, point source, int& l);

void print_map(const map<vector<int>, dtype>& SD);
void print_csr(int n, dcsr* A);
int compare_str(int n, char *s1, char *s2);
int ind(int j, int n);
void take_coord3D(int n1, int n2, int n3, int l, int& i, int& j, int& k);
void take_coord2D(int n1, int n2, int l, int& i, int& j);

map<vector<int>, double> dense_to_sparse(int m, int n, double *DD, int ldd, int *i_ind, int *j_ind, double *d);
map<vector<int>, double> block3diag_to_CSR(int n1, int n2, int blocks, double *BL, int ldbl, double *A, int lda, double *BR, int ldbr, dcsr* Acsr);
map<vector<int>, double> concat_maps(const map<vector<int>, double>& map1, const map<vector<int>, double>& map2);

dtype zdot(int size, dtype* v1, dtype* v2);
void ComputeResidual(size_m x, size_m y, size_m z, double kw, const dtype* u, const dtype *f, dtype* f_res, double &RelRes);

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

void GenerateDiagonal2DBlock(char *str, int part_of_field, size_m x, size_m y, size_m z, dtype *DD, int lddd, dtype *alpX, dtype* alpY, dtype *alpZ);
void GenerateDiagonal1DBlock(double w, int part_of_field, size_m x, size_m y, size_m z, dtype *DD, int lddd, dtype *alpX, dtype* alpY, dtype *alpZ);

void SetPml3D(int blk3D, size_m x, size_m y, size_m z, int n, dtype* alpX, dtype* alpY, dtype* alpZ);
void SetPml2D(int blk3D, int blk2D, size_m x, size_m y, size_m z, int n, dtype* alpX, dtype* alpY, dtype *alpZ);

void DiagVec(int n, dtype *H, int ldh, dtype *value);

void DirFactFastDiagStruct(int n1, int n2, int n3, double *D, int ldd, double *B, mnode** &Gstr, 
	double eps, int smallsize, char *bench);


void ResidCSR(size_m x, size_m y, size_m z, ccsr* Dcsr, dtype* x_sol, dtype *f, dtype* g, double &RelRes);
void GenSparseMatrix(size_m x, size_m y, size_m z, double *BL, int ldbl, double *A, int lda, double *BR, int ldbr, dcsr* Acsr);

void GenRHSandSolution(size_m x, size_m y, size_m z, dtype *u, dtype *f, point source, int &l);
void GenSparseMatrixOnline2D(char *str, int w, size_m x, size_m y, size_m z, dtype *BL, int ldbl, dtype *A, int lda, dtype *BR, int ldbr, ccsr* Acsr);
void GenSparseMatrixOnline3D(size_m x, size_m y, size_m z, dtype* B, dtype *BL, int ldbl, dtype *A, int lda, dtype *BR, int ldbr, ccsr* Acsr);
void GenSparseMatrixOnline2DwithPML(int i, size_m x, size_m y, size_m z, ccsr* Acsr, dtype kwave2, int* freqs);
void GenSparseMatrixOnline3DwithPML(size_m x, size_m y, size_m z, dtype* B, dtype *BL, int ldbl, dtype *A, int lda, dtype *BR, int ldbr, ccsr* Acsr, double eps);
map<vector<int>, dtype> Block1DRowMat_to_CSR(int blk, int n1, int n2, dtype *BL, int ldbl, dtype *A, int lda, dtype *BR, int ldbr, ccsr* Acsr, int& non_zeros_on_prev_level);
void GenRhs2D(int w, size_m x, size_m y, size_m z, dtype* f, dtype* f2D);
void Clear(int m, int n, dtype* A, int lda);
void GenSol1DBackward(int w, size_m x, size_m y, size_m z, dtype* x_sol_prd, dtype *u1D);
void reducePML3D(size_m x, size_m y, size_m z, int size1, dtype *vect, int size2, dtype *vect_red);
void reducePML2D(size_m x, size_m y, int size1, dtype *vect, int size2, dtype *vect_red);
void reducePML1D(size_m x, int size1, dtype *vect, int size2, dtype *vect_red);
void reducePML3D_FT(size_m x, size_m y, size_m z, int size1, dtype *vect, int size2, dtype *vect_red);

//map<vector<int>, dtype> dense_to_CSR(int m, int n, dtype *A, int lda, int *ia, int *ja, dtype *values);
map<vector<int>, dtype> BlockRowMat_to_CSR(int blk, int n1, int n2, int n3, dtype *BL, int ldbl, dtype *A, int lda, dtype *BR, int ldbr, ccsr* Acsr, int& non_zeros_on_prev_level);
//void construct_block_row(int m, int n, dtype* BL, int ldbl, dtype *A, int lda, dtype *BR, int ldbr, dtype* AR, int ldar);
void shift_values(int rows, int *ia, int shift_non_zeros, int non_zeros, int *ja, int shift_columns);

void count_dense_elements(int m, int n, double *A, int lda, int& non_zeros);
void SolvePardiso3D(size_m x, size_m y, size_m z, ccsr* Dcsr, dtype* x_pard, dtype* f, double thresh);
dtype my_exp(double val);
dtype EulerExp(dtype val);
dtype u_ex_complex(size_m xx, size_m yy, size_m zz, double x, double y, double z, point source);
dtype F3D_ex_complex(size_m xx, size_m yy, size_m zz, double x, double y, double z, point source, int& l);
dtype F2D_ex_complex(size_m xx, size_m yy, double x, double y, point source, int& l);
dtype F1D_ex_complex(size_m xx, double x, point source, int& l);
void output(char *str, bool pml_flag, size_m x, size_m y, size_m z, dtype* x_orig, dtype* x_pard);
void gnuplot(char *str1, char *str2, bool pml_flag, int col, size_m x, size_m y, size_m z);
void output2D(char *str, bool pml_flag, size_m x, size_m y, dtype* x_orig, dtype* x_pard);
void gnuplot2D(char *splot, char *sout, bool pml_flag, int col, size_m x, size_m y);
void gnuplot1D(char *splot, char *sout, bool pml_flag, int col, size_m x);
dtype alpha(size_m xyz, double i);
dtype beta(size_m, size_m y, size_m z, int diag_case, int i, int j, int k);
void check_exact_sol_Hankel(dtype alpha_k, double k2, size_m y, size_m z, dtype* x_sol_prd, double eps);
dtype Hankel(double x);
dtype Hankel(dtype z);
void get_exact_2D_Hankel(int Nx, int Ny, size_m x, size_m y, dtype* x_sol_ex, dtype k, point source);
double resid_2D_Hankel(size_m y, size_m z, ccsr* D2csr, dtype* x_sol_ex, dtype* f2D, point source);
void ResidCSR2D(size_m y, size_m z, ccsr* Dcsr, dtype* x_sol, dtype *f, dtype* g, point source, double &RelRes);
void normalization_of_exact_sol(int n1, int n2, size_m x, size_m y, dtype *x_sol_ex, dtype alpha_k);
void check_norm_result(int n1, int n2, int n3, dtype* x_orig_no_pml, dtype* x_sol);
dtype set_exact_2D_Hankel(double x, double y, dtype k, point source);
void extendPML3D(size_m x, size_m y, size_m z, int size1, dtype *vect, int size2, dtype *vect_ext);
void SetSoundSpeed3D(size_m x, size_m y, size_m z, dtype* sound, point source);
void SetSoundSpeed2D(size_m x, size_m y, size_m z, dtype* sound3D, dtype* sound2D, point source);
dtype MakeSound3D(size_m xx, size_m yy, size_m zz, double x, double y, double z, point source);
dtype MakeSound2D(size_m xx, size_m yy, double x, double y, point source);
void GenerateDeltaL(size_m x, size_m y, size_m z, dtype* sound3D, dtype* sound2D, dtype* deltaL);
void Solve1DSparseHelmholtz(size_m x, size_m y, size_m z, dtype *f1D, dtype *x_sol1D, double thresh);
void Solve2DSparseHelmholtz(size_m x, size_m y, size_m z, dtype *f2D, dtype *x_sol2D, double thresh);
dtype beta2D(size_m x, size_m y, int diag_case, int i, int j);
dtype beta1D(size_m x, int diag_case, double k2, int i);
void NullifySource2D(size_m x, size_m y, dtype *u, int src, int npoints);
void check_norm_result2(int n1, int n2, int n3, double ppw, double spg, dtype* x_orig_nopml, dtype* x_sol_nopml,
	double* x_orig_re, double* x_orig_im, double *x_sol_re, double *x_sol_im);

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
void MyFT1D_BackwardComplex(int N, size_m x, dtype *f_MYFFT, dtype* f);
void MyFT1D_ForwardComplex(int N, size_m x, dtype* f, dtype *f_MYFFT);


//
void GenSolVector(int size, dtype *vector);
void GenRHSandSolution2D_Syntetic(size_m x, size_m y, ccsr *Dcsr, dtype *u, dtype *f);
void GenExact1DHelmholtz(int n, size_m x, dtype *x_sol_ex, double k, point sourcePML);

//
void Solve3DSparseUsingFT(size_m x, size_m y, size_m z, const dtype *f, dtype* x_sol, double thresh);
void ApplyCoeffMatrixA(size_m x, size_m y, size_m z, const dtype *w, const dtype* deltaL, dtype* g, double thresh);
void OpTwoMatrices(int m, int n, const dtype *Y1, const dtype *Y2, dtype *Yres, int ldy, char sign);
void SetRHS1D(size_m xx, dtype* fD, point source, int& l);
void SetRHS2D(size_m xx, size_m yy, dtype* fD, point source, int& l);
void SetRHS3D(size_m xx, size_m yy, size_m zz, dtype* fD, point source, int& l);



// Bessel and Hankel functions
void ZBESI(REAL ZR, REAL ZI, REAL FNU, int KODE, int N, REAL *CYR, REAL *CYI, int *NZ, int *IERR);
void ZBESJ(REAL ZR, REAL ZI, REAL FNU, int KODE, int N, REAL *CYR, REAL *CYI, int *NZ, int *IERR);
void ZBESK(REAL ZR, REAL ZI, REAL FNU, int KODE, int N, REAL *CYR, REAL *CYI, int *NZ, int *IERR);
void ZBESY(REAL ZR, REAL ZI, REAL FNU, int KODE, int N, REAL *CYR, REAL *CYI, int *NZ, REAL *CWRKR, REAL *CWRKI, int *IERR);
void ZBESH(REAL ZR, REAL ZI, REAL FNU, int KODE, int M, int N, REAL *CYR, REAL *CYI, int *NZ, int *IERR);

void ZUOIK(REAL, REAL, REAL, int, int, int, REAL *, REAL *, int *, REAL, REAL, REAL);
void ZBUNK(REAL, REAL, REAL, int, int, int, REAL *, REAL *, int *, REAL, REAL, REAL);
void ZBKNU(REAL, REAL, REAL, int, int, REAL *, REAL *, int *, REAL, REAL, REAL);
void ZBUNI(REAL ZR, REAL ZI, REAL FNU, int KODE, int N, REAL *YR, REAL *YI, int *NZ, int NUI, int *NLAST, REAL FNUL, REAL TOL, REAL ELIM, REAL ALIM);
void ZBINU(REAL ZR, REAL ZI, REAL FNU, int KODE, int N, REAL *CYR, REAL *CYI, int *NZ, REAL RL, REAL FNUL, REAL TOL, REAL ELIM, REAL ALIM);
void ZACAI(REAL, REAL, REAL, int, int, int, REAL *, REAL *, int *, REAL, REAL, REAL, REAL);
void ZACON(REAL, REAL, REAL, int, int, int, REAL *, REAL *, int *, REAL, REAL, REAL, REAL, REAL);
void ZKSCL(REAL, REAL, REAL, int, REAL *, REAL *, int *, REAL *, REAL *, REAL, REAL, REAL);
void ZRATI(REAL, REAL, REAL, int, REAL *, REAL *, REAL);
void ZWRSK(REAL ZRR, REAL ZRI, REAL FNU, int KODE, int N, REAL *YR, REAL *YI, int *NZ, REAL *CWR, REAL *CWI, REAL TOL, REAL ELIM, REAL ALIM);
void ZUNI1(REAL ZR, REAL ZI, REAL FNU, int KODE, int N, REAL *YR, REAL *YI, int *NZ, int *NLAST, REAL FNUL, REAL TOL, REAL ELIM, REAL ALIM);
void ZUNI2(REAL ZR, REAL ZI, REAL FNU, int KODE, int N, REAL *YR, REAL *YI, int *NZ, int *NLAST, REAL FNUL, REAL TOL, REAL ELIM, REAL ALIM);
void ZUNHJ(REAL, REAL, REAL, int, REAL, REAL *, REAL *, REAL *, REAL *, REAL *, REAL *, REAL *, REAL *, REAL *, REAL *, REAL *, REAL *);
void ZUNIK(REAL, REAL, REAL, int, int, REAL, int, REAL *, REAL *, REAL *, REAL *, REAL *, REAL *, REAL *, REAL *, REAL *, REAL *);
void ZUNK1(REAL ZR, REAL ZI, REAL FNU, int KODE, int MR, int N, REAL *YR, REAL *YI, int *NZ, REAL TOL, REAL ELIM, REAL ALIM);
void ZUNK2(REAL ZR, REAL ZI, REAL FNU, int KODE, int MR, int N, REAL *YR, REAL *YI, int *NZ, REAL TOL, REAL ELIM, REAL ALIM);

//Functions defined in CBess0.cpp.
void ZSERI(REAL, REAL, REAL, int, int, REAL *, REAL *, int *, REAL, REAL, REAL);
void ZASYI(REAL, REAL, REAL, int, int, REAL *, REAL *, int *, REAL, REAL, REAL, REAL);
void ZAIRY(REAL ZR, REAL ZI, int ID, int KODE, REAL *AIR, REAL *AII, int *NZ, int *IERR);
void ZMLRI(REAL, REAL, REAL, int, int, REAL *, REAL *, int *, REAL);
void ZS1S2(REAL *, REAL *, REAL *, REAL *, REAL *, REAL *, int *, REAL, REAL, int *);
void ZSHCH(REAL, REAL, REAL *, REAL *, REAL *, REAL *);
void ZUCHK(REAL, REAL, int *, REAL, REAL);
REAL DGAMLN(REAL, int *);









