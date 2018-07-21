/*---------------------------------------------------------------------
!      Utility subroutines used by any program from Numath library
!      with (not intrinsic) complex numbers z = (zr,zi).
! ---------------------------------------------------------------------
! Reference: From Numath Library By Tuan Dang Trong in Fortran 77
! [BIBLI 18].
!
!                               C++ Release 1.0 By J-P Moreau, Paris
!                                       (www.jpmoreau.fr)
!--------------------------------------------------------------------*/
//Module COMPLEX
#include "..\definitions.h"

const int NMAX = 10;

typedef  REAL VEC[NMAX+1];
typedef  REAL VEC16[17];


REAL ZABS(REAL ZR, REAL ZI)
{
	/***BEGIN PROLOGUE  ZABS
	!***REFER TO  ZBESH,ZBESI,ZBESJ,ZBESK,ZBESY,ZAIRY,ZBIRY

	!     ZABS COMPUTES THE ABSOLUTE VALUE OR MAGNITUDE OF A REAL
	!     PRECISION COMPLEX VARIABLE CMPLX(ZR,ZI)

	!***ROUTINES CALLED  (NONE)
	!***END PROLOGUE  ZABS */
	//Labels: e10, e20
	REAL U, V, Q, S;

	U = fabs(ZR);
	V = fabs(ZI);
	S = U + V;
	/*--------------------------------------------------------------------
	!     S*1.0 MAKES AN UNNORMALIZED UNDERFLOW ON CDC MACHINES INTO A
	!     TRUE FLOATING ZERO
	!-------------------------------------------------------------------*/
	S = S * 1.0;
	if (S == 0.0) goto e20;
	if (U > V)  goto e10;
	Q = U / V;
	return (V*sqrt(1.0 + Q * Q));
e10:  Q = V / U;
	return (U*sqrt(1.0 + Q * Q));
e20:  return 0.0;
} //ZABS()


void ZSQRT(REAL AR, REAL AI, REAL *BR, REAL *BI)
{
	/**BEGIN PROLOGUE  ZSQRT
	!***REFER TO  ZBESH,ZBESI,ZBESJ,ZBESK,ZBESY,ZAIRY,ZBIRY
	!
	!     REAL PRECISION COMPLEX SQUARE ROOT, B=CSQRT(A)
	!
	!***ROUTINES CALLED  ZABS
	!***END PROLOGUE  ZSQRT */
	//Labels: e10,e20,e30,e40,e50,e60,e70
	REAL ZM, DTHETA, DPI, DRT;
	DRT = 7.071067811865475244008443621e-01;
	DPI = 3.141592653589793238462643383;
	ZM = ZABS(AR, AI);
	ZM = sqrt(ZM);
	if (AR == 0.0) goto e10;
	if (AI == 0.0) goto e20;
	DTHETA = atan(AI / AR);
	if (DTHETA <= 0.0) goto e40;
	if (AR < 0.0) DTHETA = DTHETA - DPI;
	goto e50;
e10:  if (AI > 0.0) goto e60;
	if (AI < 0.0) goto e70;
	*BR = 0.0;
	*BI = 0.0;
	return;
e20:  if (AR > 0.0) goto e30;
	*BR = 0.0;
	*BI = sqrt(fabs(AR));
	return;
e30:  *BR = sqrt(AR);
	*BI = 0.0;
	return;
e40:  if (AR < 0.0)  DTHETA = DTHETA + DPI;
e50:  DTHETA = DTHETA * 0.5;
	*BR = ZM * cos(DTHETA);
	*BI = ZM * sin(DTHETA);
	return;
e60:  *BR = ZM * DRT;
	*BI = ZM * DRT;
	return;
e70:  *BR = ZM * DRT;
	*BI = -ZM * DRT;
} //ZSQRT()


void ZEXP(REAL AR, REAL AI, REAL *BR, REAL *BI)
{
	/***BEGIN PROLOGUE  ZEXP
	!***REFER TO  ZBESH,ZBESI,ZBESJ,ZBESK,ZBESY,ZAIRY,ZBIRY
	!
	!     REAL PRECISION COMPLEX EXPONENTIAL FUNCTION B=EXP(A)
	!
	!***ROUTINES CALLED  (NONE)
	!***END PROLOGUE  ZEXP */
	REAL ZM, CA, CB;
	ZM = exp(AR);
	CA = ZM * cos(AI);
	CB = ZM * sin(AI);
	*BR = CA;
	*BI = CB;
} //ZEXP()


void ZMLT(REAL AR, REAL AI, REAL BR, REAL BI, REAL *CR, REAL *CI)
{
	/***BEGIN PROLOGUE  ZMLT
	!***REFER TO  ZBESH,ZBESI,ZBESJ,ZBESK,ZBESY,ZAIRY,ZBIRY
	!
	!     REAL PRECISION COMPLEX MULTIPLY, C=A*B.
	!
	!***ROUTINES CALLED  (NONE)
	!***END PROLOGUE  ZMLT */
	REAL CA, CB;
	CA = AR * BR - AI * BI;
	CB = AR * BI + AI * BR;
	*CR = CA;
	*CI = CB;
} //ZMLT()

void ZDIV(REAL AR, REAL AI, REAL BR, REAL BI, REAL *CR, REAL *CI)
{
	/***BEGIN PROLOGUE  ZDIV
	!***REFER TO  ZBESH,ZBESI,ZBESJ,ZBESK,ZBESY,ZAIRY,ZBIRY
	!
	!     REAL PRECISION COMPLEX DIVIDE C=A/B.

	!***ROUTINES CALLED  ZABS
	!***END PROLOGUE  ZDIV */
	REAL BM, CA, CB, CC, CD;
	BM = 1.0 / ZABS(BR, BI);
	CC = BR * BM;
	CD = BI * BM;
	CA = (AR*CC + AI * CD)*BM;
	CB = (AI*CC - AR * CD)*BM;
	*CR = CA;
	*CI = CB;
} //ZDIV()

void ZLOG(REAL AR, REAL AI, REAL *BR, REAL *BI, int *IERR)
{
	/***BEGIN PROLOGUE  ZLOG
	!***REFER TO  ZBESH,ZBESI,ZBESJ,ZBESK,ZBESY,ZAIRY,ZBIRY

	!     REAL PRECISION COMPLEX LOGARITHM B=CLOG(A)
	!     IERR=0,NORMAL RETURN      IERR=1, Z=CMPLX(0.0,0.0)
	!***ROUTINES CALLED  ZABS
	!***END PROLOGUE  ZLOG */
	//Labels e10,e20,e30,e40,e50,e60
	REAL ZM, DTHETA, DPI, DHPI;
	DPI = 3.141592653589793238462643383;
	DHPI = 1.570796326794896619231321696;

	*IERR = 0;
	if (AR == 0.0) goto e10;
	if (AI == 0.0) goto e20;
	DTHETA = atan(AI / AR);
	if (DTHETA <= 0.0) goto e40;
	if (AR < 0.0)  DTHETA -= DPI;
	goto e50;
e10:  if (AI == 0.0) goto e60;
	*BI = DHPI;
	*BR = log(fabs(AI));
	if (AI < 0.0) *BI = -(*BI);
	return;
e20:  if (AR > 0.0) goto e30;
	*BR = log(fabs(AR));
	*BI = DPI;
	return;
e30:  *BR = log(AR);
	*BI = 0.0;
	return;
e40:  if (AR < 0.0)  DTHETA += DPI;
e50:  ZM = ZABS(AR, AI);
	*BR = log(ZM);
	*BI = DTHETA;
	return;
e60:  *IERR = 1;
} //ZLOG()

REAL D1MACH(int I)
{
	/***BEGIN PROLOGUE  D1MACH
	!***DATE WRITTEN   750101   (YYMMDD)
	!***REVISION DATE  860501   (YYMMDD)
	!***CATEGORY NO.  R1
	!***KEYWORDS  MACHINE CONSTANTS
	!***AUTHOR  FOX, P. A., (BELL LABS)
	!           HALL, A. D., (BELL LABS)
	!           SCHRYER, N. L., (BELL LABS)
	!***PURPOSE  RETURN REAL PRECISION MACHINE DEPENDENT CONSTANTS.
	!***DESCRIPTION

	!     D1MACH CAN BE USED TO OBTAIN MACHINE-DEPENDENT PARAMETERS
	!     FOR THE LOCAL MACHINE ENVIRONMENT.  IT IS A FUNCTION
	!     SUBPROGRAM WITH ONE (INPUT) ARGUMENT, AND CAN BE CALLED
	!     AS FOLLOWS, FOR EXAMPLE

	!          D = D1MACH(I)

	!     WHERE I=1,...,5.  THE (OUTPUT) VALUE OF D ABOVE IS
	!     DETERMINED BY THE (INPUT) VALUE OF I.  THE RESULTS FOR
	!     VARIOUS VALUES OF I ARE DISCUSSED BELOW.

	!  REAL-PRECISION MACHINE CONSTANTS
	!  D1MACH( 1) = B**(EMIN-1), THE SMALLEST POSITIVE MAGNITUDE.
	!  D1MACH( 2) = B**EMAX*(1 - B**(-T)), THE LARGEST MAGNITUDE.
	!  D1MACH( 3) = B**(-T), THE SMALLEST RELATIVE SPACING.
	!  D1MACH( 4) = B**(1-T), THE LARGEST RELATIVE SPACING.
	!  D1MACH( 5) = LOG10(B)
	!***REFERENCES  FOX P.A., HALL A.D., SCHRYER N.L.,*FRAMEWORK FOR A
	!                 PORTABLE LIBRARY*, ACM TRANSACTIONS ON MATHEMATICAL
	!                 SOFTWARE, VOL. 4, NO. 2, JUNE 1978, PP. 177-188.
	!***ROUTINES CALLED  XERROR
	!***END PROLOGUE  D1MACH */

	REAL DMACH[6];

	//***FIRST EXECUTABLE STATEMENT  D1MACH
	if (I < 1 || I > 5)
		printf(" D1MACH -- I OUT OF BOUNDS\n");

	//For IBM PC or APOLLO
	DMACH[1] = 2.22559e-308;
	DMACH[2] = 1.79728e308;
	DMACH[3] = 1.11048e-16;
	DMACH[4] = 2.22096e-16;
	DMACH[5] = 0.301029995663981198;

	return DMACH[I];

} //D1MACH()

long I1MACH(int I)
{
	/***BEGIN PROLOGUE  I1MACH
	!***DATE WRITTEN   750101   (YYMMDD)
	!***REVISION DATE  890313   (YYMMDD)
	!***CATEGORY NO.  R1
	!***KEYWORDS  LIBRARY=SLATEC,TYPE=INTEGER(I1MACH-I),MACHINE CONSTANTS
	!***AUTHOR  FOX, P. A., (BELL LABS)
	!           HALL, A. D., (BELL LABS)
	!           SCHRYER, N. L., (BELL LABS)
	!***PURPOSE  Return integer machine dependent constants.
	!***DESCRIPTION

	!     I1MACH can be used to obtain machine-dependent parameters
	!     for the local machine environment.  It is a function
	!     subroutine with one (input) argument, and can be called
	!     as follows, for example

	!          K = I1MACH(I)

	!     where I=1,...,16.  The (output) value of K above is
	!     determined by the (input) value of I.  The results for
	!     various values of I are discussed below.

	!  I/O unit numbers.
	!    I1MACH( 1) = the standard input unit.
	!    I1MACH( 2) = the standard output unit.
	!    I1MACH( 3) = the standard punch unit.
	!    I1MACH( 4) = the standard error message unit.

	!  Words.
	!    I1MACH( 5) = the number of bits per integer storage unit.
	!    I1MACH( 6) = the number of characters per integer storage unit.

	!  Integers.
	!    assume integers are represented in the S-digit, base-A form

	!               sign ( X(S-1)*A**(S-1) + ... + X(1)*A + X(0) )

	!               where 0 .LE. X(I) .LT. A for I=0,...,S-1.
	!    I1MACH( 7) = A, the base.
	!    I1MACH( 8) = S, the number of base-A digits.
	!    I1MACH( 9) = A**S - 1, the largest magnitude.

	!  Floating-Point Numbers.
	!    Assume floating-point numbers are represented in the T-digit,
	!    base-B form
	!               sign (B**E)*( (X(1)/B) + ... + (X(T)/B**T) )

	!               where 0 .LE. X(I) .LT. B for I=1,...,T,
	!               0 .LT. X(1), and EMIN .LE. E .LE. EMAX.
	!    I1MACH(10) = B, the base.

	!  Single-Precision
	!    I1MACH(11) = T, the number of base-B digits.
	!    I1MACH(12) = EMIN, the smallest exponent E.
	!    I1MACH(13) = EMAX, the largest exponent E.

	!  REAL-Precision
	!    I1MACH(14) = T, the number of base-B digits.
	!    I1MACH(15) = EMIN, the smallest exponent E.
	!    I1MACH(16) = EMAX, the largest exponent E.

	!  To alter this function for a particular environment,
	!  the desired set of DATA statements should be activated by
	!  removing the ! from column 1.  Also, the values of
	!  I1MACH(1) - I1MACH(4) should be checked for consistency
	!  with the local operating system.

	!***REFERENCES  FOX P.A., HALL A.D., SCHRYER N.L.,*FRAMEWORK FOR A
	!                 PORTABLE LIBRARY*, ACM TRANSACTIONS ON MATHEMATICAL
	!                 SOFTWARE, VOL. 4, NO. 2, JUNE 1978, PP. 177-188.
	!***ROUTINES CALLED  (NONE)
	!***END PROLOGUE  I1MACH */
	long IMACH[17];

	//***FIRST EXECUTABLE STATEMENT  D1MACH
	if (I < 1 || I > 16)
		printf(" I1MACH -- I OUT OF BOUNDS\n");

	// For IBM PC or APOLLO
	IMACH[1] = 5;
	IMACH[2] = 6;
	IMACH[3] = 0;
	IMACH[4] = 0;
	IMACH[5] = 32;
	IMACH[6] = 4;
	IMACH[7] = 2;
	IMACH[8] = 31;
	IMACH[9] = 2147483647;
	IMACH[10] = 2;
	IMACH[11] = 24;
	IMACH[12] = -125;
	IMACH[13] = 127;
	IMACH[14] = 53;
	IMACH[15] = -1021;
	IMACH[16] = 1023;

	return IMACH[I];

} //I1MACH()

REAL DMAX(REAL a, REAL b)
{
	if (a >= b) return a;
	else return b;
}

REAL DMIN(REAL a, REAL b)
{
	if (a <= b) return a;
	else return b;
}

int IMAX(int a, int b)
{
	if (a >= b) return a;
	else return b;
}

int IMIN(int a, int b)
{
	if (a <= b) return a;
	else return b;
}

// end of file Complex.cpp
