/****************************************************************
* EVALUATE A Y-BESSEL FUNCTION OF COMPLEX ARGUMENT (SECOND KIND)*
* ------------------------------------------------------------- *
* SAMPLE RUN:                                                   *
* (Evaluate Y0 to Y4 for argument Z=(1.0,2.0) ).                *
*                                                               *
* zr(0) =   1.367419                                            *
* zi(0) =   1.521507                                            *
* zr(1) =  -1.089470                                            *
* zi(1) =   1.314951                                            *
* zr(2) =  -0.751245                                            *
* zi(2) =  -0.123950                                            *
* zr(3) =   0.290153                                            *
* zi(3) =  -0.212119                                            *
* zr(4) =   0.590344                                            *
* zi(4) =  -0.826960                                            *
* NZ = 0                                                        *
* Error code: 0                                                 *
*                                                               *
* ------------------------------------------------------------- *
* Ref.: From Numath Library By Tuan Dang Trong in Fortran 77    *
*       [BIBLI 18].                                             *
*                                                               *
*                        C++ Release 1.0 By J-P Moreau, Paris   *
*                                  (www.jpmoreau.fr)            *     
*****************************************************************
Note: to link with CBess0,CBess00,CBess1,CBess2,CBess3,Complex.
-------------------------------------------------------------- */
#include "../definitions.h"

#include "complex.h"
//Uses Complex, CBess0, CBess00, CBess1, CBess2, CBess3,
//     Basis_r, Vmblock.

    //Headers of functions used below
    void ZBESH(REAL ZR, REAL ZI, REAL FNU, int KODE, int M, int N,
               REAL *CYR, REAL *CYI, int *NZ, int *IERR);

	void ZUOIK(REAL, REAL, REAL, int, int, int, REAL *, REAL *, 
		       int *, REAL, REAL, REAL);

	void ZBKNU(REAL, REAL, REAL, int, int, REAL *, REAL *, int *, REAL, 
	           REAL, REAL);

	void ZACON(REAL, REAL, REAL, int, int, int, REAL *, REAL *, int *, 
		       REAL, REAL, REAL, REAL, REAL);

	void ZBUNK(REAL, REAL, REAL, int, int, int, REAL *, REAL *, int *, 
		       REAL, REAL, REAL);


void ZBESY(REAL ZR, REAL ZI, REAL FNU, int KODE, int N, REAL *CYR, REAL *CYI,
           int *NZ, REAL *CWRKR, REAL *CWRKI, int *IERR)  {

/***BEGIN PROLOGUE  ZBESY
!***DATE WRITTEN   830501   (YYMMDD)    (Original Fortran Version).
!***REVISION DATE  830501   (YYMMDD)
!***CATEGORY NO.  B5K
!***KEYWORDS  Y-BESSEL FUNCTION,BESSEL FUNCTION OF COMPLEX ARGUMENT,
!             BESSEL FUNCTION OF SECOND KIND
!***AUTHOR  AMOS, DONALD E., SANDIA NATIONAL LABORATORIES
!***PURPOSE  TO COMPUTE THE Y-BESSEL FUNCTION OF A COMPLEX ARGUMENT
!***DESCRIPTION
!
!                      ***A DOUBLE PRECISION ROUTINE***
!
!         ON KODE=1, CBESY COMPUTES AN N MEMBER SEQUENCE OF COMPLEX
!         BESSEL FUNCTIONS CY(I)=Y(FNU+I-1,Z) FOR REAL, NONNEGATIVE
!         ORDERS FNU+I-1, I=1,...,N AND COMPLEX Z IN THE CUT PLANE
!         -PI.LT.ARG(Z).LE.PI. ON KODE=2, CBESY RETURNS THE SCALED
!         FUNCTIONS
!
!         CY(I)=EXP(-ABS(Y))*Y(FNU+I-1,Z)   I = 1,...,N , Y=AIMAG(Z)
!
!         WHICH REMOVE THE EXPONENTIAL GROWTH IN BOTH THE UPPER AND
!         LOWER HALF PLANES FOR Z TO INFINITY. DEFINITIONS AND NOTATION
!         ARE FOUND IN THE NBS HANDBOOK OF MATHEMATICAL FUNCTIONS
!         (REF. 1).
!
!         INPUT      ZR,ZI,FNU ARE DOUBLE PRECISION
!           ZR,ZI  - Z=CMPLX(ZR,ZI), Z.NE.CMPLX(0.0D0,0.0D0),
!                    -PI.LT.ARG(Z).LE.PI
!           FNU    - ORDER OF INITIAL Y FUNCTION, FNU.GE.0.0D0
!           KODE   - A PARAMETER TO INDICATE THE SCALING OPTION
!                    KODE= 1  RETURNS
!                             CY(I)=Y(FNU+I-1,Z), I=1,...,N
!                        = 2  RETURNS
!                             CY(I)=Y(FNU+I-1,Z)*EXP(-ABS(Y)), I=1,...,N
!                             WHERE Y=AIMAG(Z)
!           N      - NUMBER OF MEMBERS OF THE SEQUENCE, N.GE.1
!           CWRKR, - DOUBLE PRECISION WORK VECTORS OF DIMENSION AT
!           CWRKI    AT LEAST N
!
!         OUTPUT     CYR,CYI ARE DOUBLE PRECISION
!           CYR,CYI- DOUBLE PRECISION VECTORS WHOSE FIRST N COMPONENTS
!                    CONTAIN REAL AND IMAGINARY PARTS FOR THE SEQUENCE
!                    CY(I)=Y(FNU+I-1,Z)  OR
!                    CY(I)=Y(FNU+I-1,Z)*EXP(-ABS(Y))  I=1,...,N
!                    DEPENDING ON KODE.
!           NZ     - NZ=0 , A NORMAL RETURN
!                    NZ.GT.0 , NZ COMPONENTS OF CY SET TO ZERO DUE TO
!                    UNDERFLOW (GENERALLY ON KODE=2)
!           IERR   - ERROR FLAG
!                    IERR=0, NORMAL RETURN - COMPUTATION COMPLETED
!                    IERR=1, INPUT ERROR   - NO COMPUTATION
!                    IERR=2, OVERFLOW      - NO COMPUTATION, FNU IS
!                            TOO LARGE OR CABS(Z) IS TOO SMALL OR BOTH
!                    IERR=3, CABS(Z) OR FNU+N-1 LARGE - COMPUTATION DONE
!                            BUT LOSSES OF SIGNIFCANCE BY ARGUMENT
!                            REDUCTION PRODUCE LESS THAN HALF OF MACHINE
!                            ACCURACY
!                    IERR=4, CABS(Z) OR FNU+N-1 TOO LARGE - NO COMPUTA-
!                            TION BECAUSE OF COMPLETE LOSSES OF SIGNIFI-
!                            CANCE BY ARGUMENT REDUCTION
!                    IERR=5, ERROR              - NO COMPUTATION,
!                            ALGORITHM TERMINATION CONDITION NOT MET
!
!***LONG DESCRIPTION
!
!         THE COMPUTATION IS CARRIED OUT BY THE FORMULA
!
!             Y(FNU,Z)=0.5*(H(1,FNU,Z)-H(2,FNU,Z))/I
!
!         WHERE I^2 = -1 AND THE HANKEL BESSEL FUNCTIONS H(1,FNU,Z)
!         AND H(2,FNU,Z) ARE CALCULATED IN CBESH.
!
!         FOR NEGATIVE ORDERS,THE FORMULA
!
!             Y(-FNU,Z) = Y(FNU,Z)*COS(PI*FNU) + J(FNU,Z)*SIN(PI*FNU)
!
!         CAN BE USED. HOWEVER,FOR LARGE ORDERS CLOSE TO HALF ODD
!         INTEGERS THE FUNCTION CHANGES RADICALLY. WHEN FNU IS A LARGE
!         POSITIVE HALF ODD INTEGER,THE MAGNITUDE OF Y(-FNU,Z)=J(FNU,Z)*
!         SIN(PI*FNU) IS A LARGE NEGATIVE POWER OF TEN. BUT WHEN FNU IS
!         NOT A HALF ODD INTEGER, Y(FNU,Z) DOMINATES IN MAGNITUDE WITH A
!         LARGE POSITIVE POWER OF TEN AND THE MOST THAT THE SECOND TERM
!         CAN BE REDUCED IS BY UNIT ROUNDOFF FROM THE COEFFICIENT. THUS,
!         WIDE CHANGES CAN OCCUR WITHIN UNIT ROUNDOFF OF A LARGE HALF
!         ODD INTEGER. HERE, LARGE MEANS FNU.GT.CABS(Z).
!
!         IN MOST COMPLEX VARIABLE COMPUTATION, ONE MUST EVALUATE ELE-
!         MENTARY FUNCTIONS. WHEN THE MAGNITUDE OF Z OR FNU+N-1 IS
!         LARGE, LOSSES OF SIGNIFICANCE BY ARGUMENT REDUCTION OCCUR.
!         CONSEQUENTLY, IF EITHER ONE EXCEEDS U1=SQRT(0.5/UR), THEN
!         LOSSES EXCEEDING HALF PRECISION ARE LIKELY AND AN ERROR FLAG
!         IERR=3 IS TRIGGERED WHERE UR=DMAX1(D1MACH(4),1.0D-18) IS
!         DOUBLE PRECISION UNIT ROUNDOFF LIMITED TO 18 DIGITS PRECISION.
!         IF EITHER IS LARGER THAN U2=0.5/UR, THEN ALL SIGNIFICANCE IS
!         LOST AND IERR=4. IN ORDER TO USE THE INT FUNCTION, ARGUMENTS
!         MUST BE FURTHER RESTRICTED NOT TO EXCEED THE LARGEST MACHINE
!         INTEGER, U3=I1MACH(9). THUS, THE MAGNITUDE OF Z AND FNU+N-1 IS
!         RESTRICTED BY MIN(U2,U3). ON 32 BIT MACHINES, U1,U2, AND U3
!         ARE APPROXIMATELY 2.0E+3, 4.2E+6, 2.1E+9 IN SINGLE PRECISION
!         ARITHMETIC AND 1.3E+8, 1.8E+16, 2.1E+9 IN DOUBLE PRECISION
!         ARITHMETIC RESPECTIVELY. THIS MAKES U2 AND U3 LIMITING IN
!         THEIR RESPECTIVE ARITHMETICS. THIS MEANS THAT ONE CAN EXPECT
!         TO RETAIN, IN THE WORST CASES ON 32 BIT MACHINES, NO DIGITS
!         IN SINGLE AND ONLY 7 DIGITS IN DOUBLE PRECISION ARITHMETIC.
!         SIMILAR CONSIDERATIONS HOLD FOR OTHER MACHINES.
!
!         THE APPROXIMATE RELATIVE ERROR IN THE MAGNITUDE OF A COMPLEX
!         BESSEL FUNCTION CAN BE EXPRESSED BY P*10^S WHERE P=MAX(UNIT
!         ROUNDOFF,1.0E-18) IS THE NOMINAL PRECISION AND 10^S REPRE-
!         SENTS THE INCREASE IN ERROR DUE TO ARGUMENT REDUCTION IN THE
!         ELEMENTARY FUNCTIONS. HERE, S=MAX(1,ABS(LOG10(CABS(Z))),
!         ABS(LOG10(FNU))) APPROXIMATELY (I.E. S=MAX(1,ABS(EXPONENT OF
!         CABS(Z),ABS(EXPONENT OF FNU)) ). HOWEVER, THE PHASE ANGLE MAY
!         HAVE ONLY ABSOLUTE ACCURACY. THIS IS MOST LIKELY TO OCCUR WHEN
!         ONE COMPONENT (IN ABSOLUTE VALUE) IS LARGER THAN THE OTHER BY
!         SEVERAL ORDERS OF MAGNITUDE. IF ONE COMPONENT IS 10^K LARGER
!         THAN THE OTHER, THEN ONE CAN EXPECT ONLY MAX(ABS(LOG10(P))-K,
!         0) SIGNIFICANT DIGITS; OR, STATED ANOTHER WAY, WHEN K EXCEEDS
!         THE EXPONENT OF P, NO SIGNIFICANT DIGITS REMAIN IN THE SMALLER
!         COMPONENT. HOWEVER, THE PHASE ANGLE RETAINS ABSOLUTE ACCURACY
!         BECAUSE, IN COMPLEX ARITHMETIC WITH PRECISION P, THE SMALLER
!         COMPONENT WILL NOT (AS A RULE) DECREASE BELOW P TIMES THE
!         MAGNITUDE OF THE LARGER COMPONENT. IN THESE EXTREME CASES,
!         THE PRINCIPAL PHASE ANGLE IS ON THE ORDER OF +P, -P, PI/2-P,
!         OR -PI/2+P.
!
!***REFERENCES  HANDBOOK OF MATHEMATICAL FUNCTIONS BY M. ABRAMOWITZ
!                 AND I. A. STEGUN, NBS AMS SERIES 55, U.S. DEPT. OF
!                 COMMERCE, 1955.
!
!               COMPUTATION OF BESSEL FUNCTIONS OF COMPLEX ARGUMENT
!                 BY D. E. AMOS, SAND83-0083, MAY, 1983.
!
!               COMPUTATION OF BESSEL FUNCTIONS OF COMPLEX ARGUMENT
!                 AND LARGE ORDER BY D. E. AMOS, SAND83-0643, MAY, 1983
!
!               A SUBROUTINE PACKAGE FOR BESSEL FUNCTIONS OF A COMPLEX
!                 ARGUMENT AND NONNEGATIVE ORDER BY D. E. AMOS, SAND85-
!                 1018, MAY, 1985
!
!               A PORTABLE PACKAGE FOR BESSEL FUNCTIONS OF A COMPLEX
!                 ARGUMENT AND NONNEGATIVE ORDER BY D. E. AMOS, TRANS.
!                 MATH. SOFTWARE, 1986
!
!***ROUTINES CALLED  ZBESH,I1MACH,D1MACH
!***END PROLOGUE  ZBESY
!
!     COMPLEX CWRK,CY,C1,C2,EX,HCI,Z */
//Labels: e60, e70, e90, e170

      REAL C1I, C1R, C2I, C2R, ELIM, EXI, EXR, EY, HCII, STI, STR, TAY;
      int I, K, K1, K2, NZ1, NZ2;

//***FIRST EXECUTABLE STATEMENT ZBESY
      *IERR = 0;
      *NZ=0;
      if (ZR == 0.0 && ZI == 0.0) *IERR=1;
      if (FNU < 0.0) *IERR=1;
      if (KODE < 1 || KODE > 2) *IERR=1;
      if (N < 1) *IERR=1;
      if (*IERR != 0) return;   //bad parameter(s)
      HCII = 0.5;
      ZBESH(ZR, ZI, FNU, KODE, 1, N, CYR, CYI, &NZ1, IERR);
      if (*IERR != 0 && *IERR != 3) goto e170;
      ZBESH(ZR, ZI, FNU, KODE, 2, N, CWRKR, CWRKI, &NZ2, IERR);
      if (*IERR != 0 && *IERR != 3) goto e170;
      *NZ = IMIN(NZ1,NZ2);
      if (KODE == 2) goto e60;
      for (I=1; I<=N; I++) {
        STR = CWRKR[I] - CYR[I];
        STI = CWRKI[I] - CYI[I];
        CYR[I] = -STI*HCII;
        CYI[I] = STR*HCII;
      }
      return;
e60:  K1 = I1MACH(15);
      K2 = I1MACH(16);
      K = IMIN(ABS(K1),ABS(K2));
/*----------------------------------------------------------------------
!     ELIM IS THE APPROXIMATE EXPONENTIAL UNDER- AND OVERFLOW LIMIT
!---------------------------------------------------------------------*/
      ELIM = 2.303*(K*D1MACH(5)-3.0);
      EXR = COS(ZR);
      EXI = SIN(ZR);
      EY = 0.0;
      TAY = ABS(ZI+ZI);
      if (TAY < ELIM)  EY = EXP(-TAY);
      if (ZI < 0.0)  goto e90;
      C1R = EXR*EY;
      C1I = EXI*EY;
      C2R = EXR;
      C2I = -EXI;
e70:  *NZ = 0;
      for (I=1; I<=N; I++) {
        STR = C1R*CYR[I] - C1I*CYI[I];
        STI = C1R*CYI[I] + C1I*CYR[I];
        STR = -STR + C2R*CWRKR[I] - C2I*CWRKI[I];
        STI = -STI + C2R*CWRKI[I] + C2I*CWRKR[I];
        CYR[I] = -STI*HCII;
        CYI[I] = STR*HCII;
        if (STR == 0.0 && STI == 0.0 && EY == 0.0)  *NZ = *NZ + 1;
      }
      return;
e90:  C1R = EXR;
      C1I = EXI;
      C2R = EXR*EY;
      C2I = -EXI*EY;
      goto e70;
e170: *NZ = 0;
} //ZBESY()


void ZBESH(REAL ZR, REAL ZI, REAL FNU, int KODE, int M, int N,
	REAL *CYR, REAL *CYI, int *NZ, int *IERR) {
	/***BEGIN PROLOGUE  ZBESH
	!***DATE WRITTEN   830501   (YYMMDD)
	!***REVISION DATE  830501   (YYMMDD)
	!***CATEGORY NO.  B5K
	!***KEYWORDS  H-BESSEL FUNCTIONS,BESSEL FUNCTIONS OF COMPLEX ARGUMENT,
	!             BESSEL FUNCTIONS OF THIRD KIND,HANKEL FUNCTIONS
	!***AUTHOR  AMOS, DONALD E., SANDIA NATIONAL LABORATORIES
	!***PURPOSE  TO COMPUTE THE H-BESSEL FUNCTIONS OF A COMPLEX ARGUMENT
	!***DESCRIPTION
	!
	!                      ***A DOUBLE PRECISION ROUTINE***
	!         ON KODE=1, ZBESH COMPUTES AN N MEMBER SEQUENCE OF COMPLEX
	!         HANKEL (BESSEL) FUNCTIONS CY(J)=H(M,FNU+J-1,Z) FOR KINDS M=1
	!         OR 2, REAL, NONNEGATIVE ORDERS FNU+J-1, J=1,...,N, AND COMPLEX
	!         Z.NE.CMPLX(0.0,0.0) IN THE CUT PLANE -PI.LT.ARG(Z).LE.PI.
	!         ON KODE=2, ZBESH RETURNS THE SCALED HANKEL FUNCTIONS
	!
	!         CY(I)=EXP(-MM*Z*I)*H(M,FNU+J-1,Z)       MM=3-2*M,   I^2=-1.
	!
	!         WHICH REMOVES THE EXPONENTIAL BEHAVIOR IN BOTH THE UPPER AND
	!         LOWER HALF PLANES. DEFINITIONS AND NOTATION ARE FOUND IN THE
	!         NBS HANDBOOK OF MATHEMATICAL FUNCTIONS (REF. 1).
	!
	!         INPUT      ZR,ZI,FNU ARE DOUBLE PRECISION
	!           ZR,ZI  - Z=CMPLX(ZR,ZI), Z.NE.CMPLX(0.0D0,0.0D0),
	!                    -PT.LT.ARG(Z).LE.PI
	!           FNU    - ORDER OF INITIAL H FUNCTION, FNU.GE.0.0D0
	!           KODE   - A PARAMETER TO INDICATE THE SCALING OPTION
	!                    KODE= 1  RETURNS
	!                             CY(J)=H(M,FNU+J-1,Z),   J=1,...,N
	!                        = 2  RETURNS
	!                             CY(J)=H(M,FNU+J-1,Z)*EXP(-I*Z*(3-2M))
	!                                  J=1,...,N  ,  I^2=-1
	!           M      - KIND OF HANKEL FUNCTION, M=1 OR 2
	!           N      - NUMBER OF MEMBERS IN THE SEQUENCE, N.GE.1
	!
	!         OUTPUT     CYR,CYI ARE DOUBLE PRECISION
	!           CYR,CYI- DOUBLE PRECISION VECTORS WHOSE FIRST N COMPONENTS
	!                    CONTAIN REAL AND IMAGINARY PARTS FOR THE SEQUENCE
	!                    CY(J)=H(M,FNU+J-1,Z)  OR
	!                    CY(J)=H(M,FNU+J-1,Z)*EXP(-I*Z*(3-2M))  J=1,...,N
	!                    DEPENDING ON KODE, I^2=-1.
	!           NZ     - NUMBER OF COMPONENTS SET TO ZERO DUE TO UNDERFLOW,
	!                    NZ= 0   , NORMAL RETURN
	!                    NZ.GT.0 , FIRST NZ COMPONENTS OF CY SET TO ZERO DUE
	!                              TO UNDERFLOW, CY(J)=CMPLX(0.0D0,0.0D0)
	!                              J=1,...,NZ WHEN Y.GT.0.0 AND M=1 OR
	!                              Y.LT.0.0 AND M=2. FOR THE COMPLMENTARY
	!                              HALF PLANES, NZ STATES ONLY THE NUMBER
	!                              OF UNDERFLOWS.
	!           IERR   - ERROR FLAG
	!                    IERR=0, NORMAL RETURN - COMPUTATION COMPLETED
	!                    IERR=1, INPUT ERROR   - NO COMPUTATION
	!                    IERR=2, OVERFLOW      - NO COMPUTATION, FNU TOO
	!                            LARGE OR CABS(Z) TOO SMALL OR BOTH
	!                    IERR=3, CABS(Z) OR FNU+N-1 LARGE - COMPUTATION DONE
	!                            BUT LOSSES OF SIGNIFCANCE BY ARGUMENT
	!                            REDUCTION PRODUCE LESS THAN HALF OF MACHINE
	!                            ACCURACY
	!                    IERR=4, CABS(Z) OR FNU+N-1 TOO LARGE - NO COMPUTA-
	!                            TION BECAUSE OF COMPLETE LOSSES OF SIGNIFI-
	!                            CANCE BY ARGUMENT REDUCTION
	!                    IERR=5, ERROR              - NO COMPUTATION,
	!                            ALGORITHM TERMINATION CONDITION NOT MET
	!
	!***LONG DESCRIPTION
	!
	!         THE COMPUTATION IS CARRIED OUT BY THE RELATION
	!
	!         H(M,FNU,Z)=(1/MP)*EXP(-MP*FNU)*K(FNU,Z*EXP(-MP))
	!             MP=MM*HPI*I,  MM=3-2*M,  HPI=PI/2,  I^2=-1
	!
	!         FOR M=1 OR 2 WHERE THE K BESSEL FUNCTION IS COMPUTED FOR THE
	!         RIGHT HALF PLANE RE(Z).GE.0.0. THE K FUNCTION IS CONTINUED
	!         TO THE LEFT HALF PLANE BY THE RELATION
	!
	!         K(FNU,Z*EXP(MP)) = EXP(-MP*FNU)*K(FNU,Z)-MP*I(FNU,Z)
	!         MP=MR*PI*I, MR=+1 OR -1, RE(Z).GT.0, I^2=-1
	!
	!         WHERE I(FNU,Z) IS THE I BESSEL FUNCTION.
	!
	!         EXPONENTIAL DECAY OF H(M,FNU,Z) OCCURS IN THE UPPER HALF Z
	!         PLANE FOR M=1 AND THE LOWER HALF Z PLANE FOR M=2.  EXPONENTIAL
	!         GROWTH OCCURS IN THE COMPLEMENTARY HALF PLANES.  SCALING
	!         BY EXP(-MM*Z*I) REMOVES THE EXPONENTIAL BEHAVIOR IN THE
	!         WHOLE Z PLANE FOR Z TO INFINITY.
	!
	!         FOR NEGATIVE ORDERS,THE FORMULAE
	!
	!               H(1,-FNU,Z) = H(1,FNU,Z)*CEXP( PI*FNU*I)
	!               H(2,-FNU,Z) = H(2,FNU,Z)*CEXP(-PI*FNU*I)
	!                         I^2=-1
	!
	!         CAN BE USED.
	!
	!         IN MOST COMPLEX VARIABLE COMPUTATION, ONE MUST EVALUATE ELE-
	!         MENTARY FUNCTIONS. WHEN THE MAGNITUDE OF Z OR FNU+N-1 IS
	!         LARGE, LOSSES OF SIGNIFICANCE BY ARGUMENT REDUCTION OCCUR.
	!         CONSEQUENTLY, IF EITHER ONE EXCEEDS U1=SQRT(0.5/UR), THEN
	!         LOSSES EXCEEDING HALF PRECISION ARE LIKELY AND AN ERROR FLAG
	!         IERR=3 IS TRIGGERED WHERE UR=DMAX1(D1MACH(4),1.0D-18) IS
	!         DOUBLE PRECISION UNIT ROUNDOFF LIMITED TO 18 DIGITS PRECISION.
	!         IF EITHER IS LARGER THAN U2=0.5/UR, THEN ALL SIGNIFICANCE IS
	!         LOST AND IERR=4. IN ORDER TO USE THE INT FUNCTION, ARGUMENTS
	!         MUST BE FURTHER RESTRICTED NOT TO EXCEED THE LARGEST MACHINE
	!         INTEGER, U3=I1MACH(9). THUS, THE MAGNITUDE OF Z AND FNU+N-1 IS
	!         RESTRICTED BY MIN(U2,U3). ON 32 BIT MACHINES, U1,U2, AND U3
	!         ARE APPROXIMATELY 2.0E+3, 4.2E+6, 2.1E+9 IN SINGLE PRECISION
	!         ARITHMETIC AND 1.3E+8, 1.8E+16, 2.1E+9 IN DOUBLE PRECISION
	!         ARITHMETIC RESPECTIVELY. THIS MAKES U2 AND U3 LIMITING IN
	!         THEIR RESPECTIVE ARITHMETICS. THIS MEANS THAT ONE CAN EXPECT
	!         TO RETAIN, IN THE WORST CASES ON 32 BIT MACHINES, NO DIGITS
	!         IN SINGLE AND ONLY 7 DIGITS IN DOUBLE PRECISION ARITHMETIC.
	!         SIMILAR CONSIDERATIONS HOLD FOR OTHER MACHINES.
	!
	!         THE APPROXIMATE RELATIVE ERROR IN THE MAGNITUDE OF A COMPLEX
	!         BESSEL FUNCTION CAN BE EXPRESSED BY P*10^S WHERE P=MAX(UNIT
	!         ROUNDOFF,1.0D-18) IS THE NOMINAL PRECISION AND 10^S REPRE-
	!         SENTS THE INCREASE IN ERROR DUE TO ARGUMENT REDUCTION IN THE
	!         ELEMENTARY FUNCTIONS. HERE, S=MAX(1,ABS(LOG10(CABS(Z))),
	!         ABS(LOG10(FNU))) APPROXIMATELY (I.E. S=MAX(1,ABS(EXPONENT OF
	!         CABS(Z),ABS(EXPONENT OF FNU)) ). HOWEVER, THE PHASE ANGLE MAY
	!         HAVE ONLY ABSOLUTE ACCURACY. THIS IS MOST LIKELY TO OCCUR WHEN
	!         ONE COMPONENT (IN ABSOLUTE VALUE) IS LARGER THAN THE OTHER BY
	!         SEVERAL ORDERS OF MAGNITUDE. IF ONE COMPONENT IS 10^K LARGER
	!         THAN THE OTHER, THEN ONE CAN EXPECT ONLY MAX(ABS(LOG10(P))-K,
	!         0) SIGNIFICANT DIGITS; OR, STATED ANOTHER WAY, WHEN K EXCEEDS
	!         THE EXPONENT OF P, NO SIGNIFICANT DIGITS REMAIN IN THE SMALLER
	!         COMPONENT. HOWEVER, THE PHASE ANGLE RETAINS ABSOLUTE ACCURACY
	!         BECAUSE, IN COMPLEX ARITHMETI! WITH PRECISION P, THE SMALLER
	!         COMPONENT WILL NOT (AS A RULE) DECREASE BELOW P TIMES THE
	!         MAGNITUDE OF THE LARGER COMPONENT. IN THESE EXTREME CASES,
	!         THE PRINCIPAL PHASE ANGLE IS ON THE ORDER OF +P, -P, PI/2-P,
	!         OR -PI/2+P.
	!
	!***REFERENCES  HANDBOOK OF MATHEMATICAL FUNCTIONS BY M. ABRAMOWITZ
	!                 AND I. A. STEGUN, NBS AMS SERIES 55, U.S. DEPT. OF
	!                 COMMERCE, 1955.
	!
	!               COMPUTATION OF BESSEL FUNCTIONS OF COMPLEX ARGUMENT
	!                 BY D. E. AMOS, SAND83-0083, MAY, 1983.
	!
	!               COMPUTATION OF BESSEL FUNCTIONS OF COMPLEX ARGUMENT
	!                 AND LARGE ORDER BY D. E. AMOS, SAND83-0643, MAY, 1983
	!
	!               A SUBROUTINE PACKAGE FOR BESSEL FUNCTIONS OF A COMPLEX
	!                 ARGUMENT AND NONNEGATIVE ORDER BY D. E. AMOS, SAND85-
	!                 1018, MAY, 1985
	!
	!               A PORTABLE PACKAGE FOR BESSEL FUNCTIONS OF A COMPLEX
	!                 ARGUMENT AND NONNEGATIVE ORDER BY D. E. AMOS, TRANS.
	!                 MATH. SOFTWARE, 1986
	!
	!***ROUTINES CALLED  ZACON,ZBKNU,ZBUNK,ZUOIK,ZABS,I1MACH,D1MACH
	!***END PROLOGUE  ZBESH
	!
	!     COMPLEX CY,Z,ZN,ZT */
	//Labels: e60,e70,e80,e90,e100,e110,e120,e140,e230,e240,e260

	REAL AA, ALIM, ALN, ARG, AZ, DIG, ELIM, FMM, FN, FNUL, HPI,
		RHPI, RL, R1M5, SGN, STR, TOL, UFL, ZNI, ZNR, ZTI, BB;
	int I, INU, INUH, IR, K, K1, K2, MM, MR, NN, NUF, NW;

	//***FIRST EXECUTABLE STATEMENT ZBESH
	HPI = 1.57079632679489662;
	*IERR = 0;
	*NZ = 0;
	if (ZR == 0.0 && ZI == 0.0) *IERR = 1;
	if (FNU < 0.0) *IERR = 1;
	if (M < 1 || M > 2) *IERR = 1;
	if (KODE < 1 || KODE > 2) *IERR = 1;
	if (N < 1) *IERR = 1;
	if (*IERR != 0) return;  //bad parameter(s)
	NN = N;
	/*----------------------------------------------------------------------
	!     SET PARAMETERS RELATED TO MACHINE CONSTANTS.
	!     TOL IS THE APPROXIMATE UNIT ROUNDOFF LIMITED TO 1E-18.
	!     ELIM IS THE APPROXIMATE EXPONENTIAL OVER- AND UNDERFLOW LIMIT.
	!     EXP(-ELIM).LT.EXP(-ALIM)=EXP(-ELIM)/TOL    AND
	!     EXP(ELIM).GT.EXP(ALIM)=EXP(ELIM)*TOL       ARE INTERVALS NEAR
	!     UNDERFLOW AND OVERFLOW LIMITS WHERE SCALED ARITHMETIC IS DONE.
	!     RL IS THE LOWER BOUNDARY OF THE ASYMPTOTIC EXPANSION FOR LARGE Z.
	!     DIG = NUMBER OF BASE 10 DIGITS IN TOL = 10^(-DIG).
	!     FNUL IS THE LOWER BOUNDARY OF THE ASYMPTOTIC SERIES FOR LARGE FNU
	!---------------------------------------------------------------------*/
	TOL = DMAX(D1MACH(4), 1e-18);
	K1 = I1MACH(15);
	K2 = I1MACH(16);
	R1M5 = D1MACH(5);
	K = IMIN(ABS(K1), ABS(K2));
	ELIM = 2.303*(K*R1M5 - 3.0);
	K1 = I1MACH(14) - 1;
	AA = R1M5 * 1.0*K1;
	DIG = DMIN(AA, 18.0);
	AA = AA * 2.303;
	ALIM = ELIM + DMAX(-AA, -41.45);
	FNUL = 10.0 + 6.0*(DIG - 3.0);
	RL = 1.2*DIG + 3.0;
	FN = FNU + 1.0*(NN - 1);
	MM = 3 - M - M;
	FMM = 1.0*MM;
	ZNR = FMM * ZI;
	ZNI = -FMM * ZR;
	/*----------------------------------------------------------------------
	!     TEST FOR PROPER RANGE
	!---------------------------------------------------------------------*/
	AZ = ZABS(ZR, ZI);
	AA = 0.5 / TOL;
	BB = 0.5*I1MACH(9);
	AA = DMIN(AA, BB);
	if (AZ > AA) goto e260;
	if (FN > AA) goto e260;
	AA = SQRT(AA);
	if (AZ > AA) *IERR = 3;
	if (FN > AA) *IERR = 3;
	/*----------------------------------------------------------------------
	!     OVERFLOW TEST ON THE LAST MEMBER OF THE SEQUENCE
	!---------------------------------------------------------------------*/
	UFL = EXP(-ELIM);
	if (AZ < UFL) goto e230;
	if (FNU > FNUL) goto e90;
	if (FN <= 1.0) goto e70;
	if (FN > 2.0) goto e60;
	if (AZ > TOL) goto e70;
	ARG = 0.5*AZ;
	ALN = -FN * log(ARG);
	if (ALN > ELIM) goto e230;
	goto e70;
e60:  ZUOIK(ZNR, ZNI, FNU, KODE, 2, NN, CYR, CYI, &NUF, TOL, ELIM, ALIM);
	if (NUF < 0) goto e230;
	*NZ = *NZ + NUF;
	NN = NN - NUF;
	/*----------------------------------------------------------------------
	!     HERE NN=N OR NN=0 SINCE NUF=0,NN, OR -1 ON RETURN FROM CUOIK
	!     IF NUF=NN, THEN CY(I)=CZERO FOR ALL I
	!---------------------------------------------------------------------*/
	if (NN == 0) goto e140;
e70:  if (ZNR < 0.0 || (ZNR == 0.0 && ZNI < 0.0 && M == 2)) goto e80;
	/*----------------------------------------------------------------------
	!     RIGHT HALF PLANE COMPUTATION, XN.GE.0. .AND. (XN.NE.0. .OR.
	!     YN.GE.0. .OR. M=1)
	!---------------------------------------------------------------------*/
	ZBKNU(ZNR, ZNI, FNU, KODE, NN, CYR, CYI, NZ, TOL, ELIM, ALIM);
	goto e110;
	/*----------------------------------------------------------------------
	!     LEFT HALF PLANE COMPUTATION
	!---------------------------------------------------------------------*/
e80:  MR = -MM;
	ZACON(ZNR, ZNI, FNU, KODE, MR, NN, CYR, CYI, &NW, RL, FNUL,
		TOL, ELIM, ALIM);
	if (NW < 0) goto e240;
	*NZ = NW;
	goto e110;
	/*----------------------------------------------------------------------
	!     UNIFORM ASYMPTOTIC EXPANSIONS FOR FNU.GT.FNUL
	!---------------------------------------------------------------------*/
e90:  MR = 0;
	if (ZNR >= 0.0 && (ZNR != 0.0 || ZNI >= 0.0 || M != 2)) goto e100;
	MR = -MM;
	if (ZNR != 0.0 || ZNI >= 0.0) goto e100;
	ZNR = -ZNR;
	ZNI = -ZNI;
e100: ZBUNK(ZNR, ZNI, FNU, KODE, MR, NN, CYR, CYI, &NW, TOL, ELIM, ALIM);
	printf(" ZBUNK Ok.\n");
	if (NW < 0) goto e240;
	*NZ = *NZ + NW;
	/*----------------------------------------------------------------------
	!     H(M,FNU,Z) = -FMM*(I/HPI)*(ZT^FNU)*K(FNU,-Z*ZT)
	!
	!     ZT=EXP(-FMM*HPI*I) = CMPLX(0.0,-FMM), FMM=3-2*M, M=1,2
	!---------------------------------------------------------------------*/
e110: SGN = SIGN(HPI, -FMM);
	/*----------------------------------------------------------------------
	!     CALCULATE EXP(FNU*HPI*I) TO MINIMIZE LOSSES OF SIGNIFICANCE
	!     WHEN FNU IS LARGE
	!---------------------------------------------------------------------*/
	INU = (int)floor(FNU);
	INUH = INU / 2;
	IR = INU - 2 * INUH;
	ARG = (FNU - 1.0*(INU - IR))*SGN;
	RHPI = 1.0 / SGN;
	ZNI = RHPI * COS(ARG);
	ZNR = -RHPI * SIN(ARG);
	if ((INUH % 2) == 0) goto e120;
	ZNR = -ZNR;
	ZNI = -ZNI;
e120: ZTI = -FMM;
	for (I = 1; I <= NN; I++) {
		STR = CYR[I] * ZNR - CYI[I] * ZNI;
		CYI[I] = CYR[I] * ZNI + CYI[I] * ZNR;
		CYR[I] = STR;
		STR = -ZNI * ZTI;
		ZNI = ZNR * ZTI;
		ZNR = STR;
	}
	return;
e140: if (ZNR < 0.0) goto e230;
	return;
e230: *NZ = 0;
	*IERR = 2;
	return;
e240: if (NW == -1) goto e230;
	*NZ = 0;
	*IERR = 5;
	return;
e260: *NZ = 0;
	*IERR = 4;
} //ZBESH()

#if 0
int main()
{

	REAL zr, zi;
	REAL *cyr, *cyi, *cwr, *cwi;
	int i, ierr, n, nz;

	void *vmblock = NULL;

	n = 5;

	//memory allocation for cyr, cyi, cwr, cwi
	vmblock = vminit();
	cyr = (REAL *)vmalloc(vmblock, VEKTOR, n + 1, 0); //index 0 not used
	cyi = (REAL *)vmalloc(vmblock, VEKTOR, n + 1, 0);
	cwr = (REAL *)vmalloc(vmblock, VEKTOR, n + 1, 0);
	cwi = (REAL *)vmalloc(vmblock, VEKTOR, n + 1, 0);

	if (!vmcomplete(vmblock)) {
		LogError("No Memory", 0, __FILE__, __LINE__);
		return 0;
	}

	zr = PI / 4; zi = 0.0;

	ZBESY(zr, zi, 0, 1, n, cyr, cyi, &nz, cwr, cwi, &ierr);

	printf("\n");
	for (i = 1; i <= n; i++) {
		printf(" zr(%d) = %10.6f\n", i - 1, cyr[i]);
		printf(" zi(%d) = %10.6f\n", i - 1, cyi[i]);
	}
	printf(" NZ = %d\n", nz);
	printf(" Error code: %d\n\n", ierr);
	vmfree(vmblock);
	system("pause");
}
#endif
// end of file tzbesy.cpp
