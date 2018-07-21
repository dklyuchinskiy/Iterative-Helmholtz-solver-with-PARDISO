/*************************************************************************
*            Functions used By programs TZBESJ, TZBESK, TZBESY           *
*    (Evalute Bessel Functions with complex argument, 1st to 3rd kind)   *
* ---------------------------------------------------------------------- *
* Reference:  AMOS, DONALD E., SANDIA NATIONAL LABORATORIES, 1983.       *
*                                                                        *
*                         C++ Release By J-P Moreau, Paris (07/01/2005). *
*                                     (www.jpmoreau.fr)                  *
*************************************************************************/
#include "../definitions.h"


#include "complex.h"

  //Functions defined below.
  void ZACAI(REAL, REAL, REAL, int, int, int, REAL *, REAL *, int *, REAL, 
	         REAL, REAL, REAL);

  void ZBKNU(REAL, REAL, REAL, int, int, REAL *, REAL *, int *, REAL, 
	         REAL, REAL);

  void ZKSCL(REAL, REAL, REAL, int, REAL *, REAL *, int *, REAL *, REAL *, 
	         REAL, REAL, REAL);

  //Functions defined in CBess0.cpp.
  void ZSERI(REAL, REAL, REAL, int, int, REAL *, REAL *, int *, REAL, REAL, REAL);

  void ZASYI(REAL, REAL, REAL, int, int, REAL *, REAL *, int *, REAL, REAL, REAL, REAL);

  void ZMLRI(REAL, REAL, REAL, int, int, REAL *, REAL *, int *, REAL);

  void ZS1S2(REAL *, REAL *, REAL *, REAL *, REAL *, REAL *, int *, REAL, REAL, int *);

  void ZSHCH(REAL, REAL, REAL *, REAL *, REAL *, REAL *);

  REAL DGAMLN(REAL, int *);

  void ZUCHK(REAL, REAL, int *, REAL, REAL);


void ZAIRY(REAL ZR, REAL ZI, int ID, int KODE, REAL *AIR, REAL *AII,
           int *NZ, int *IERR)  {
/***BEGIN PROLOGUE  ZAIRY
!***DATE WRITTEN   830501   (YYMMDD)
!***REVISION DATE  830501   (YYMMDD)
!***CATEGORY NO.  B5K
!***KEYWORDS  AIRY FUNCTION,BESSEL FUNCTIONS OF ORDER ONE THIRD
!***AUTHOR  AMOS, DONALD E., SANDIA NATIONAL LABORATORIES
!***PURPOSE  TO COMPUTE AIRY FUNCTIONS AI(Z) AND DAI(Z) FOR COMPLEX Z
!***DESCRIPTION
!
!                      ***A DOUBLE PRECISION ROUTINE***
!         ON KODE=1, ZAIRY COMPUTES THE COMPLEX AIRY FUNCTION AI(Z) OR
!         ITS DERIVATIVE DAI(Z)/DZ ON ID=0 OR ID=1 RESPECTIVELY. ON
!         KODE=2, A SCALING OPTION CEXP(ZTA)*AI(Z) OR CEXP(ZTA)*
!         DAI(Z)/DZ IS PROVIDED TO REMOVE THE EXPONENTIAL DECAY IN
!         -PI/3 < ARG(Z) < PI/3 AND THE EXPONENTIAL GROWTH IN
!         PI/3 < ABS(ARG(Z)) < PI, WHERE ZTA=(2/3)*Z*CSQRT(Z).
!
!         WHILE THE AIRY FUNCTIONS AI(Z) AND DAI(Z)/DZ ARE ANALYTIC IN
!         THE WHOLE Z PLANE, THE CORRESPONDING SCALED FUNCTIONS DEFINED
!         FOR KODE=2 HAVE A CUT ALONG THE NEGATIVE REAL AXIS.
!         DEFINTIONS AND NOTATION ARE FOUND IN THE NBS HANDBOOK OF
!         MATHEMATICAL FUNCTIONS (REF. 1).
!
!         INPUT      ZR,ZI ARE DOUBLE PRECISION
!           ZR,ZI  - Z=CMPLX(ZR,ZI)
!           ID     - ORDER OF DERIVATIVE, ID=0 OR ID=1
!           KODE   - A PARAMETER TO INDICATE THE SCALING OPTION
!                    KODE= 1  RETURNS
!                             AI=AI(Z)                 ON ID=0 OR
!                             AI=DAI(Z)/DZ             ON ID=1
!                        = 2  RETURNS
!                             AI=CEXP(ZTA)*AI(Z)       ON ID=0 OR
!                             AI=CEXP(ZTA)*DAI(Z)/DZ   ON ID=1 WHERE
!                             ZTA=(2/3)*Z*CSQRT(Z)
!
!         OUTPUT     AIR,AII ARE DOUBLE PRECISION
!           AIR,AII- COMPLEX ANSWER DEPENDING ON THE CHOICES FOR ID AND
!                    KODE
!           NZ     - UNDERFLOW INDICATOR
!                    NZ= 0   , NORMAL RETURN
!                    NZ= 1   , AI=CMPLX(0.0,0.0) DUE TO UNDERFLOW IN
!                              -PI/3 < ARG(Z) < PI/3 ON KODE=1.
!           IERR   - ERROR FLAG
!                    IERR=0, NORMAL RETURN - COMPUTATION COMPLETED
!                    IERR=1, INPUT ERROR   - NO COMPUTATION
!                    IERR=2, OVERFLOW      - NO COMPUTATION, REAL(ZTA)
!                            TOO LARGE ON KODE=1
!                    IERR=3, CABS(Z) LARGE      - COMPUTATION COMPLETED
!                            LOSSES OF SIGNIFCANCE BY ARGUMENT REDUCTION
!                            PRODUCE LESS THAN HALF OF MACHINE ACCURACY
!                    IERR=4, CABS(Z) TOO LARGE  - NO COMPUTATION
!                            COMPLETE LOSS OF ACCURACY BY ARGUMENT
!                            REDUCTION
!                    IERR=5, ERROR              - NO COMPUTATION,
!                            ALGORITHM TERMINATION CONDITION NOT MET
!
!***LONG DESCRIPTION
!
!         AI AND DAI ARE COMPUTED FOR CABS(Z).GT.1.0 FROM THE K BESSEL
!         FUNCTIONS BY
!
!            AI(Z)=C*SQRT(Z)*K(1/3,ZTA) , DAI(Z)=-C*Z*K(2/3,ZTA)
!                           C=1.0/(PI*SQRT(3.0))
!                           ZTA=(2/3)*Z^(3/2)
!
!         WITH THE POWER SERIES FOR CABS(Z) <= 1.
!
!         IN MOST COMPLEX VARIABLE COMPUTATION, ONE MUST EVALUATE ELE-
!         MENTARY FUNCTIONS. WHEN THE MAGNITUDE OF Z IS LARGE, LOSSES
!         OF SIGNIFICANCE BY ARGUMENT REDUCTION OCCUR. CONSEQUENTLY, IF
!         THE MAGNITUDE OF ZETA=(2/3)*Z^1.5 EXCEEDS U1=SQRT(0.5/UR),
!         THEN LOSSES EXCEEDING HALF PRECISION ARE LIKELY AND AN ERROR
!         FLAG IERR=3 IS TRIGGERED WHERE UR=DMAX(D1MACH(4),1.0D-18) IS
!         DOUBLE PRECISION UNIT ROUNDOFF LIMITED TO 18 DIGITS PRECISION.
!         ALSO, IF THE MAGNITUDE OF ZETA IS LARGER THAN U2=0.5/UR, THEN
!         ALL SIGNIFICANCE IS LOST AND IERR=4. IN ORDER TO USE THE INT
!         FUNCTION, ZETA MUST BE FURTHER RESTRICTED NOT TO EXCEED THE
!         LARGEST INTEGER, U3=I1MACH(9). THUS, THE MAGNITUDE OF ZETA
!         MUST BE RESTRICTED BY MIN(U2,U3). ON 32 BIT MACHINES, U1,U2,
!         AND U3 ARE APPROXIMATELY 2.0E+3, 4.2E+6, 2.1E+9 IN SINGLE
!         PRECISION ARITHMETIC AND 1.3E+8, 1.8E+16, 2.1E+9 IN DOUBLE
!         PRECISION ARITHMETIC RESPECTIVELY. THIS MAKES U2 AND U3 LIMIT-
!         ING IN THEIR RESPECTIVE ARITHMETICS. THIS MEANS THAT THE MAG-
!         NITUDE OF Z CANNOT EXCEED 3.1E+4 IN SINGLE AND 2.1E+6 IN
!         DOUBLE PRECISION ARITHMETIC. THIS ALSO MEANS THAT ONE CAN
!         EXPECT TO RETAIN, IN THE WORST CASES ON 32 BIT MACHINES,
!         NO DIGITS IN SINGLE PRECISION AND ONLY 7 DIGITS IN DOUBLE
!         PRECISION ARITHMETIC. SIMILAR CONSIDERATIONS HOLD FOR OTHER
!         MACHINES.
!
!         THE APPROXIMATE RELATIVE ERROR IN THE MAGNITUDE OF A COMPLEX
!         BESSEL FUNCTION CAN BE EXPRESSED BY P*10^S WHERE P=MAX(UNIT
!         ROUNDOFF,1E-18) IS THE NOMINAL PRECISION AND 10^S REPRE-
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
!***ROUTINES CALLED  ZACAI,ZBKNU,ZEXP,ZSQRT,I1MACH,D1MACH
!***END PROLOGUE  ZAIRY
!     COMPLEX AI,CONE,CSQ,CY,S1,S2,TRM1,TRM2,Z,ZTA,Z3 */
// Labels: e40,e50,e60,e70,e80,e90,e100,e110,e120,e130,e140,e150,e160,e170,
//         e180,e190,e200,e210,e260,e270,e280

      REAL AA, AD, AK, ALIM, ATRM, AZ, AZ3, BK,
      CC, CK, COEF, CONEI, CONER, CSQI, CSQR, C1, C2, DIG,
      DK, D1, D2, ELIM, FID, FNU, PTR, RL, R1M5, SFAC, STI, STR,
      S1I, S1R, S2I, S2R, TOL, TRM1I, TRM1R, TRM2I, TRM2R, TTH, ZEROI,
      ZEROR, ZTAI, ZTAR, Z3I, Z3R, ALAZ, BB;
      int IFLAG, K, K1, K2, MR, NN;
      REAL CYI[10], CYR[10];

      TTH = 6.66666666666666667e-01;
      C1  = 3.55028053887817240e-01;
      C2  = 2.58819403792806799e-01; 
      COEF= 1.83776298473930683e-01;
      ZEROR=0.0; ZEROI=0.0; CONER=1.0; CONEI=0.0;
      *IERR = 0;
      *NZ=0;
      if (ID < 0 || ID >1)  *IERR=1;
      if (KODE < 1 || KODE > 2) *IERR=1;
      if (*IERR != 0) return;
      AZ = ZABS(ZR,ZI);
      TOL = DMAX(D1MACH(4),1e-18);
      FID = 1.0*ID;
      if (AZ > 1.0) goto e70;
/*----------------------------------------------------------------------
!     POWER SERIES FOR CABS(Z).LE.1.
!---------------------------------------------------------------------*/
      S1R = CONER;
      S1I = CONEI;
      S2R = CONER;
      S2I = CONEI;
      if (AZ < TOL) goto e170;
      AA = AZ*AZ;
      if (AA < TOL/AZ) goto e40;
      TRM1R = CONER;
      TRM1I = CONEI;
      TRM2R = CONER;
      TRM2I = CONEI;
      ATRM = 1.0;
      STR = ZR*ZR - ZI*ZI;
      STI = ZR*ZI + ZI*ZR;
      Z3R = STR*ZR - STI*ZI;
      Z3I = STR*ZI + STI*ZR;
      AZ3 = AZ*AA;
      AK = 2.0 + FID;
      BK = 3.0 - FID - FID;
      CK = 4.0 - FID;
      DK = 3.0 + FID + FID;
      D1 = AK*DK;
      D2 = BK*CK;
      AD = DMIN(D1,D2);
      AK = 24.0 + 9.0*FID;
      BK = 30.0 - 9.0*FID;
      for (K=1; K<=25; K++) {
        STR = (TRM1R*Z3R-TRM1I*Z3I)/D1;
        TRM1I = (TRM1R*Z3I+TRM1I*Z3R)/D1;
        TRM1R = STR;
        S1R = S1R + TRM1R;
        S1I = S1I + TRM1I;
        STR = (TRM2R*Z3R-TRM2I*Z3I)/D2;
        TRM2I = (TRM2R*Z3I+TRM2I*Z3R)/D2;
        TRM2R = STR;
        S2R = S2R + TRM2R;
        S2I = S2I + TRM2I;
        ATRM = ATRM*AZ3/AD;
        D1 = D1 + AK;
        D2 = D2 + BK;
        AD = DMIN(D1,D2);
        if (ATRM < TOL*AD) goto e40;
        AK += 18.0;
        BK += 18.0;
      }
e40:  if (ID == 1) goto e50;
      *AIR = S1R*C1 - C2*(ZR*S2R-ZI*S2I);
      *AII = S1I*C1 - C2*(ZR*S2I+ZI*S2R);
      if (KODE == 1) return;
      ZSQRT(ZR, ZI, &STR, &STI);
      ZTAR = TTH*(ZR*STR-ZI*STI);
      ZTAI = TTH*(ZR*STI+ZI*STR);
      ZEXP(ZTAR, ZTAI, &STR, &STI);
      PTR = (*AIR)*STR - (*AII)*STI;
      *AII = (*AIR)*STI + (*AII)*STR;
      *AIR = PTR;
      return;
e50:  *AIR = -S2R*C2;
      *AII = -S2I*C2;
      if (AZ <= TOL) goto e60;
      STR = ZR*S1R - ZI*S1I;
      STI = ZR*S1I + ZI*S1R;
      CC = C1/(1.0+FID);
      *AIR = *AIR + CC*(STR*ZR-STI*ZI);
      *AII = *AII + CC*(STR*ZI+STI*ZR);
e60:  if (KODE == 1) return;
      ZSQRT(ZR, ZI, &STR, &STI);
      ZTAR = TTH*(ZR*STR-ZI*STI);
      ZTAI = TTH*(ZR*STI+ZI*STR);
      ZEXP(ZTAR, ZTAI, &STR, &STI);
      PTR = STR*(*AIR) - STI*(*AII);
      *AII = STR*(*AII) + STI*(*AIR);
      *AIR = PTR;
      return;
/*----------------------------------------------------------------------
!     CASE FOR CABS(Z).GT.1.0
!---------------------------------------------------------------------*/
e70:  FNU = (1.0+FID)/3.0;
/*----------------------------------------------------------------------
!     SET PARAMETERS RELATED TO MACHINE CONSTANTS.
!     TOL IS THE APPROXIMATE UNIT ROUNDOFF LIMITED TO 1.0D-18.
!     ELIM IS THE APPROXIMATE EXPONENTIAL OVER- AND UNDERFLOW LIMIT.
!     EXP(-ELIM).LT.EXP(-ALIM)=EXP(-ELIM)/TOL    AND
!     EXP(ELIM).GT.EXP(ALIM)=EXP(ELIM)*TOL       ARE INTERVALS NEAR
!     UNDERFLOW AND OVERFLOW LIMITS WHERE SCALED ARITHMETIC IS DONE.
!     RL IS THE LOWER BOUNDARY OF THE ASYMPTOTIC EXPANSION FOR LARGE Z.
!     DIG = NUMBER OF BASE 10 DIGITS IN TOL = 10**(-DIG).
!---------------------------------------------------------------------*/
      K1 = I1MACH(15);
      K2 = I1MACH(16);
      R1M5 = D1MACH(5);
      K = IMIN(ABS(K1),ABS(K2));
      ELIM = 2.303*(K*R1M5-3.0);
      K1 = I1MACH(14) - 1;
      AA = R1M5*K1;
      DIG = DMIN(AA,18.0);
      AA = AA*2.303;
      ALIM = ELIM + DMAX(-AA,-41.45);
      RL = 1.2*DIG + 3.0;
      ALAZ = log(AZ);
/*----------------------------------------------------------------------
!     TEST FOR PROPER RANGE
!---------------------------------------------------------------------*/
      AA=0.5/TOL;
      BB=0.5*I1MACH(9);
      AA=DMIN(AA,BB);
      AA=pow(AA,TTH);
      if (AZ > AA) goto e260;
      AA=SQRT(AA);
      if (AZ > AA)  *IERR=3;
      ZSQRT(ZR, ZI, &CSQR, &CSQI);
      ZTAR = TTH*(ZR*CSQR-ZI*CSQI);
      ZTAI = TTH*(ZR*CSQI+ZI*CSQR);
/*----------------------------------------------------------------------
!     RE(ZTA).LE.0 WHEN RE(Z).LT.0, ESPECIALLY WHEN IM(Z) IS SMALL
!---------------------------------------------------------------------*/
      IFLAG = 0;
      SFAC = 1.0;
      AK = ZTAI;
      if (ZR >= 0.0) goto e80;
      BK = ZTAR;
      CK = -ABS(BK);
      ZTAR = CK;
      ZTAI = AK;
e80:  if (ZI != 0.0) goto e90;
      if (ZR > 0.0) goto e90;
      ZTAR = 0.0;
      ZTAI = AK;
e90:  AA = ZTAR;
      if (AA >= 0.0 && ZR > 0.0) goto e110;
      if (KODE == 2) goto e100;
/*----------------------------------------------------------------------
!     OVERFLOW TEST
!---------------------------------------------------------------------*/
      if (AA > -ALIM) goto e100;
      AA = -AA + 0.25*ALAZ;
      IFLAG = 1;
      SFAC = TOL;
      if (AA > ELIM) goto e270;
/*----------------------------------------------------------------------
!     CBKNU AND CACON RETURN EXP(ZTA)*K(FNU,ZTA) ON KODE=2
!---------------------------------------------------------------------*/
e100: MR = 1;
      if (ZI < 0.0)  MR = -1;
      ZACAI(ZTAR, ZTAI, FNU, KODE, MR, 1, CYR, CYI, &NN, RL, TOL, ELIM, ALIM);
      if (NN < 0) goto e280;
      *NZ = *NZ + NN;
      goto e130;
e110: if (KODE == 2) goto e120;
/*----------------------------------------------------------------------
!     UNDERFLOW TEST
!---------------------------------------------------------------------*/
      if (AA < ALIM) goto e120;
      AA = -AA - 0.25*ALAZ;
      IFLAG = 2;
      SFAC = 1.0/TOL;
      if (AA < -ELIM) goto e210;
e120: ZBKNU(ZTAR, ZTAI, FNU, KODE, 1, CYR, CYI, NZ, TOL, ELIM, ALIM);
e130: S1R = CYR[1]*COEF;
      S1I = CYI[1]*COEF;
      if (IFLAG != 0) goto e150;
      if (ID == 1) goto e140;
      *AIR = CSQR*S1R - CSQI*S1I;
      *AII = CSQR*S1I + CSQI*S1R;
      return;
e140: *AIR = -(ZR*S1R-ZI*S1I);
      *AII = -(ZR*S1I+ZI*S1R);
      return;
e150: S1R = S1R*SFAC;
      S1I = S1I*SFAC;
      if (ID == 1) goto e160;
      STR = S1R*CSQR - S1I*CSQI;
      S1I = S1R*CSQI + S1I*CSQR;
      S1R = STR;
      *AIR = S1R/SFAC;
      *AII = S1I/SFAC;
      return;
e160: STR = -(S1R*ZR-S1I*ZI);
      S1I = -(S1R*ZI+S1I*ZR);
      S1R = STR;
      *AIR = S1R/SFAC;
      *AII = S1I/SFAC;
      return;
e170: AA = 1000*D1MACH(1);
      S1R = ZEROR;
      S1I = ZEROI;
      if (ID == 1) goto e190;
      if (AZ <= AA) goto e180;
      S1R = C2*ZR;
      S1I = C2*ZI;
e180: *AIR = C1 - S1R;
      *AII = -S1I;
      return;
e190: *AIR = -C2;
      *AII = 0.0;
      AA = SQRT(AA);
      if (AZ <= AA) goto e200;
      S1R = 0.5*(ZR*ZR-ZI*ZI);
      S1I = ZR*ZI;
e200: *AIR = *AIR + C1*S1R;
      *AII = *AII + C1*S1I;
      return;
e210: *NZ = 1;
      *AIR = ZEROR;
      *AII = ZEROI;
      return;
e270: *NZ = 0;
      *IERR=2;
      return;
e280: if (NN == -1) goto e270;
      *NZ=0;
      *IERR=5;
      return;
e260: *IERR=4;
      *NZ=0;
} //ZAIRY()


void ZACAI(REAL ZR, REAL ZI, REAL FNU, int KODE, int MR, int N, 
           REAL *YR, REAL *YI, int *NZ, REAL RL, REAL TOL, REAL ELIM, REAL ALIM) {
/***BEGIN PROLOGUE  ZACAI
!***REFER TO  ZAIRY
!
!     ZACAI APPLIES THE ANALYTIC CONTINUATION FORMULA
!
!         K(FNU,ZN*EXP(MP))=K(FNU,ZN)*EXP(-MP*FNU) - MP*I(FNU,ZN)
!                 MP=PI*MR*CMPLX(0.0,1.0)
!
!     TO CONTINUE THE K FUNCTION FROM THE RIGHT HALF TO THE LEFT
!     HALF Z PLANE FOR USE WITH ZAIRY WHERE FNU=1/3 OR 2/3 AND N=1.
!     ZACAI IS THE SAME AS ZACON WITH THE PARTS FOR LARGER ORDERS AND
!     RECURRENCE REMOVED. A RECURSIVE CALL TO ZACON CAN RESULT IF ZACON
!     IS CALLED FROM ZAIRY.
!
!***ROUTINES CALLED  ZASYI,ZBKNU,ZMLRI,ZSERI,ZS1S2,D1MACH,ZABS
!***END PROLOGUE  ZACAI
!     COMPLEX CSGN,CSPN,C1,C2,Y,Z,ZN,CY */
//Labels: e10, e20, e30, e40, e50, e60, e70, e80

      REAL ARG, ASCLE, AZ, CSGNR, CSGNI, CSPNR, CSPNI, C1R, C1I, C2R, C2I,
      DFNU, FMR, SGN, YY, ZNR, ZNI;
      int INU, IUF, NN, NW;
      REAL *CYR, *CYI;
	  void *vmblock = NULL;

//    Initialize CYR, CYI
      vmblock = vminit();  
      CYR = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0); //index 0 not used
      CYI = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0);
      if (! vmcomplete(vmblock)) {
        LogError ("No Memory", 0, __FILE__, __LINE__);
        return;
	  } 
      *NZ = 0;
      ZNR = -ZR;
      ZNI = -ZI;
      AZ = ZABS(ZR,ZI);
      NN = N;
      DFNU = FNU + 1.0*(N-1);
      if (AZ <= 2.0) goto e10;
      if (AZ*AZ*0.25 > DFNU+1.0) goto e20;
/*----------------------------------------------------------------------
!     POWER SERIES FOR THE I FUNCTION
!---------------------------------------------------------------------*/
e10:  ZSERI(ZNR, ZNI, FNU, KODE, NN, YR, YI, &NW, TOL, ELIM, ALIM);
      goto e40;
e20:  if (AZ < RL) goto e30;
/*----------------------------------------------------------------------
!     ASYMPTOTIC EXPANSION FOR LARGE Z FOR THE I FUNCTION
!---------------------------------------------------------------------*/
      ZASYI(ZNR, ZNI, FNU, KODE, NN, YR, YI, &NW, RL, TOL, ELIM, ALIM);
      if (NW < 0) goto e80;
      goto e40;
/*----------------------------------------------------------------------
!     MILLER ALGORITHM NORMALIZED BY THE SERIES FOR THE I FUNCTION
!---------------------------------------------------------------------*/
e30:  ZMLRI(ZNR, ZNI, FNU, KODE, NN, YR, YI, &NW, TOL);
      if (NW < 0) goto e80;
/*----------------------------------------------------------------------
!     ANALYTIC CONTINUATION TO THE LEFT HALF PLANE FOR THE K FUNCTION
!---------------------------------------------------------------------*/
e40:  ZBKNU(ZNR, ZNI, FNU, KODE, 1, CYR, CYI, &NW, TOL, ELIM, ALIM);
      if (NW != 0) goto e80;
      FMR = 1.0*MR;
      SGN = -SIGN(PI,FMR);
      CSGNR = 0.0;
      CSGNI = SGN;
      if (KODE == 1) goto e50;
      YY = -ZNI;
      CSGNR = -CSGNI*SIN(YY);
      CSGNI = CSGNI*COS(YY);
/*----------------------------------------------------------------------
!     CALCULATE CSPN=EXP(FNU*PI*I) TO MINIMIZE LOSSES OF SIGNIFICANCE
!     WHEN FNU IS LARGE
!---------------------------------------------------------------------*/
e50:  INU = (int) floor(FNU);
      ARG = (FNU-1.0*INU)*SGN;
      CSPNR = COS(ARG);
      CSPNI = SIN(ARG);
      if ((INU % 2) == 0) goto e60;
      CSPNR = -CSPNR;
      CSPNI = -CSPNI;
e60:  C1R = CYR[1];
      C1I = CYI[1];
      C2R = YR[1];
      C2I = YI[1];
      if (KODE == 1) goto e70;
      IUF = 0;
      ASCLE = 1000*D1MACH(1)/TOL;
      ZS1S2(&ZNR, &ZNI, &C1R, &C1I, &C2R, &C2I, &NW, ASCLE, ALIM, &IUF);
      *NZ = *NZ + NW;
e70:  YR[1] = CSPNR*C1R - CSPNI*C1I + CSGNR*C2R - CSGNI*C2I;
      YI[1] = CSPNR*C1I + CSPNI*C1R + CSGNR*C2I + CSGNI*C2R;
	  vmfree(vmblock);
      return;
e80:  *NZ = -1;
      if (NW == -2) *NZ=-2;
	  vmfree(vmblock);
} //ZACAI()


void ZBKNU(REAL ZR, REAL ZI, REAL FNU, int KODE, int N, REAL *YR, REAL *YI,
           int *NZ, REAL TOL, REAL ELIM, REAL ALIM)  {
/***BEGIN PROLOGUE  ZBKNU
!***REFER TO  ZBESI,ZBESK,ZAIRY,ZBESH
!
!     ZBKNU COMPUTES THE K BESSEL FUNCTION IN THE RIGHT HALF Z PLANE.
!
!***ROUTINES CALLED  DGAMLN,I1MACH,D1MACH,ZKSCL,ZSHCH,ZUCHK,ZABS,ZDIV,
!                    ZEXP,ZLOG,ZMLT,ZSQRT
!***END PROLOGUE  ZBKNU */
//Labels: e10,e30,e40,e50,e60,e70,e80,e90,e100,e110,e120,e130,e140,e160,e170,
//        e180,e200,e210,e215,e220,e225,e230,e240,e250,e260,e261,e262,e263,
//        e264,e270,e280,e290,e300,e310

      REAL AA, AK, ASCLE, A1, A2, BB, BK, CAZ,
      CBI, CBR, CCHI, CCHR, CKI, CKR, COEFI, COEFR, CONEI, CONER,
      CRSCR, CSCLR, CSHI, CSHR, CSI, CSR, CTWOI, CTWOR,
      CZEROI, CZEROR, CZI, CZR, DNU, DNU2, DPI, ETEST, FC, FHS,
      FI, FK, FKS, FMUI, FMUR, FPI, FR, G1, G2, HPI, PI0, PR, PTI,
      PTR, P1I, P1R, P2I, P2M, P2R, QI, QR, RAK, RCAZ, RTHPI, RZI,
      RZR, R1, S, SMUI, SMUR, SPI, STI, STR, S1I, S1R, S2I, S2R, TM,
      TTH, T1, T2, ELM, CELMR, ZDR, ZDI, AS, ALAS, HELIM;
      int I, IFLAG, INU, K, KFLAG, KK, KMAX, KODED, IDUM, J, IC, INUB, NW;
      REAL *CC, *CSSR, *CSRR, *BRY, *CYR, *CYI;
	  void *vmblock = NULL;
//    COMPLEX: Z,Y,A,B,RZ,SMU,FU,FMU,F,FLRZ,CZ,S1,S2,CSH,CCH,
//    CK,P,Q,COEF,P1,P2,CBK,PT,CZERO,CONE,CTWO,ST,EZ,CS,DK
//    Note: PI is replaced here by PI0 (PI is defined in basis.cpp as PI=3.1415926...).

      KMAX=30;
      CZEROR=0.0; CZEROI=0.0; CONER=1.0; CONEI=0.0;
      CTWOR=2.0; CTWOI=0.0; R1=2.0;

      DPI=3.14159265358979324; RTHPI=1.25331413731550025;
      SPI=1.90985931710274403; HPI=1.57079632679489662;
      FPI=1.89769999331517738; TTH=6.66666666666666666e-01;

//    Initialize CC, ..., CYI
      vmblock = vminit();  
      CC   = (REAL *) vmalloc(vmblock, VEKTOR,  9, 0);  //index 0 not used
      CSSR = (REAL *) vmalloc(vmblock, VEKTOR,  4, 0);
      CSRR = (REAL *) vmalloc(vmblock, VEKTOR,  4, 0);
      BRY  = (REAL *) vmalloc(vmblock, VEKTOR,  4, 0);
	  CYR  = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0);
	  CYI  = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0);

      if (! vmcomplete(vmblock)) {
        LogError ("No Memory", 0, __FILE__, __LINE__);
        return;
	  } 	

      CC[1]= 5.77215664901532861e-01; CC[2]=-4.20026350340952355e-02;
      CC[3]=-4.21977345555443367e-02; CC[4]= 7.21894324666309954e-03;
      CC[5]=-2.15241674114950973e-04; CC[6]=-2.01348547807882387e-05;
      CC[7]= 1.13302723198169588e-06; CC[8]= 6.11609510448141582e-09;

      CAZ = ZABS(ZR,ZI);
      CSCLR = 1.0/TOL;
      CRSCR = TOL;
      CSSR[1] = CSCLR;
      CSSR[2] = 1.0;
      CSSR[3] = CRSCR;
      CSRR[1] = CRSCR;
      CSRR[2] = 1.0;
      CSRR[3] = CSCLR;
      BRY[1] = 1000*D1MACH(1)/TOL;
      BRY[2] = 1.0/BRY[1];
      BRY[3] = D1MACH(2);
      *NZ = 0;
      IFLAG = 0;
      KODED = KODE;
      RCAZ = 1.0/CAZ;
      STR = ZR*RCAZ;
      STI = -ZI*RCAZ;
      RZR = (STR+STR)*RCAZ;
      RZI = (STI+STI)*RCAZ;
      INU = (int) floor(FNU+0.5);
      DNU = FNU - 1.0*INU;
      if (ABS(DNU) == 0.5) goto e110;
      DNU2 = 0.0;
      if (ABS(DNU) > TOL)  DNU2 = DNU*DNU;
      if (CAZ > R1) goto e110;
/*----------------------------------------------------------------------
!     SERIES FOR CABS(Z).LE.R1
!---------------------------------------------------------------------*/
      FC = 1.0;
      ZLOG(RZR, RZI, &SMUR, &SMUI, &IDUM);
      FMUR = SMUR*DNU;
      FMUI = SMUI*DNU;
      ZSHCH(FMUR, FMUI, &CSHR, &CSHI, &CCHR, &CCHI);
      if (DNU == 0.0) goto e10;
      FC = DNU*DPI;
      FC = FC/SIN(FC);
      SMUR = CSHR/DNU;
      SMUI = CSHI/DNU;
e10:  A2 = 1.0 + DNU;
/*----------------------------------------------------------------------
!     GAM(1-Z)*GAM(1+Z)=PI*Z/SIN(PI*Z), T1=1/GAM(1-DNU), T2=1/GAM(1+DNU)
!---------------------------------------------------------------------*/
      T2 = EXP(-DGAMLN(A2,&IDUM));
      T1 = 1.0/(T2*FC);
      if (ABS(DNU) > 0.1) goto e40;
/*----------------------------------------------------------------------
!     SERIES FOR F0 TO RESOLVE INDETERMINACY FOR SMALL ABS(DNU)
!---------------------------------------------------------------------*/
      AK = 1.0;
      S = CC[1];
      for (K=2; K<9; K++) {
        AK = AK*DNU2;
        TM = CC[K]*AK;
        S = S + TM;
        if (ABS(TM) < TOL) goto e30;
      }
e30:  G1 = -S;
      goto e50;
e40:  G1 = (T1-T2)/(DNU+DNU);
e50:  G2 = (T1+T2)*0.5;
      FR = FC*(CCHR*G1+SMUR*G2);
      FI = FC*(CCHI*G1+SMUI*G2);
      ZEXP(FMUR, FMUI, &STR, &STI);
      PR = 0.5*STR/T2;
      PI0 = 0.5*STI/T2;
      ZDIV(0.5, 0.0, STR, STI, &PTR, &PTI);
      QR = PTR/T1;
      QI = PTI/T1;
      S1R = FR;
      S1I = FI;
      S2R = PR;
      S2I = PI0;
      AK = 1.0;
      A1 = 1.0;
      CKR = CONER;
      CKI = CONEI;
      BK = 1.0 - DNU2;
      if (INU > 0 || N >1) goto e80;
/*----------------------------------------------------------------------
!     GENERATE K(FNU,Z), 0.0D0 .LE. FNU .LT. 0.5D0 AND N=1
!---------------------------------------------------------------------*/
      if (CAZ < TOL) goto e70;
      ZMLT(ZR, ZI, ZR, ZI, &CZR, &CZI);
      CZR = 0.25*CZR;
      CZI = 0.25*CZI;
      T1 = 0.25*CAZ*CAZ;
e60:  FR = (FR*AK+PR+QR)/BK;
      FI = (FI*AK+PI0+QI)/BK;
      STR = 1.0/(AK-DNU);
      PR = PR*STR;
      PI0 = PI0*STR;
      STR = 1.0/(AK+DNU);
      QR = QR*STR;
      QI = QI*STR;
      STR = CKR*CZR - CKI*CZI;
      RAK = 1.0/AK;
      CKI = (CKR*CZI+CKI*CZR)*RAK;
      CKR = STR*RAK;
      S1R = CKR*FR - CKI*FI + S1R;
      S1I = CKR*FI + CKI*FR + S1I;
      A1 = A1*T1*RAK;
      BK = BK + AK + AK + 1.0;
      AK = AK + 1.0;
      if (A1 > TOL) goto e60;
e70:  YR[1] = S1R;
      YI[1] = S1I;
      if (KODED == 1) {
		vmfree(vmblock);  
	    return;
      }
      ZEXP(ZR, ZI, &STR, &STI);
      ZMLT(S1R, S1I, STR, STI, &YR[1], &YI[1]);
      vmfree(vmblock);
      return;
/*----------------------------------------------------------------------
!     GENERATE K(DNU,Z) AND K(DNU+1,Z) FOR FORWARD RECURRENCE
!---------------------------------------------------------------------*/
e80:  if (CAZ < TOL) goto e100;
      ZMLT(ZR, ZI, ZR, ZI, &CZR, &CZI);
      CZR = 0.25*CZR;
      CZI = 0.25*CZI;
      T1 = 0.25*CAZ*CAZ;
e90:  FR = (FR*AK+PR+QR)/BK;
      FI = (FI*AK+PI0+QI)/BK;
      STR = 1.0/(AK-DNU);
      PR = PR*STR;
      PI0 = PI0*STR;
      STR = 1.0/(AK+DNU);
      QR = QR*STR;
      QI = QI*STR;
      STR = CKR*CZR - CKI*CZI;
      RAK = 1.0/AK;
      CKI = (CKR*CZI+CKI*CZR)*RAK;
      CKR = STR*RAK;
      S1R = CKR*FR - CKI*FI + S1R;
      S1I = CKR*FI + CKI*FR + S1I;
      STR = PR - FR*AK;
      STI = PI0 - FI*AK;
      S2R = CKR*STR - CKI*STI + S2R;
      S2I = CKR*STI + CKI*STR + S2I;
      A1 = A1*T1*RAK;
      BK = BK + AK + AK + 1.0;
      AK = AK + 1.0;
      if (A1 > TOL) goto e90;
e100: KFLAG = 2;
      A1 = FNU + 1.0;
      AK = A1*ABS(SMUR);
      if (AK > ALIM)  KFLAG = 3;
      STR = CSSR[KFLAG];
      P2R = S2R*STR;
      P2I = S2I*STR;
      ZMLT(P2R, P2I, RZR, RZI, &S2R, &S2I);
      S1R = S1R*STR;
      S1I = S1I*STR;
      if (KODED == 1) goto e210;
      ZEXP(ZR, ZI, &FR, &FI);
      ZMLT(S1R, S1I, FR, FI, &S1R, &S1I);
      ZMLT(S2R, S2I, FR, FI, &S2R, &S2I);
      goto e210;
/*----------------------------------------------------------------------
!     IFLAG=0 MEANS NO UNDERFLOW OCCURRED
!     IFLAG=1 MEANS AN UNDERFLOW OCCURRED- COMPUTATION PROCEEDS WITH
!     KODED=2 AND A TEST FOR ON SCALE VALUES IS MADE DURING FORWARD
!     RECURSION
!---------------------------------------------------------------------*/
e110: ZSQRT(ZR, ZI, &STR, &STI);
      ZDIV(RTHPI, CZEROI, STR, STI, &COEFR, &COEFI);
      KFLAG = 2;
      if (KODED == 2) goto e120;
      if (ZR > ALIM) goto e290;

      STR = EXP(-ZR)*CSSR[KFLAG];
      STI = -STR*SIN(ZI);
      STR = STR*COS(ZI);
      ZMLT(COEFR, COEFI, STR, STI, &COEFR, &COEFI);
e120: if (ABS(DNU) == 0.5) goto e300;
/*----------------------------------------------------------------------
!     MILLER ALGORITHM FOR CABS(Z).GT.R1
!---------------------------------------------------------------------*/
      AK = COS(DPI*DNU);
      AK = ABS(AK);
      if (AK == CZEROR) goto e300;
      FHS = ABS(0.25-DNU2);
      if (FHS == CZEROR) goto e300;
/*----------------------------------------------------------------------
!     COMPUTE R2=F(E). IF CABS(Z).GE.R2, USE FORWARD RECURRENCE TO
!     DETERMINE THE BACKWARD INDEX K. R2=F(E) IS A STRAIGHT LINE ON
!     12.LE.E.LE.60. E IS COMPUTED FROM 2**(-E)=B**(1-I1MACH(14))=
!     TOL WHERE B IS THE BASE OF THE ARITHMETIC.
!---------------------------------------------------------------------*/
      T1 = 1.0*(I1MACH(14)-1);
      T1 = T1*D1MACH(5)*3.321928094;
      T1 = DMAX(T1,12.0);
      T1 = DMIN(T1,60.0);
      T2 = TTH*T1 - 6.0;
      if (ZR != 0.0) goto e130;
      T1 = HPI;
      goto e140;
e130: T1 = atan(ZI/ZR);
      T1 = ABS(T1);
e140: if (T2 > CAZ) goto e170;
/*----------------------------------------------------------------------
!     FORWARD RECURRENCE LOOP WHEN CABS(Z).GE.R2
!---------------------------------------------------------------------*/
      ETEST = AK/(DPI*CAZ*TOL);
      FK = CONER;
      if (ETEST < CONER) goto e180;
      FKS = CTWOR;
      CKR = CAZ + CAZ + CTWOR;
      P1R = CZEROR;
      P2R = CONER;
      for (I=1; I<=KMAX; I++) {
        AK = FHS/FKS;
        CBR = CKR/(FK+CONER);
        PTR = P2R;
        P2R = CBR*P2R - P1R*AK;
        P1R = PTR;
        CKR = CKR + CTWOR;
        FKS = FKS + FK + FK + CTWOR;
        FHS = FHS + FK + FK;
        FK = FK + CONER;
        STR = ABS(P2R)*FK;
        if (ETEST < STR) goto e160;
      }
      goto e310;
e160: FK = FK + SPI*T1*SQRT(T2/CAZ);
      FHS = ABS(0.25-DNU2);
      goto e180;
/*----------------------------------------------------------------------
!     COMPUTE BACKWARD INDEX K FOR CABS(Z).LT.R2
!---------------------------------------------------------------------*/
e170: A2 = SQRT(CAZ);
      AK = FPI*AK/(TOL*SQRT(A2));
      AA = 3.0*T1/(1.0+CAZ);
      BB = 14.7*T1/(28.0+CAZ);
      AK = (log(AK)+CAZ*COS(AA)/(1.0+0.008*CAZ))/COS(BB);
      FK = 0.12125*AK*AK/CAZ + 1.5;
/*----------------------------------------------------------------------
!     BACKWARD RECURRENCE LOOP FOR MILLER ALGORITHM
!---------------------------------------------------------------------*/
e180: K = (int) floor(FK);
      FK = 1.0*K;
      FKS = FK*FK;
      P1R = CZEROR;
      P1I = CZEROI;
      P2R = TOL;
      P2I = CZEROI;
      CSR = P2R;
      CSI = P2I;
      for (I=1; I<=K; I++) {
        A1 = FKS - FK;
        AK = (FKS+FK)/(A1+FHS);
        RAK = 2.0/(FK+CONER);
        CBR = (FK+ZR)*RAK;
        CBI = ZI*RAK;
        PTR = P2R;
        PTI = P2I;
        P2R = (PTR*CBR-PTI*CBI-P1R)*AK;
        P2I = (PTI*CBR+PTR*CBI-P1I)*AK;
        P1R = PTR;
        P1I = PTI;
        CSR = CSR + P2R;
        CSI = CSI + P2I;
        FKS = A1 - FK + CONER;
        FK = FK - CONER;
      }
/*----------------------------------------------------------------------
!     COMPUTE (P2/CS)=(P2/CABS(CS))*(CONJG(CS)/CABS(CS)) FOR BETTER
!     SCALING
!---------------------------------------------------------------------*/
      TM = ZABS(CSR,CSI);
      PTR = 1.0/TM;
      S1R = P2R*PTR;
      S1I = P2I*PTR;
      CSR = CSR*PTR;
      CSI = -CSI*PTR;
      ZMLT(COEFR, COEFI, S1R, S1I, &STR, &STI);
      ZMLT(STR, STI, CSR, CSI, &S1R, &S1I);
      if (INU > 0 || N > 1) goto e200;
      ZDR = ZR;
      ZDI = ZI;
      if (IFLAG == 1) goto e270;
      goto e240;
/*----------------------------------------------------------------------
!     COMPUTE P1/P2=(P1/CABS(P2)*CONJG(P2)/CABS(P2) FOR SCALING
!---------------------------------------------------------------------*/
e200: TM = ZABS(P2R,P2I);
      PTR = 1.0/TM;
      P1R = P1R*PTR;
      P1I = P1I*PTR;
      P2R = P2R*PTR;
      P2I = -P2I*PTR;
      ZMLT(P1R, P1I, P2R, P2I, &PTR, &PTI);
      STR = DNU + 0.5 - PTR;
      STI = -PTI;
      ZDIV(STR, STI, ZR, ZI, &STR, &STI);
      STR = STR + 1.0;
      ZMLT(STR, STI, S1R, S1I, &S2R, &S2I);
/*----------------------------------------------------------------------
!     FORWARD RECURSION ON THE THREE TERM RECURSION WITH RELATION WITH
!     SCALING NEAR EXPONENT EXTREMES ON KFLAG=1 OR KFLAG=3
!---------------------------------------------------------------------*/
e210: STR = DNU + 1.0;
      CKR = STR*RZR;
      CKI = STR*RZI;
      if (N == 1)  INU -= 1;
      if (INU > 0) goto e220;
      if (N > 1) goto e215;
      S1R = S2R;
      S1I = S2I;
e215: ZDR = ZR;
      ZDI = ZI;
      if (IFLAG == 1) goto e270;
      goto e240;
e220: INUB = 1;
      if (IFLAG == 1) goto e261;
e225: P1R = CSRR[KFLAG];
      ASCLE = BRY[KFLAG];
      for (I=INUB; I<=INU; I++) {
        STR = S2R;
        STI = S2I;
        S2R = CKR*STR - CKI*STI + S1R;
        S2I = CKR*STI + CKI*STR + S1I;
        S1R = STR;
        S1I = STI;
        CKR = CKR + RZR;
        CKI = CKI + RZI;
        if (KFLAG >= 3) goto e230;
        P2R = S2R*P1R;
        P2I = S2I*P1R;
        STR = ABS(P2R);
        STI = ABS(P2I);
        P2M = DMAX(STR,STI);
        if (P2M <= ASCLE) goto e230;
        KFLAG++;
        ASCLE = BRY[KFLAG];
        S1R = S1R*P1R;
        S1I = S1I*P1R;
        S2R = P2R;
        S2I = P2I;
        STR = CSSR[KFLAG];
        S1R = S1R*STR;
        S1I = S1I*STR;
        S2R = S2R*STR;
        S2I = S2I*STR;
        P1R = CSRR[KFLAG];
e230:;}
      if (N != 1) goto e240;
      S1R = S2R;
      S1I = S2I;
e240: STR = CSRR[KFLAG];
      YR[1] = S1R*STR;
      YI[1] = S1I*STR;
      if (N == 1) {
		vmfree(vmblock);  
	    return;
      }
      YR[2] = S2R*STR;
      YI[2] = S2I*STR;
      if (N == 2) {
		vmfree(vmblock);  
	    return;
      }
      KK = 2;
e250: KK++;
      if (KK > N) {
		vmfree(vmblock);  
	    return;
      }
      P1R = CSRR[KFLAG];
      ASCLE = BRY[KFLAG];
      for (I=KK; I<=N; I++) {
        P2R = S2R;
        P2I = S2I;
        S2R = CKR*P2R - CKI*P2I + S1R;
        S2I = CKI*P2R + CKR*P2I + S1I;
        S1R = P2R;
        S1I = P2I;
        CKR = CKR + RZR;
        CKI = CKI + RZI;
        P2R = S2R*P1R;
        P2I = S2I*P1R;
        YR[I] = P2R;
        YI[I] = P2I;
        if (KFLAG >= 3) goto e260;
        STR = ABS(P2R);
        STI = ABS(P2I);
        P2M = DMAX(STR,STI);
        if (P2M <= ASCLE) goto e260;
        KFLAG = KFLAG + 1;
        ASCLE = BRY[KFLAG];
        S1R = S1R*P1R;
        S1I = S1I*P1R;
        S2R = P2R;
        S2I = P2I;
        STR = CSSR[KFLAG];
        S1R = S1R*STR;
        S1I = S1I*STR;
        S2R = S2R*STR;
        S2I = S2I*STR;
        P1R = CSRR[KFLAG];
e260:;} //I loop
	  vmfree(vmblock);
      return;
/*----------------------------------------------------------------------
!     IFLAG=1 CASES, FORWARD RECURRENCE ON SCALED VALUES ON UNDERFLOW
!---------------------------------------------------------------------*/
e261: HELIM = 0.5*ELIM;
      ELM = EXP(-ELIM);
      CELMR = ELM;
      ASCLE = BRY[1];
      ZDR = ZR;
      ZDI = ZI;
      IC = -1;
      J = 2;
      for (I=1; I<=INU; I++) {
        STR = S2R;
        STI = S2I;
        S2R = STR*CKR-STI*CKI+S1R;
        S2I = STI*CKR+STR*CKI+S1I;
        S1R = STR;
        S1I = STI;
        CKR = CKR+RZR;
        CKI = CKI+RZI;
        AS = ZABS(S2R,S2I);
        ALAS = log(AS);
        P2R = -ZDR+ALAS;
        if (P2R < -ELIM) goto e263;
        ZLOG(S2R,S2I,&STR,&STI,&IDUM);
        P2R = -ZDR+STR;
        P2I = -ZDI+STI;
        P2M = EXP(P2R)/TOL;
        P1R = P2M*COS(P2I);
        P1I = P2M*SIN(P2I);
        ZUCHK(P1R,P1I,&NW,ASCLE,TOL);
        if (NW != 0) goto e263;
        J = 3 - J;
        CYR[J] = P1R;
        CYI[J] = P1I;
        if (IC == I-1) goto e264;
        IC = I;
        goto e262;
e263:   if (ALAS < HELIM) goto e262;
        ZDR = ZDR-ELIM;
        S1R = S1R*CELMR;
        S1I = S1I*CELMR;
        S2R = S2R*CELMR;
        S2I = S2I*CELMR;
e262:;} // I loop
      if (N != 1) goto e270;
      S1R = S2R;
      S1I = S2I;
      goto e270;
e264: KFLAG = 1;
      INUB = I+1;
      S2R = CYR[J];
      S2I = CYI[J];
      J = 3 - J;
      S1R = CYR[J];
      S1I = CYI[J];
      if (INUB <= INU) goto e225;
      if (N != 1) goto e240;
      S1R = S2R;
      S1I = S2I;
      goto e240;
e270: YR[1] = S1R;
      YI[1] = S1I;
      if (N == 1) goto e280;
      YR[2] = S2R;
      YI[2] = S2I;
e280: ASCLE = BRY[1];
      ZKSCL(ZDR,ZDI,FNU,N,YR,YI,NZ,&RZR,&RZI,ASCLE,TOL,ELIM);
      INU = N - (*NZ);
      if (INU <= 0) {
		vmfree(vmblock);  
	    return;
      }
      KK = *NZ + 1;
      S1R = YR[KK];
      S1I = YI[KK];
      YR[KK] = S1R*CSRR[1];
      YI[KK] = S1I*CSRR[1];
      if (INU == 1) {
		vmfree(vmblock);  
	    return;
      }
      KK = *NZ + 2;
      S2R = YR[KK];
      S2I = YI[KK];
      YR[KK] = S2R*CSRR[1];
      YI[KK] = S2I*CSRR[1];
      if (INU == 2) {
		vmfree(vmblock);  
	    return;
      }
      T2 = FNU + 1.0*(KK-1);
      CKR = T2*RZR;
      CKI = T2*RZI;
      KFLAG = 1;
      goto e250;
/*----------------------------------------------------------------------
!     SCALE BY DEXP(Z), IFLAG = 1 CASES
!---------------------------------------------------------------------*/
e290: KODED = 2;
      IFLAG = 1;
      KFLAG = 2;
      goto e120;
/*----------------------------------------------------------------------
!     FNU=HALF ODD INTEGER CASE, DNU=-0.5
!---------------------------------------------------------------------*/
e300: S1R = COEFR;
      S1I = COEFI;
      S2R = COEFR;
      S2I = COEFI;
      goto e210;
e310: *NZ=-2;
	  vmfree(vmblock);  
} //ZBKNU()


void ZKSCL(REAL ZRR, REAL ZRI, REAL FNU, int N, REAL *YR, REAL *YI, int *NZ,
           REAL *RZR, REAL *RZI, REAL ASCLE, REAL TOL, REAL ELIM)  {
/***BEGIN PROLOGUE  ZKSCL
!***REFER TO  ZBESK
!
!     SET K FUNCTIONS TO ZERO ON UNDERFLOW, CONTINUE RECURRENCE
!     ON SCALED FUNCTIONS UNTIL TWO MEMBERS COME ON SCALE, THEN
!     RETURN WITH MIN(NZ+2,N) VALUES SCALED BY 1/TOL.
!
!***ROUTINES CALLED  ZUCHK,ZABS,ZLOG
!***END PROLOGUE  ZKSCL
!     COMPLEX CK,CS,CY,CZERO,RZ,S1,S2,Y,ZR,ZD,CELM */
//Labels: e10, e20, e25, e30, e40, e45

      REAL ACS, AS, CKI, CKR, CSI, CSR, FN, STR, S1I, S1R, S2I, S2R,
      ZEROI, ZEROR, ZDR, ZDI, CELMR, ELM, HELIM, ALAS;
      int I, IC, IDUM, KK, NN, NW;
      REAL *CYR, *CYI;
      void *vmblock = NULL;

//    Initialize CYR, CYI
      vmblock = vminit();  
	  CYR  = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0); //index 0 not used
	  CYI  = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0);
      if (! vmcomplete(vmblock)) {
        LogError ("No Memory", 0, __FILE__, __LINE__);
        return;
	  }
	  
      ZEROR=0.0; ZEROI=0.0;
      *NZ = 0;
      IC = 0;
      NN = IMIN(2,N);
      for (I=1; I<=NN; I++) {
        S1R = YR[I];
        S1I = YI[I];
        CYR[I] = S1R;
        CYI[I] = S1I;
        AS = ZABS(S1R,S1I);
        ACS = -ZRR + log(AS);
        *NZ = *NZ + 1;
        YR[I] = ZEROR;
        YI[I] = ZEROI;
        if (ACS < -ELIM) goto e10;
        ZLOG(S1R, S1I, &CSR, &CSI, &IDUM);
        CSR = CSR - ZRR;
        CSI = CSI - ZRI;
        STR = EXP(CSR)/TOL;
        CSR = STR*COS(CSI);
        CSI = STR*SIN(CSI);
        ZUCHK(CSR, CSI, &NW, ASCLE, TOL);
        if (NW != 0) goto e10;
        YR[I] = CSR;
        YI[I] = CSI;
        IC = I;
        *NZ = *NZ - 1;
e10: ;}
      if (N == 1) {
		vmfree(vmblock);  
	    return;
      }
      if (IC > 1) goto e20;
      YR[1] = ZEROR;
      YI[1] = ZEROI;
      *NZ = 2;
e20:  if (N == 2) {
		vmfree(vmblock);  
	    return;
      }
      if (*NZ == 0) {
		vmfree(vmblock);  
	    return;
      }
      FN = FNU + 1.0;
      CKR = FN*(*RZR);
      CKI = FN*(*RZI);
      S1R = CYR[1];
      S1I = CYI[1];
      S2R = CYR[2];
      S2I = CYI[2];
      HELIM = 0.5*ELIM;
      ELM = EXP(-ELIM);
      CELMR = ELM;
      ZDR = ZRR;
      ZDI = ZRI;

//    FIND TWO CONSECUTIVE Y VALUES ON SCALE. SCALE RECURRENCE IF
//    S2 GETS LARGER THAN EXP(ELIM/2).

      for (I=3; I<=N; I++) {
        KK = I;
        CSR = S2R;
        CSI = S2I;
        S2R = CKR*CSR - CKI*CSI + S1R;
        S2I = CKI*CSR + CKR*CSI + S1I;
        S1R = CSR;
        S1I = CSI;
        CKR = CKR + (*RZR);
        CKI = CKI + (*RZI);
        AS = ZABS(S2R,S2I);
        ALAS = log(AS);
        ACS = -ZDR + ALAS;
        *NZ = *NZ + 1;
        YR[I] = ZEROR;
        YI[I] = ZEROI;
        if (ACS < -ELIM) goto e25;
        ZLOG(S2R, S2I, &CSR, &CSI, &IDUM);
        CSR = CSR - ZDR;
        CSI = CSI - ZDI;
        STR = EXP(CSR)/TOL;
        CSR = STR*COS(CSI);
        CSI = STR*SIN(CSI);
        ZUCHK(CSR, CSI, &NW, ASCLE, TOL);
        if (NW != 0) goto e25;
        YR[I] = CSR;
        YI[I] = CSI;
        *NZ = *NZ - 1;
        if (IC == KK-1) goto e40;
        IC = KK;
        goto e30;
e25:    if (ALAS < HELIM) goto e30;
        ZDR = ZDR - ELIM;
        S1R = S1R*CELMR;
        S1I = S1I*CELMR;
        S2R = S2R*CELMR;
        S2I = S2I*CELMR;
e30:  ;}
      *NZ = N;
      if (IC == N)  *NZ = N-1;
      goto e45;
e40:  *NZ = KK - 2;
e45:  for (I=1; I<=*NZ; I++) {
        YR[I] = ZEROR;
        YI[I] = ZEROI;
      }
	  vmfree(vmblock);  
} //ZKSCL()

// end of file Cbess00.cpp
