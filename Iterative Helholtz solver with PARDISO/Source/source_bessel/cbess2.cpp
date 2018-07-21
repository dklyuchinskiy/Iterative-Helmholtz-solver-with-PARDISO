/*************************************************************************
*            Functions used By programs TZBESJ, TZBESK, TZBESY           *
*    (Evalute Bessel Functions with complex argument, 1st to 3rd kind)   *
* ---------------------------------------------------------------------- *
* Reference:  AMOS, DONALD E., SANDIA NATIONAL LABORATORIES, 1983.       *
*                                                                        *
*                         C++ Release By J-P Moreau, Paris (07/05/2005). *
*                                     (www.jpmoreau.fr)                  *
*************************************************************************/
#include "..\definitions.h"

#include "complex.h"

  //headers of functions used below
  void ZUNI1(REAL ZR, REAL ZI, REAL FNU, int KODE, int N, REAL *YR, REAL *YI,
	         int *NZ, int *NLAST, REAL FNUL, REAL TOL, REAL ELIM, REAL ALIM);

  void ZUNI2(REAL ZR, REAL ZI, REAL FNU, int KODE, int N, REAL *YR, REAL *YI,
	         int *NZ, int *NLAST, REAL FNUL, REAL TOL, REAL ELIM, REAL ALIM);

  void ZUNIK(REAL, REAL, REAL, int, int, REAL, int, REAL *, REAL *, REAL *, REAL *, 
	         REAL *, REAL *, REAL *, REAL *, REAL *, REAL *);

  void ZUCHK(REAL, REAL, int *, REAL, REAL);

  void ZUOIK(REAL, REAL, REAL, int, int, int, REAL *, REAL *, int *, REAL, REAL, REAL);

  void ZUNHJ(REAL, REAL, REAL, int,  REAL, REAL *, REAL *, REAL *, REAL *, REAL *, REAL *, 
	         REAL *, REAL *, REAL *, REAL *, REAL *, REAL *);

  void ZAIRY(REAL, REAL, int, int, REAL *, REAL *, int *, int *);


void ZBUNI(REAL ZR, REAL ZI, REAL FNU, int KODE, int N, REAL *YR, REAL *YI, 
           int *NZ, int NUI, int *NLAST, REAL FNUL, REAL TOL, REAL ELIM, REAL ALIM) {
/***BEGIN PROLOGUE  ZBUNI
!***REFER TO  ZBESI,ZBESK
!
!     ZBUNI COMPUTES THE I BESSEL FUNCTION FOR LARGE CABS(Z) >
!     FNUL AND FNU+N-1 < FNUL. THE ORDER IS INCREASED FROM
!     FNU+N-1 GREATER THAN FNUL BY ADDING NUI AND COMPUTING
!     ACCORDING TO THE UNIFORM ASYMPTOTIC EXPANSION FOR I(FNU,Z)
!     ON IFORM=1 AND THE EXPANSION FOR J(FNU,Z) ON IFORM=2.
!
!***ROUTINES CALLED  ZUNI1,ZUNI2,ZABS,D1MACH
!***END PROLOGUE  ZBUNI
!     COMPLEX CSCL,CSCR,CY,RZ,ST,S1,S2,Y,Z */
//Labels: e10,e20,e21,e25,e30,e40,e50,e60,e70,e80,e90

      REAL AX, AY, CSCLR, CSCRR, DFNU, FNUI, GNU, RAZ, RZI, RZR, STI, STR,
      S1I, S1R, S2I, S2R, ASCLE, C1R, C1I, C1M;
      int I, IFLAG, IFORM, K, NL, NW;
      REAL BRY[4];

      *NZ = 0;
      AX = ABS(ZR)*1.7321;
      AY = ABS(ZI);
      IFORM = 1;
      if (AY > AX) IFORM = 2;
      if (NUI == 0) goto e60;
      FNUI = 1.0*NUI;
      DFNU = FNU + 1.0*(N-1);
      GNU = DFNU + FNUI;
      if (IFORM == 2) goto e10;
/*----------------------------------------------------------------------
!     ASYMPTOTIC EXPANSION FOR I(FNU,Z) FOR LARGE FNU APPLIED IN
!     -PI/3 <= ARG(Z) <= PI/3
!---------------------------------------------------------------------*/
      ZUNI1(ZR, ZI, GNU, KODE, 2, YR, YI, &NW, NLAST, FNUL, TOL, ELIM, ALIM);
      goto e20;
/*----------------------------------------------------------------------
!     ASYMPTOTIC EXPANSION FOR J(FNU,Z*EXP(M*HPI)) FOR LARGE FNU
!     APPLIED IN PI/3 < ABS(ARG(Z)) <= PI/2 WHERE M=+I OR -I
!     AND HPI=PI/2
!---------------------------------------------------------------------*/
e10:  ZUNI2(ZR, ZI, GNU, KODE, 2, YR, YI, &NW, NLAST, FNUL, TOL, ELIM, ALIM);
e20:  if (NW < 0) goto e50;
      if (NW != 0) goto e90;
      STR = ZABS(YR[1],YI[1]);
/*---------------------------------------------------------------------
!     SCALE BACKWARD RECURRENCE, BRY(3) IS DEFINED BUT NEVER USED
!--------------------------------------------------------------------*/
      BRY[1]=1000*D1MACH(1)/TOL;
      BRY[2] = 1.0/BRY[1];
      BRY[3] = BRY[2];
      IFLAG = 2;
      ASCLE = BRY[2];
      CSCLR = 1.0;
      if (STR > BRY[1]) goto e21;
      IFLAG = 1;
      ASCLE = BRY[1];
      CSCLR = 1.0/TOL;
      goto e25;
e21:  if (STR < BRY[2]) goto e25;
      IFLAG = 3;
      ASCLE=BRY[3];
      CSCLR = TOL;
e25:  CSCRR = 1.0/CSCLR;
      S1R = YR[2]*CSCLR;
      S1I = YI[2]*CSCLR;
      S2R = YR[1]*CSCLR;
      S2I = YI[1]*CSCLR;
      RAZ = 1.0/ZABS(ZR,ZI);
      STR = ZR*RAZ;
      STI = -ZI*RAZ;
      RZR = (STR+STR)*RAZ;
      RZI = (STI+STI)*RAZ;
      for (I=1; I<=NUI; I++) {
        STR = S2R;
        STI = S2I;
        S2R = (DFNU+FNUI)*(RZR*STR-RZI*STI) + S1R;
        S2I = (DFNU+FNUI)*(RZR*STI+RZI*STR) + S1I;
        S1R = STR;
        S1I = STI;
        FNUI = FNUI - 1.0;
        if (IFLAG >= 3) goto e30;
        STR = S2R*CSCRR;
        STI = S2I*CSCRR;
        C1R = ABS(STR);
        C1I = ABS(STI);
        C1M = DMAX(C1R,C1I);
        if (C1M <= ASCLE) goto e30;
        IFLAG++;
        ASCLE = BRY[IFLAG];
        S1R = S1R*CSCRR;
        S1I = S1I*CSCRR;
        S2R = STR;
        S2I = STI;
        CSCLR = CSCLR*TOL;
        CSCRR = 1.0/CSCLR;
        S1R = S1R*CSCLR;
        S1I = S1I*CSCLR;
        S2R = S2R*CSCLR;
        S2I = S2I*CSCLR;
e30: ;}
      YR[N] = S2R*CSCRR;
      YI[N] = S2I*CSCRR;
      if (N == 1) return;
      NL = N - 1;
      FNUI = 1.0*NL;
      K = NL;
      for (I=1; I<=NL; I++) {
        STR = S2R;
        STI = S2I;
        S2R = (FNU+FNUI)*(RZR*STR-RZI*STI) + S1R;
        S2I = (FNU+FNUI)*(RZR*STI+RZI*STR) + S1I;
        S1R = STR;
        S1I = STI;
        STR = S2R*CSCRR;
        STI = S2I*CSCRR;
        YR[K] = STR;
        YI[K] = STI;
        FNUI = FNUI - 1.0;
        K--;
        if (IFLAG >= 3) goto e40;
        C1R = ABS(STR);
        C1I = ABS(STI);
        C1M = DMAX(C1R,C1I);
        if (C1M <= ASCLE) goto e40;
        IFLAG++;
        ASCLE = BRY[IFLAG];
        S1R = S1R*CSCRR;
        S1I = S1I*CSCRR;
        S2R = STR;
        S2I = STI;
        CSCLR = CSCLR*TOL;
        CSCRR = 1.0/CSCLR;
        S1R = S1R*CSCLR;
        S1I = S1I*CSCLR;
        S2R = S2R*CSCLR;
        S2I = S2I*CSCLR;
e40: ;} // I loop
      return;
e50:  *NZ = -1;
      if (NW == -2) *NZ=-2;
      return;
e60:  if (IFORM == 2) goto e70;
/*----------------------------------------------------------------------
!     ASYMPTOTIC EXPANSION FOR I(FNU,Z) FOR LARGE FNU APPLIED IN
!     -PI/3 <= ARG(Z) <= PI/3
!---------------------------------------------------------------------*/
      ZUNI1(ZR, ZI, FNU, KODE, N, YR, YI, &NW, NLAST, FNUL, TOL, ELIM, ALIM);
      goto e80;
/*----------------------------------------------------------------------
!     ASYMPTOTIC EXPANSION FOR J(FNU,Z*EXP(M*HPI)) FOR LARGE FNU
!     APPLIED IN PI/3 < ABS(ARG(Z)) <= PI/2 WHERE M=+I OR -I
!     AND HPI=PI/2
!---------------------------------------------------------------------*/
e70:  ZUNI2(ZR, ZI, FNU, KODE, N, YR, YI, &NW, NLAST, FNUL, TOL, ELIM, ALIM);
e80:  if (NW < 0) goto e50;
      *NZ = NW;
      return;
e90:  *NLAST = N;
} //ZBUNI()


void ZUNI1(REAL ZR, REAL ZI, REAL FNU, int KODE, int N, REAL *YR, REAL *YI,
           int *NZ, int *NLAST, REAL FNUL, REAL TOL, REAL ELIM, REAL ALIM)  {
/***BEGIN PROLOGUE  ZUNI1
!***REFER TO  ZBESI,ZBESK
!
!     ZUNI1 COMPUTES I(FNU,Z)  BY MEANS OF THE UNIFORM ASYMPTOTIC
!     EXPANSION FOR I(FNU,Z) IN -PI/3 <= ARG Z <= PI/3.
!
!     FNUL IS THE SMALLEST ORDER PERMITTED FOR THE ASYMPTOTIC
!     EXPANSION. NLAST=0 MEANS ALL OF THE Y VALUES WERE SET.
!     NLAST <> 0 IS THE NUMBER LEFT TO BE COMPUTED BY ANOTHER
!     FORMULA FOR ORDERS FNU TO FNU+NLAST-1 BECAUSE FNU+NLAST-1 < FNUL.
!     Y(I)=CZERO FOR I=NLAST+1,N.
!
!***ROUTINES CALLED  ZUCHK,ZUNIK,ZUOIK,D1MACH,ZABS
!***END PROLOGUE  ZUNI1
!     COMPLEX CFN,CONE,CRSC,CSCL,CSR,CSS,CWRK,CZERO,C1,C2,PHI,RZ,SUM,S1,
!     S2,Y,Z,ZETA1,ZETA2 */
//Labels: e10,e20,e30,e40,e50,e60,e70,e90,e100,e110,e120,e130

      REAL APHI, ASCLE, CONEI, CONER, CRSC, CSCL, C1R, C2I, C2M, C2R, FN,
      PHII, PHIR, RAST, RS1, RZI, RZR, STI, STR, SUMI, SUMR, S1I, S1R,
      S2I, S2R, ZEROI, ZEROR, ZETA1I, ZETA1R, ZETA2I, ZETA2R;
      int I, IFLAG, INIT, K, M, ND, NN, NUF, NW;
      REAL CWRKR[17], CWRKI[17];
      REAL *BRY, *CSSR, *CSRR, *CYR, *CYI;
      void *vmblock = NULL;

//    Initialize BRY, CSSR, CSRR, CYR, CYI
      vmblock = vminit(); 
	  BRY  = (REAL *) vmalloc(vmblock, VEKTOR,  4, 0);
	  CSSR = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0);
      CSRR = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0);
      CYR  = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0);
      CYI  = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0);
      if (! vmcomplete(vmblock)) {
        LogError ("No Memory", 0, __FILE__, __LINE__);
        return;
	  } 	 

      ZEROR=0.0; ZEROI=0.0; CONER=1.0; CONEI=0.0;

      *NZ = 0;
      ND = N;
      *NLAST = 0;
/*----------------------------------------------------------------------
!     COMPUTED VALUES WITH EXPONENTS BETWEEN ALIM AND ELIM IN MAG-
!     NITUDE ARE SCALED TO KEEP INTERMEDIATE ARITHMETIC ON SCALE,
!     EXP(ALIM)=EXP(ELIM)*TOL
!---------------------------------------------------------------------*/
      CSCL = 1.0/TOL;
      CRSC = TOL;
      CSSR[1] = CSCL;
      CSSR[2] = CONER;
      CSSR[3] = CRSC;
      CSRR[1] = CRSC;
      CSRR[2] = CONER;
      CSRR[3] = CSCL;
      BRY[1] = 1000*D1MACH(1)/TOL;
/*----------------------------------------------------------------------
!     CHECK FOR UNDERFLOW AND OVERFLOW ON FIRST MEMBER
!---------------------------------------------------------------------*/
      FN = DMAX(FNU,1.0);
      INIT = 0;
      ZUNIK(ZR, ZI, FN, 1, 1, TOL, INIT, &PHIR, &PHII, &ZETA1R,
            &ZETA1I, &ZETA2R, &ZETA2I, &SUMR, &SUMI, CWRKR, CWRKI);
      if (KODE == 1) goto e10;
      STR = ZR + ZETA2R;
      STI = ZI + ZETA2I;
      RAST = FN/ZABS(STR,STI);
      STR = STR*RAST*RAST;
      STI = -STI*RAST*RAST;
      S1R = -ZETA1R + STR;
      S1I = -ZETA1I + STI;
      goto e20;
e10:  S1R = -ZETA1R + ZETA2R;
      S1I = -ZETA1I + ZETA2I;
e20:  RS1 = S1R;
      if (ABS(RS1) > ELIM) goto e130;
e30:  NN = IMIN(2,ND);
      for (I=1; I<=NN; I++) {
        FN = FNU + 1.0*(ND-I);
        INIT = 0;
        ZUNIK(ZR, ZI, FN, 1, 0, TOL, INIT, &PHIR, &PHII, &ZETA1R,
              &ZETA1I, &ZETA2R, &ZETA2I, &SUMR, &SUMI, CWRKR, CWRKI);
        if (KODE == 1) goto e40;
        STR = ZR + ZETA2R;
        STI = ZI + ZETA2I;
        RAST = FN/ZABS(STR,STI);
        STR = STR*RAST*RAST;
        STI = -STI*RAST*RAST;
        S1R = -ZETA1R + STR;
        S1I = -ZETA1I + STI + ZI;
        goto e50;
e40:    S1R = -ZETA1R + ZETA2R;
        S1I = -ZETA1I + ZETA2I;
/*----------------------------------------------------------------------
!     TEST FOR UNDERFLOW AND OVERFLOW
!---------------------------------------------------------------------*/
e50:    RS1 = S1R;
        if (ABS(RS1) > ELIM) goto e110;
        if (I == 1)  IFLAG = 2;
        if (ABS(RS1) < ALIM) goto e60;
/*----------------------------------------------------------------------
!     REFINE  TEST AND SCALE
!---------------------------------------------------------------------*/
        APHI = ZABS(PHIR,PHII);
        RS1 = RS1 + log(APHI);
        if (ABS(RS1) > ELIM) goto e110;
        if (I == 1)  IFLAG = 1;
        if (RS1 < 0.0) goto e60;
        if (I == 1)  IFLAG = 3;
/*----------------------------------------------------------------------
!     SCALE S1 IF CABS(S1) < ASCLE
!---------------------------------------------------------------------*/
e60:    S2R = PHIR*SUMR - PHII*SUMI;
        S2I = PHIR*SUMI + PHII*SUMR;
        STR = EXP(S1R)*CSSR[IFLAG];
        S1R = STR*COS(S1I);
        S1I = STR*SIN(S1I);
        STR = S2R*S1R - S2I*S1I;
        S2I = S2R*S1I + S2I*S1R;
        S2R = STR;
        if (IFLAG != 1) goto e70;
        ZUCHK(S2R, S2I, &NW, BRY[1], TOL);
        if (NW != 0) goto e110;
e70:    CYR[I] = S2R;
        CYI[I] = S2I;
        M = ND - I + 1;
        YR[M] = S2R*CSRR[IFLAG];
        YI[M] = S2I*CSRR[IFLAG];
      } // i loop
      if (ND <= 2) goto e100;
      RAST = 1.0/ZABS(ZR,ZI);
      STR = ZR*RAST;
      STI = -ZI*RAST;
      RZR = (STR+STR)*RAST;
      RZI = (STI+STI)*RAST;
      BRY[2] = 1.0/BRY[1];
      BRY[3] = D1MACH(2);
      S1R = CYR[1];
      S1I = CYI[1];
      S2R = CYR[2];
      S2I = CYI[2];
      C1R = CSRR[IFLAG];
      ASCLE = BRY[IFLAG];
      K = ND - 2;
      FN = 1.0*K;
      for (I=3; I<=ND; I++) {
        C2R = S2R;
        C2I = S2I;
        S2R = S1R + (FNU+FN)*(RZR*C2R-RZI*C2I);
        S2I = S1I + (FNU+FN)*(RZR*C2I+RZI*C2R);
        S1R = C2R;
        S1I = C2I;
        C2R = S2R*C1R;
        C2I = S2I*C1R;
        YR[K] = C2R;
        YI[K] = C2I;
        K = K - 1;
        FN = FN - 1.0;
        if (IFLAG >= 3) goto e90;
        STR = ABS(C2R);
        STI = ABS(C2I);
        C2M = DMAX(STR,STI);
        if (C2M <= ASCLE) goto e90;
        IFLAG++;
        ASCLE = BRY[IFLAG];
        S1R = S1R*C1R;
        S1I = S1I*C1R;
        S2R = C2R;
        S2I = C2I;
        S1R = S1R*CSSR[IFLAG];
        S1I = S1I*CSSR[IFLAG];
        S2R = S2R*CSSR[IFLAG];
        S2I = S2I*CSSR[IFLAG];
        C1R = CSRR[IFLAG];
e90: ;} // i loop
e100: vmfree(vmblock);
	  return;
/*----------------------------------------------------------------------
!     SET UNDERFLOW AND UPDATE PARAMETERS
!---------------------------------------------------------------------*/
e110: if (RS1 > 0.0) goto e120;
      YR[ND] = ZEROR;
      YI[ND] = ZEROI;
      *NZ = *NZ + 1;
      ND--;
      if (ND == 0) goto e100;
      ZUOIK(ZR, ZI, FNU, KODE, 1, ND, YR, YI, &NUF, TOL, ELIM, ALIM);
      if (NUF < 0) goto e120;
      ND -= NUF;
      *NZ = *NZ + NUF;
      if (ND == 0) goto e100;
      FN = FNU + 1.0*(ND-1);
      if (FN >= FNUL) goto e30;
      *NLAST = ND;
	  vmfree(vmblock);
      return;
e120: *NZ = -1;
	  vmfree(vmblock);
      return;
e130: if (RS1 > 0.0) goto e120;
      *NZ = N;
      for (I=1; I<=N; I++) {
        YR[I] = ZEROR;
        YI[I] = ZEROI;
      }
	  vmfree(vmblock);
} //ZUNI1()


void ZUNI2(REAL ZR, REAL ZI, REAL FNU, int KODE, int N, REAL *YR, REAL *YI,
		   int *NZ, int *NLAST, REAL FNUL, REAL TOL, REAL ELIM, REAL ALIM) {
/***BEGIN PROLOGUE  ZUNI2
!***REFER TO  ZBESI,ZBESK
!
!     ZUNI2 COMPUTES I(FNU,Z) IN THE RIGHT HALF PLANE BY MEANS OF
!     UNIFORM ASYMPTOTIC EXPANSION FOR J(FNU,ZN) WHERE ZN IS Z*I
!     OR -Z*I AND ZN IS IN THE RIGHT HALF PLANE ALSO.
!
!     FNUL IS THE SMALLEST ORDER PERMITTED FOR THE ASYMPTOTIC
!     EXPANSION. NLAST=0 MEANS ALL OF THE Y VALUES WERE SET.
!     NLAST <> 0 IS THE NUMBER LEFT TO BE COMPUTED BY ANOTHER
!     FORMULA FOR ORDERS FNU TO FNU+NLAST-1 BECAUSE FNU+NLAST-1 < FNUL.
!     Y(I)=CZERO FOR I=NLAST+1,N.
!
!***ROUTINES CALLED  ZAIRY,ZUCHK,ZUNHJ,ZUOIK,D1MACH,ZABS
!***END PROLOGUE  ZUNI2
!     COMPLEX AI,ARG,ASUM,BSUM,CFN,CI,CID,CIP,CONE,CRSC,CSCL,CSR,CSS,
!     CZERO,C1,C2,DAI,PHI,RZ,S1,S2,Y,Z,ZB,ZETA1,ZETA2,ZN */
//Labels: e10,e20,e30,e40,e50,e60,e70,e80,e100,e110,e120,e130,e140,e150

      REAL AARG, AIC, AII, AIR, ANG, APHI, ARGI, ARGR, ASCLE, ASUMI, ASUMR,
      BSUMI, BSUMR, CIDI, CONEI, CONER, CRSC, CSCL, C1R, C2I, C2M, C2R,
      DAII, DAIR, FN, HPI, PHII, PHIR, RAST, RAZ, RS1, RZI, RZR, STI,
      STR, S1I, S1R, S2I, S2R, ZBI, ZBR, ZEROI, ZEROR, ZETA1I, ZETA1R,
      ZETA2I, ZETA2R, ZNI, ZNR;
      int I, IFLAG, IN0, INU, J, K, NAI, ND, NDAI, NN, NUF, NW, IDUM;
      REAL *BRY, *CIPR, *CIPI, *CSSR, *CSRR, *CYR, *CYI;
	  void *vmblock = NULL;

//    Initialize BRY, ..., CYR, CYI
      vmblock = vminit(); 
	  BRY  = (REAL *) vmalloc(vmblock, VEKTOR,  4, 0);
	  CIPR = (REAL *) vmalloc(vmblock, VEKTOR,  5, 0);
      CIPI = (REAL *) vmalloc(vmblock, VEKTOR,  5, 0);
	  CSSR = (REAL *) vmalloc(vmblock, VEKTOR,  4, 0);
      CSRR = (REAL *) vmalloc(vmblock, VEKTOR,  4, 0);
      CYR  = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0);
      CYI  = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0);
      if (! vmcomplete(vmblock)) {
        LogError ("No Memory", 0, __FILE__, __LINE__);
        return;
	  } 	 

      ZEROR=0.0; ZEROI=0.0; CONER=1.0; CONEI=0.0;
      CIPR[1]= 1.0; CIPI[1]=0.0; CIPR[2]=0.0; CIPI[2]= 1.0;
      CIPR[3]=-1.0; CIPI[3]=0.0; CIPR[4]=0.0; CIPI[4]=-1.0;

      HPI=1.57079632679489662; AIC=1.265512123484645396;

      *NZ = 0;
      ND = N;
      *NLAST = 0;
/*----------------------------------------------------------------------
!     COMPUTED VALUES WITH EXPONENTS BETWEEN ALIM AND ELIM IN MAG-
!     NITUDE ARE SCALED TO KEEP INTERMEDIATE ARITHMETIC ON SCALE,
!     EXP(ALIM)=EXP(ELIM)*TOL
!---------------------------------------------------------------------*/
      CSCL = 1.0/TOL;
      CRSC = TOL;
      CSSR[1] = CSCL;
      CSSR[2] = CONER;
      CSSR[3] = CRSC;
      CSRR[1] = CRSC;
      CSRR[2] = CONER;
      CSRR[3] = CSCL;
      BRY[1] = 1000*D1MACH(1)/TOL;
/*----------------------------------------------------------------------
!     ZN IS IN THE RIGHT HALF PLANE AFTER ROTATION BY CI OR -CI
!---------------------------------------------------------------------*/
      ZNR = ZI;
      ZNI = -ZR;
      ZBR = ZR;
      ZBI = ZI;
      CIDI = -CONER;
      INU = (int) floor(FNU);
      ANG = HPI*(FNU-1.0*INU);
      C2R = COS(ANG);
      C2I = SIN(ANG);
      IN0 = INU + N - 1;
      IN0 = (IN0 % 4) + 1;
      STR = C2R*CIPR[IN0] - C2I*CIPI[IN0];
      C2I = C2R*CIPI[IN0] + C2I*CIPR[IN0];
      C2R = STR;
      if (ZI > 0.0) goto e10;
      ZNR = -ZNR;
      ZBI = -ZBI;
      CIDI = -CIDI;
      C2I = -C2I;
/*----------------------------------------------------------------------
!     CHECK FOR UNDERFLOW AND OVERFLOW ON FIRST MEMBER
!---------------------------------------------------------------------*/
e10:  FN = DMAX(FNU,1.0);
      ZUNHJ(ZNR, ZNI, FN, 1, TOL, &PHIR, &PHII, &ARGR, &ARGI, &ZETA1R,
            &ZETA1I, &ZETA2R, &ZETA2I, &ASUMR, &ASUMI, &BSUMR, &BSUMI);
      if (KODE == 1) goto e20;
      STR = ZBR + ZETA2R;
      STI = ZBI + ZETA2I;
      RAST = FN/ZABS(STR,STI);
      STR = STR*RAST*RAST;
      STI = -STI*RAST*RAST;
      S1R = -ZETA1R + STR;
      S1I = -ZETA1I + STI;
      goto e30;
e20:  S1R = -ZETA1R + ZETA2R;
      S1I = -ZETA1I + ZETA2I;
e30:  RS1 = S1R;
      if (ABS(RS1) > ELIM) goto e150;
e40:  NN = IMIN(2,ND);
      for (I=1; I<=NN; I++) {
        FN = FNU + 1.0*(ND-I);
        ZUNHJ(ZNR, ZNI, FN, 0, TOL, &PHIR, &PHII, &ARGR, &ARGI, &ZETA1R, 
			  &ZETA1I, &ZETA2R, &ZETA2I, &ASUMR, &ASUMI, &BSUMR, &BSUMI);
        if (KODE == 1) goto e50;
        STR = ZBR + ZETA2R;
        STI = ZBI + ZETA2I;
        RAST = FN/ZABS(STR,STI);
        STR = STR*RAST*RAST;
        STI = -STI*RAST*RAST;
        S1R = -ZETA1R + STR;
        S1I = -ZETA1I + STI + ABS(ZI);
        goto e60;
e50:    S1R = -ZETA1R + ZETA2R;
        S1I = -ZETA1I + ZETA2I;
/*----------------------------------------------------------------------
!     TEST FOR UNDERFLOW AND OVERFLOW
!---------------------------------------------------------------------*/
e60:    RS1 = S1R;
        if (ABS(RS1) > ELIM) goto e120;
        if (I == 1)  IFLAG = 2;
        if (ABS(RS1) < ALIM) goto e70;
/*----------------------------------------------------------------------
!     REFINE  TEST AND SCALE
!---------------------------------------------------------------------*/
        APHI = ZABS(PHIR,PHII);
        AARG = ZABS(ARGR,ARGI);
        RS1 = RS1 + log(APHI) - 0.25*log(AARG) - AIC;
        if (ABS(RS1) > ELIM) goto e120;
        if (I == 1)  IFLAG = 1;
        if (RS1 < 0.0) goto e70;
        if (I == 1)  IFLAG = 3;
/*----------------------------------------------------------------------
!     SCALE S1 TO KEEP INTERMEDIATE ARITHMETIC ON SCALE NEAR
!     EXPONENT EXTREMES
!---------------------------------------------------------------------*/
e70:    ZAIRY(ARGR, ARGI, 0, 2, &AIR, &AII, &NAI, &IDUM);
        ZAIRY(ARGR, ARGI, 1, 2, &DAIR, &DAII, &NDAI, &IDUM);
        STR = DAIR*BSUMR - DAII*BSUMI;
        STI = DAIR*BSUMI + DAII*BSUMR;
        STR = STR + (AIR*ASUMR-AII*ASUMI);
        STI = STI + (AIR*ASUMI+AII*ASUMR);
        S2R = PHIR*STR - PHII*STI;
        S2I = PHIR*STI + PHII*STR;
        STR = EXP(S1R)*CSSR[IFLAG];
        S1R = STR*COS(S1I);
        S1I = STR*SIN(S1I);
        STR = S2R*S1R - S2I*S1I;
        S2I = S2R*S1I + S2I*S1R;
        S2R = STR;
        if (IFLAG != 1) goto e80;
        ZUCHK(S2R, S2I, &NW, BRY[1], TOL);
        if (NW != 0) goto e120;
e80:    if (ZI <= 0.0)  S2I = -S2I;
        STR = S2R*C2R - S2I*C2I;
        S2I = S2R*C2I + S2I*C2R;
        S2R = STR;
        CYR[I] = S2R;
        CYI[I] = S2I;
        J = ND - I + 1;
        YR[J] = S2R*CSRR[IFLAG];
        YI[J] = S2I*CSRR[IFLAG];
        STR = -C2I*CIDI;
        C2I = C2R*CIDI;
        C2R = STR;
      } // I loop
      if (ND <= 2) goto e110;
      RAZ = 1.0/ZABS(ZR,ZI);
      STR = ZR*RAZ;
      STI = -ZI*RAZ;
      RZR = (STR+STR)*RAZ;
      RZI = (STI+STI)*RAZ;
      BRY[2] = 1.0/BRY[1];
      BRY[3] = D1MACH(2);
      S1R = CYR[1];
      S1I = CYI[1];
      S2R = CYR[2];
      S2I = CYI[2];
      C1R = CSRR[IFLAG];
      ASCLE = BRY[IFLAG];
      K = ND - 2;
      FN = 1.0*K;
      for (I=3; I<=ND; I++) {
        C2R = S2R;
        C2I = S2I;
        S2R = S1R + (FNU+FN)*(RZR*C2R-RZI*C2I);
        S2I = S1I + (FNU+FN)*(RZR*C2I+RZI*C2R);
        S1R = C2R;
        S1I = C2I;
        C2R = S2R*C1R;
        C2I = S2I*C1R;
        YR[K] = C2R;
        YI[K] = C2I;
        K = K - 1;
        FN = FN - 1.0;
        if (IFLAG >= 3) goto e100;
        STR = ABS(C2R);
        STI = ABS(C2I);
        C2M = DMAX(STR,STI);
        if (C2M <= ASCLE) goto e100;
        IFLAG = IFLAG + 1;
        ASCLE = BRY[IFLAG];
        S1R = S1R*C1R;
        S1I = S1I*C1R;
        S2R = C2R;
        S2I = C2I;
        S1R = S1R*CSSR[IFLAG];
        S1I = S1I*CSSR[IFLAG];
        S2R = S2R*CSSR[IFLAG];
        S2I = S2I*CSSR[IFLAG];
        C1R = CSRR[IFLAG];
e100:;}
e110: vmfree(vmblock);
	  return;
e120: if (RS1 > 0.0) goto e140;
/*----------------------------------------------------------------------
!     SET UNDERFLOW AND UPDATE PARAMETERS
!---------------------------------------------------------------------*/
      YR[ND] = ZEROR;
      YI[ND] = ZEROI;
      *NZ = *NZ + 1;
      ND = ND - 1;
      if (ND == 0) goto e110;
      ZUOIK(ZR, ZI, FNU, KODE, 1, ND, YR, YI, &NUF, TOL, ELIM, ALIM);
      if (NUF < 0) goto e140;
      ND -= NUF;
      *NZ = *NZ + NUF;
      if (ND == 0) goto e110;
      FN = FNU + 1.0*(ND-1);
      if (FN < FNUL) goto e130;
      FN = CIDI;
      J = NUF + 1;
      K = (J % 4) + 1;
      S1R = CIPR[K];
      S1I = CIPI[K];
      if (FN < 0.0)  S1I = -S1I;
      STR = C2R*S1R - C2I*S1I;
      C2I = C2R*S1I + C2I*S1R;
      C2R = STR;
      goto e40;
e130: *NLAST = ND;
      vmfree(vmblock);
      return;
e140: *NZ = -1;
	  vmfree(vmblock);
      return;
e150: if (RS1 < 0.0) goto e140;
      *NZ = N;
      for (I=1; I<=N; I++) {
        YR[I] = ZEROR;
        YI[I] = ZEROI;
      }
	  vmfree(vmblock);
} // ZUNI2()

//end of file cbess2.cpp
