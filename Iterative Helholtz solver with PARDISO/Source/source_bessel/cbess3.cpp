/*************************************************************************
*    Procedures and Functions used By programs TZBESJ, TZBESK, TZBESY    *
*    (Evalute Bessel Functions with complex argument, 1st to 3rd kind)   *
* ---------------------------------------------------------------------- *
* Reference:  AMOS, DONALD E., SANDIA NATIONAL LABORATORIES, 1983.       *
*                                                                        *
*                         C++ Release By J-P Moreau, Paris (07/10/2005). *
*                                     (www.jpmoreau.fr)                  *
*************************************************************************/
#include "../definitions.h"

#include "complex.h"

  //Headers of functions used below
  void ZACON(REAL ZR, REAL ZI, REAL FNU, int KODE, int MR, int N, REAL *YR, 
             REAL *YI, int *NZ, REAL RL, REAL FNUL, REAL TOL, REAL ELIM, REAL ALIM);

  void ZUNK1(REAL ZR, REAL ZI, REAL FNU, int KODE, int MR, int N, REAL *YR, 
             REAL *YI, int *NZ, REAL TOL, REAL ELIM, REAL ALIM);

  void ZUNK2(REAL ZR, REAL ZI, REAL FNU, int KODE, int MR, int N, REAL *YR, 
             REAL *YI, int *NZ, REAL TOL, REAL ELIM, REAL ALIM);

  void ZUNIK(REAL, REAL, REAL, int, int, REAL, int, REAL *, REAL *, REAL *, REAL *, 
	         REAL *, REAL *, REAL *, REAL *, REAL *, REAL *);

  void ZUCHK(REAL, REAL, int *, REAL, REAL);

  void ZS1S2(REAL *, REAL *, REAL *, REAL *, REAL *, REAL *, int *, REAL, REAL, int *);

  void ZUNHJ(REAL, REAL, REAL, int,  REAL, REAL *, REAL *, REAL *, REAL *, REAL *, REAL *, 
	         REAL *, REAL *, REAL *, REAL *, REAL *, REAL *);

  void ZAIRY(REAL, REAL, int, int, REAL *, REAL *, int *, int *);

  void ZBINU(REAL, REAL, REAL, int, int, REAL *, REAL *, int *, REAL, REAL, REAL, 
	         REAL, REAL); 

  void ZBKNU(REAL, REAL, REAL, int, int, REAL *, REAL *, int *, REAL, 
	         REAL, REAL);


void ZBUNK(REAL ZR, REAL ZI, REAL FNU, int KODE, int MR, int N, REAL *YR, REAL *YI,
           int *NZ, REAL TOL, REAL ELIM, REAL ALIM)  {
/***BEGIN PROLOGUE  ZBUNK
!***REFER TO  ZBESK,ZBESH
!
!     ZBUNK COMPUTES THE K BESSEL FUNCTION FOR FNU.GT.FNUL.
!     ACCORDING TO THE UNIFORM ASYMPTOTIC EXPANSION FOR K(FNU,Z)
!     IN ZUNK1 AND THE EXPANSION FOR H(2,FNU,Z) IN ZUNK2
!
!***ROUTINES CALLED  ZUNK1,ZUNK2
!***END PROLOGUE  ZBUNK
!     COMPLEX Y,Z */
//Label: e10

      REAL AX, AY;

      *NZ = 0;
      AX = ABS(ZR)*1.7321;
      AY = ABS(ZI);
      if (AY > AX) goto e10;
/*----------------------------------------------------------------------
!     ASYMPTOTIC EXPANSION FOR K(FNU,Z) FOR LARGE FNU APPLIED IN
!     -PI/3 <= ARG(Z) <= PI/3
!---------------------------------------------------------------------*/
      ZUNK1(ZR, ZI, FNU, KODE, MR, N, YR, YI, NZ, TOL, ELIM, ALIM);
      return;
/*----------------------------------------------------------------------
!     ASYMPTOTIC EXPANSION FOR H(2,FNU,Z*EXP(M*HPI)) FOR LARGE FNU
!     APPLIED IN PI/3.LT.ABS(ARG(Z)).LE.PI/2 WHERE M=+I OR -I
!     AND HPI=PI/2
!---------------------------------------------------------------------*/
e10:  ZUNK2(ZR, ZI, FNU, KODE, MR, N, YR, YI, NZ, TOL, ELIM, ALIM);
} //ZBUNK()


void ZUNK1(REAL ZR, REAL ZI, REAL FNU, int KODE, int MR, int N, REAL *YR, 
           REAL *YI, int *NZ, REAL TOL, REAL ELIM, REAL ALIM) {
/***BEGIN PROLOGUE  ZUNK1
!***REFER TO  ZBESK
!
!     ZUNK1 COMPUTES K(FNU,Z) AND ITS ANALYTIC CONTINUATION FROM THE
!     RIGHT HALF PLANE TO THE LEFT HALF PLANE BY MEANS OF THE
!     UNIFORM ASYMPTOTIC EXPANSION.
!     MR INDICATES THE DIRECTION OF ROTATION FOR ANALYTIC CONTINUATION.
!     NZ=-1 MEANS AN OVERFLOW WILL OCCUR
!
!***ROUTINES CALLED  ZKSCL,ZS1S2,ZUCHK,ZUNIK,D1MACH,ZABS
!***END PROLOGUE  ZUNK1
!     COMPLEX CFN,CK,CONE,CRSC,CS,CSCL,CSGN,CSPN,CSR,CSS,CWRK,CY,CZERO,
!     C1,C2,PHI,PHID,RZ,SUM,SUMD,S1,S2,Y,Z,ZETA1,ZETA1D,ZETA2,ZETA2D,ZR */
//Labels: e10,e20,e30,e40,e50,e60,e70,e75,e80,e90,e95,e100,e120,e160,e170,
//        e172,e175,e180,e200,e210,e220,e230,e250,e255,e260,e270,e275,e280,
//        e290,e300

      REAL ANG, APHI, ASC, ASCLE, CKI, CKR, CONEI, CONER, CRSC, CSCL, CSGNI,
      CSPNI, CSPNR, CSR, C1I, C1R, C2I, C2M, C2R, FMR, FN, FNF, PHIDI,
      PHIDR, RAST, RAZR, RS1, RZI, RZR, SGN, STI, STR, SUMDI, SUMDR,
      S1I, S1R, S2I, S2R, ZEROI, ZEROR, ZET1DI, ZET1DR, ZET2DI, ZET2DR,
      ZRI, ZRR;
      int I, IB, IFLAG, IFN, II, IL, INU, IUF, K, KDFLG, KFLAG, KK, NW, INITD,
      IC, IPARD, J, M;
      REAL *BRY, *SUMR, *SUMI, *ZETA1R, *ZETA1I, *ZETA2R, *ZETA2I, *CYR, *CYI;
      int  *INIT;          //size 0..3
      REAL **CWRKR, **CWRKI; //size 0..16, 0..3     
	  REAL *CSSR, *CSRR, *PHIR, *PHII;
      REAL *TMPR, *TMPI;   //size 0..16
      void *vmblock = NULL;

//***  First executable statement ZUNK1

      //initialize pointers to vectors
      vmblock = vminit();  
      BRY  = (REAL *) vmalloc(vmblock, VEKTOR,  4, 0); //index 0 not used
      SUMR = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0);
      SUMI = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0);
      ZETA1R = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0);
      ZETA1I = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0);
      ZETA2R = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0);
      ZETA2I = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0);
      CYR  = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0);
      CYI  = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0);
      INIT = (int *) vmalloc(vmblock, VEKTOR,  4, 0);
      CWRKR = (REAL **) vmalloc(vmblock, MATRIX,  17, 4);
      CWRKI = (REAL **) vmalloc(vmblock, MATRIX,  17, 4);
      TMPR = (REAL *) vmalloc(vmblock, VEKTOR,  17, 0);
      TMPI = (REAL *) vmalloc(vmblock, VEKTOR,  17, 0);
      CSSR = (REAL *) vmalloc(vmblock, VEKTOR,  4, 0);
      CSRR = (REAL *) vmalloc(vmblock, VEKTOR,  4, 0);
      PHIR = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0);
      PHII = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0);

      if (! vmcomplete(vmblock)) {
        LogError ("No Memory", 0, __FILE__, __LINE__);
        return;
	  }
	  
	  printf(" ZUNK1:allocations Ok.\n"); getchar();

      ZEROR=0.0; ZEROI=0.0; CONER=1.0; CONEI=0.0;
      KDFLG = 1;
      *NZ = 0;
/*----------------------------------------------------------------------
!     EXP(-ALIM)=EXP(-ELIM)/TOL=APPROX. ONE PRECISION GREATER THAN
!     THE UNDERFLOW LIMIT
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
      BRY[2] = 1.0/BRY[1];
      BRY[3] = D1MACH(2);
      ZRR = ZR;
      ZRI = ZI;
      if (ZR >= 0.0) goto e10;
      ZRR = -ZR;
      ZRI = -ZI;
e10:  J = 2;
      for (I=1; I<=N; I++) {
/*----------------------------------------------------------------------
!     J FLIP FLOPS BETWEEN 1 AND 2 IN J = 3 - J
!---------------------------------------------------------------------*/
        J = 3 - J;
        FN = FNU + 1.0*(I-1);
        INIT[J] = 0;
        for (II=1; II<=16; II++) {
          TMPR[II]=CWRKR[II][J];
          TMPI[II]=CWRKI[II][J];
        }
        ZUNIK(ZRR, ZRI, FN, 2, 0, TOL, INIT[J], &PHIR[J], &PHII[J],
              &ZETA1R[J], &ZETA1I[J], &ZETA2R[J], &ZETA2I[J], &SUMR[J], &SUMI[J],
              TMPR, TMPI);
        for (II=1; II<=16; II++) {
          CWRKR[II][J]=TMPR[II];
          CWRKI[II][J]=TMPI[II];
        }
        if (KODE == 1) goto e20;
        STR = ZRR + ZETA2R[J];
        STI = ZRI + ZETA2I[J] ;
        RAST = FN/ZABS(STR,STI);
        STR = STR*RAST*RAST;
        STI = -STI*RAST*RAST;
        S1R = ZETA1R[J] - STR;
        S1I = ZETA1I[J] - STI;
        goto e30;
e20:    S1R = ZETA1R[J] - ZETA2R[J];
        S1I = ZETA1I[J] - ZETA2I[J];
e30:    RS1 = S1R;
/*----------------------------------------------------------------------
!     TEST FOR UNDERFLOW AND OVERFLOW
!---------------------------------------------------------------------*/
        if (ABS(RS1) > ELIM) goto e60;
        if (KDFLG == 1)  KFLAG = 2;
        if (ABS(RS1) < ALIM)  goto e40;
/*----------------------------------------------------------------------
!     REFINE  TEST AND SCALE
!---------------------------------------------------------------------*/
        APHI = ZABS(PHIR[J],PHII[J]);
        RS1 = RS1 + log(APHI);
        if (ABS(RS1) > ELIM) goto e60;
        if (KDFLG == 1) KFLAG = 1;
        if (RS1 < 0.0)  goto e40;
        if (KDFLG == 1) KFLAG = 3;
/*----------------------------------------------------------------------
!     SCALE S1 TO KEEP INTERMEDIATE ARITHMETIC ON SCALE NEAR
!     EXPONENT EXTREMES
!---------------------------------------------------------------------*/
e40:    S2R = PHIR[J]*SUMR[J] - PHII[J]*SUMI[J];
        S2I = PHIR[J]*SUMI[J] + PHII[J]*SUMR[J];
        STR = EXP(S1R)*CSSR[KFLAG];
        S1R = STR*COS(S1I);
        S1I = STR*SIN(S1I);
        STR = S2R*S1R - S2I*S1I;
        S2I = S1R*S2I + S2R*S1I;
        S2R = STR;
        if (KFLAG != 1) goto e50;
        ZUCHK(S2R, S2I, &NW, BRY[1], TOL);
        if (NW != 0) goto e60;
e50:    CYR[KDFLG] = S2R;
        CYI[KDFLG] = S2I;
        YR[I] = S2R*CSRR[KFLAG];
        YI[I] = S2I*CSRR[KFLAG];
        if (KDFLG == 2) goto e75;
        KDFLG = 2;
        goto e70;
e60:    if (RS1 > 0.0) goto e300;
/*----------------------------------------------------------------------
!     FOR ZR.LT.0.0, THE I FUNCTION TO BE ADDED WILL OVERFLOW
!---------------------------------------------------------------------*/
        if (ZR < 0.0) goto e300;
        KDFLG = 1;
        YR[I]=ZEROR;
        YI[I]=ZEROI;
        NZ=NZ+1;
        if (I == 1) goto e70;
        if (YR[I-1] == ZEROR && YI[I-1] == ZEROI) goto e70;
        YR[I-1]=ZEROR;
        YI[I-1]=ZEROI;
        *NZ=*NZ+1;
e70: ;} // I loop
      I = N;
e75:  RAZR = 1.0/ZABS(ZRR,ZRI);
      STR = ZRR*RAZR;
      STI = -ZRI*RAZR;
      RZR = (STR+STR)*RAZR;
      RZI = (STI+STI)*RAZR;
      CKR = FN*RZR;
      CKI = FN*RZI;
      IB = I + 1;
      if (N < IB) goto e160;
/*----------------------------------------------------------------------
!     TEST LAST MEMBER FOR UNDERFLOW AND OVERFLOW. SET SEQUENCE TO ZERO
!     ON UNDERFLOW.
!---------------------------------------------------------------------*/
      FN = FNU + 1.0*(N-1);
      IPARD = 1;
      if (MR != 0)  IPARD = 0;
      INITD = 0;
      for (II=1; II<=16; II++) {
        TMPR[II]=CWRKR[II][3];
        TMPI[II]=CWRKI[II][3];
      }
      ZUNIK(ZRR, ZRI, FN, 2, IPARD, TOL, INITD, &PHIDR, &PHIDI, &ZET1DR,
            &ZET1DI, &ZET2DR, &ZET2DI, &SUMDR, &SUMDI, TMPR, TMPI);
      for (II=1; II<=16; II++) {
        CWRKR[II][3]=TMPR[II];
        CWRKI[II][3]=TMPI[II];
      }
      if (KODE == 1) goto e80;
      STR = ZRR + ZET2DR;
      STI = ZRI + ZET2DI;
      RAST = FN/ZABS(STR,STI);
      STR = STR*RAST*RAST;
      STI = -STI*RAST*RAST;
      S1R = ZET1DR - STR;
      S1I = ZET1DI - STI;
      goto e90;
e80:  S1R = ZET1DR - ZET2DR;
      S1I = ZET1DI - ZET2DI;
e90:  RS1 = S1R;
      if (ABS(RS1) > ELIM) goto e95;
      if (ABS(RS1) < ALIM) goto e100;
/*----------------------------------------------------------------------
!     REFINE ESTIMATE AND TEST
!---------------------------------------------------------------------*/
      APHI = ZABS(PHIDR,PHIDI);
      RS1 = RS1+log(APHI);
      if (ABS(RS1) < ELIM) goto e100;
e95:  if (ABS(RS1) > 0.0) goto e300;
/*----------------------------------------------------------------------
!     FOR ZR.LT.0.0, THE I FUNCTION TO BE ADDED WILL OVERFLOW
!---------------------------------------------------------------------*/
      if (ZR < 0.0) goto e300;
      *NZ = N;
      for (I=1; I<=N; I++) {
        YR[I] = ZEROR;
        YI[I] = ZEROI;
      }
	  vmfree(vmblock);  
      return;
/*----------------------------------------------------------------------
!     FORWARD RECUR FOR REMAINDER OF THE SEQUENCE
!---------------------------------------------------------------------*/
e100: S1R = CYR[1];
      S1I = CYI[1];
      S2R = CYR[2];
      S2I = CYI[2];
      C1R = CSRR[KFLAG];
      ASCLE = BRY[KFLAG];
      for (I=IB; I<=N; I++) {
        C2R = S2R;
        C2I = S2I;
        S2R = CKR*C2R - CKI*C2I + S1R;
        S2I = CKR*C2I + CKI*C2R + S1I;
        S1R = C2R;
        S1I = C2I;
        CKR = CKR + RZR;
        CKI = CKI + RZI;
        C2R = S2R*C1R;
        C2I = S2I*C1R;
        YR[I] = C2R;
        YI[I] = C2I;
        if (KFLAG >= 3) goto e120;
        STR = ABS(C2R);
        STI = ABS(C2I);
        C2M = DMAX(STR,STI);
        if (C2M <= ASCLE) goto e120;
        KFLAG++;
        ASCLE = BRY[KFLAG];
        S1R = S1R*C1R;
        S1I = S1I*C1R;
        S2R = C2R;
        S2I = C2I;
        S1R = S1R*CSSR[KFLAG];
        S1I = S1I*CSSR[KFLAG];
        S2R = S2R*CSSR[KFLAG];
        S2I = S2I*CSSR[KFLAG];
        C1R = CSRR[KFLAG];
e120:;} // I loop
e160: if (MR == 0) {
		vmfree(vmblock);  
	    return;
      }
/*----------------------------------------------------------------------
!     ANALYTIC CONTINUATION FOR RE(Z).LT.0.0D0
!---------------------------------------------------------------------*/
      *NZ = 0;
      FMR = 1.0*MR;
      SGN = -SIGN(PI,FMR);
/*----------------------------------------------------------------------
!     CSPN AND CSGN ARE COEFF OF K AND I FUNCTIONS RESP.
!---------------------------------------------------------------------*/
      CSGNI = SGN;
      INU = (int) floor(FNU);
      FNF = FNU - 1.0*INU;
      IFN = INU + N - 1;
      ANG = FNF*SGN;
      CSPNR = COS(ANG);
      CSPNI = SIN(ANG);
      if ((IFN % 2) == 0) goto e170;
      CSPNR = -CSPNR;
      CSPNI = -CSPNI;
e170: ASC = BRY[1];
      IUF = 0;
      KK = N;
      KDFLG = 1;
      IB = IB - 1;
      IC = IB - 1;
      for (K=1; K<=N; K++) {
        FN = FNU + 1.0*(KK-1);
/*----------------------------------------------------------------------
!     LOGIC TO SORT OUT CASES WHOSE PARAMETERS WERE SET FOR THE K
!     FUNCTION ABOVE
!---------------------------------------------------------------------*/
        M=3;
        if (N > 2) goto e175;
e172:   INITD = INIT[J];
        PHIDR = PHIR[J];
        PHIDI = PHII[J];
        ZET1DR = ZETA1R[J];
        ZET1DI = ZETA1I[J];
        ZET2DR = ZETA2R[J];
        ZET2DI = ZETA2I[J];
        SUMDR = SUMR[J];
        SUMDI = SUMI[J];
        M = J;
        J = 3 - J;
        goto e180;
e175:   if (KK == N && IB < N) goto e180;
        if (KK == IB || KK == IC) goto e172;
        INITD = 0;
e180:   for (II=1; II<=16; II++) {
          TMPR[II]=CWRKR[II][M];
          TMPI[II]=CWRKI[II][M];
        }
        ZUNIK(ZRR, ZRI, FN, 1, 0, TOL, INITD, &PHIDR, &PHIDI,
              &ZET1DR, &ZET1DI, &ZET2DR, &ZET2DI, &SUMDR, &SUMDI,
              TMPR,TMPI);
        for (II=1; II<=16; II++) {
          CWRKR[II][M]=TMPR[II];
          CWRKI[II][M]=TMPI[II];
        }
        if (KODE == 1) goto e200;
        STR = ZRR + ZET2DR;
        STI = ZRI + ZET2DI;
        RAST = FN/ZABS(STR,STI);
        STR = STR*RAST*RAST;
        STI = -STI*RAST*RAST;
        S1R = -ZET1DR + STR;
        S1I = -ZET1DI + STI;
        goto e210;
e200:   S1R = -ZET1DR + ZET2DR;
        S1I = -ZET1DI + ZET2DI;
/*----------------------------------------------------------------------
!     TEST FOR UNDERFLOW AND OVERFLOW
!---------------------------------------------------------------------*/
e210:   RS1 = S1R;
        if (ABS(RS1) > ELIM) goto e260;
        if (KDFLG == 1)  IFLAG = 2;
        if (ABS(RS1) < ALIM) goto e220;
/*----------------------------------------------------------------------
!     REFINE  TEST AND SCALE
!---------------------------------------------------------------------*/
        APHI = ZABS(PHIDR,PHIDI);
        RS1 = RS1 + log(APHI);
        if (ABS(RS1) > ELIM) goto e260;
        if (KDFLG == 1) IFLAG = 1;
        if (RS1 < 0.0) goto e220;
        if (KDFLG == 1) IFLAG = 3;
e220:   STR = PHIDR*SUMDR - PHIDI*SUMDI;
        STI = PHIDR*SUMDI + PHIDI*SUMDR;
        S2R = -CSGNI*STI;
        S2I = CSGNI*STR;
        STR = EXP(S1R)*CSSR[IFLAG];
        S1R = STR*COS(S1I);
        S1I = STR*SIN(S1I);
        STR = S2R*S1R - S2I*S1I;
        S2I = S2R*S1I + S2I*S1R;
        S2R = STR;
        if (IFLAG != 1) goto e230;
        ZUCHK(S2R, S2I, &NW, BRY[1], TOL);
        if (NW == 0) goto e230;
        S2R = ZEROR;
        S2I = ZEROI;
e230:   CYR[KDFLG] = S2R;
        CYI[KDFLG] = S2I;
        C2R = S2R;
        C2I = S2I;
        S2R = S2R*CSRR[IFLAG];
        S2I = S2I*CSRR[IFLAG];
/*----------------------------------------------------------------------
!     ADD I AND K FUNCTIONS, K SEQUENCE IN Y(I), I=1,N
!---------------------------------------------------------------------*/
        S1R = YR[KK];
        S1I = YI[KK];
        if (KODE == 1) goto e250;
        ZS1S2(&ZRR, &ZRI, &S1R, &S1I, &S2R, &S2I, &NW, ASC, ALIM, &IUF);
        *NZ = *NZ + NW;
e250:   YR[KK] = S1R*CSPNR - S1I*CSPNI + S2R;
        YI[KK] = CSPNR*S1I + CSPNI*S1R + S2I;
        KK = KK - 1;
        CSPNR = -CSPNR;
        CSPNI = -CSPNI;
        if (C2R != 0.0  || C2I != 0.0)  goto e255;
        KDFLG = 1;
        goto e270;
e255:   if (KDFLG == 2) goto e275;
        KDFLG = 2;
        goto e270;
e260:   if (RS1 > 0.0) goto e300;
        S2R = ZEROR;
        S2I = ZEROI;
        goto e230;
e270:;} // K loop
      K = N;
e275: IL = N - K;
      if (IL == 0) {
		vmfree(vmblock);  
	    return;
      }
/*----------------------------------------------------------------------
!     RECUR BACKWARD FOR REMAINDER OF I SEQUENCE AND ADD IN THE
!     K FUNCTIONS, SCALING THE I SEQUENCE DURING RECURRENCE TO KEEP
!     INTERMEDIATE ARITHMETIC ON SCALE NEAR EXPONENT EXTREMES.
!---------------------------------------------------------------------*/
      S1R = CYR[1];
      S1I = CYI[1];
      S2R = CYR[2];
      S2I = CYI[2];
      CSR = CSRR[IFLAG];
      ASCLE = BRY[IFLAG];
      FN = 1.0*(INU+IL);
      for (I=1; I<=IL; I++) {
        C2R = S2R;
        C2I = S2I;
        S2R = S1R + (FN+FNF)*(RZR*C2R-RZI*C2I);
        S2I = S1I + (FN+FNF)*(RZR*C2I+RZI*C2R);
        S1R = C2R;
        S1I = C2I;
        FN = FN - 1.0;
        C2R = S2R*CSR;
        C2I = S2I*CSR;
        CKR = C2R;
        CKI = C2I;
        C1R = YR[KK];
        C1I = YI[KK];
        if (KODE == 1) goto e280;
        ZS1S2(&ZRR, &ZRI, &C1R, &C1I, &C2R, &C2I, &NW, ASC, ALIM, &IUF);
        *NZ = *NZ + NW;
e280:   YR[KK] = C1R*CSPNR - C1I*CSPNI + C2R;
        YI[KK] = C1R*CSPNI + C1I*CSPNR + C2I;
        KK--;
        CSPNR = -CSPNR;
        CSPNI = -CSPNI;
        if (IFLAG >= 3) goto e290;
        C2R = ABS(CKR);
        C2I = ABS(CKI);
        C2M = DMAX(C2R,C2I);
        if (C2M <= ASCLE) goto e290;
        IFLAG++;
        ASCLE = BRY[IFLAG];
        S1R = S1R*CSR;
        S1I = S1I*CSR;
        S2R = CKR;
        S2I = CKI;
        S1R = S1R*CSSR[IFLAG];
        S1I = S1I*CSSR[IFLAG];
        S2R = S2R*CSSR[IFLAG];
        S2I = S2I*CSSR[IFLAG];
        CSR = CSRR[IFLAG];
e290:;} // I loop
	  vmfree(vmblock);
      return;
e300: *NZ = -1;
      vmfree(vmblock);
} //ZUNK1()


void ZUNK2(REAL ZR, REAL ZI, REAL FNU, int KODE, int MR, int N, REAL *YR, 
           REAL *YI, int *NZ, REAL TOL, REAL ELIM, REAL ALIM)  {
/***BEGIN PROLOGUE  ZUNK2
!***REFER TO  ZBESK
!
!     ZUNK2 COMPUTES K(FNU,Z) AND ITS ANALYTIC CONTINUATION FROM THE
!     RIGHT HALF PLANE TO THE LEFT HALF PLANE BY MEANS OF THE
!     UNIFORM ASYMPTOTIC EXPANSIONS FOR H(KIND,FNU,ZN) AND J(FNU,ZN)
!     WHERE ZN IS IN THE RIGHT HALF PLANE, KIND=(3-MR)/2, MR=+1 OR
!     -1. HERE ZN=ZR*I OR -ZR*I WHERE ZR=Z IF Z IS IN THE RIGHT
!     HALF PLANE OR ZR=-Z IF Z IS IN THE LEFT HALF PLANE. MR INDIC-
!     ATES THE DIRECTION OF ROTATION FOR ANALYTIC CONTINUATION.
!     NZ=-1 MEANS AN OVERFLOW WILL OCCUR
!
!***ROUTINES CALLED  ZAIRY,ZKSCL,ZS1S2,ZUCHK,ZUNHJ,D1MACH,ZABS
!***END PROLOGUE  ZUNK2
!     COMPLEX AI,ARG,ARGD,ASUM,ASUMD,BSUM,BSUMD,CFN,CI,CIP,CK,CONE,CRSC,
!     CR1,CR2,CS,CSCL,CSGN,CSPN,CSR,CSS,CY,CZERO,C1,C2,DAI,PHI,PHID,RZ,
!     S1,S2,Y,Z,ZB,ZETA1,ZETA1D,ZETA2,ZETA2D,ZN,ZR */
//Labels: e10,e20,e30,e40,e50,e60,e70,e80,e85,e90,e100,e105,e120,e130,e172,
//	      e175,e180,e190,e210,e220,e230,e240,e250,e255,e270,e280,e290,e295,
//	      e300,e310,e320

      REAL AARG, AIC, AII, AIR, ANG, APHI, ARGDI, ARGDR, ASC, ASCLE, ASUMDI,
      ASUMDR, BSUMDI, BSUMDR,CAR, CKI, CKR, CONEI, CONER, CRSC, CR1I, CR1R,
      CR2I, CR2R, CSCL, CSGNI, CSI, CSPNI, CSPNR, CSR, C1I, C1R, C2I, C2M,
      C2R, DAII, DAIR, FMR, FN, FNF, HPI, PHIDI, PHIDR, PTI, PTR, RAST,
      RAZR, RS1, RZI, RZR, SAR, SGN, STI, STR, S1I, S1R, S2I, S2R, YY,
      ZBI, ZBR, ZEROI, ZEROR, ZET1DI, ZET1DR, ZET2DI, ZET2DR, ZNI, ZNR,
      ZRI, ZRR;
      int I, IB, IFLAG, IFN, IL, IN0, INU, IUF, K, KDFLG, KFLAG, KK, NAI,
      NDAI, NW, IDUM, J, IPARD, IC;
      REAL *BRY, *ASUMR, *ASUMI, *BSUMR, *BSUMI, *PHIR, *PHII, *ARGR, *ARGI, *ZETA1R,
      *ZETA1I, *ZETA2R, *ZETA2I, *CYR, *CYI, *CIPR, *CIPI, *CSSR, *CSRR;

	  void *vmblock = NULL;

//*** First executable statement ZUNK2

      //initialize pointers to vectors
      vmblock = vminit();  
      BRY   = (REAL *) vmalloc(vmblock, VEKTOR,  4, 0); //index 0 not used
      ASUMR = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0);
      ASUMI = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0);
      BSUMR = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0);
      BSUMI = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0);
      PHIR  = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0);
      PHII  = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0);
      ZETA1R = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0);
      ZETA1I = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0);
      ZETA2R = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0);
      ZETA2I = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0);
      CYR  = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0);
      CYI  = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0);
      CIPR = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0);
      CIPI = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0);
      ARGR = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0);
      ARGI = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0);
      CSSR = (REAL *) vmalloc(vmblock, VEKTOR,  4, 0);
      CSRR = (REAL *) vmalloc(vmblock, VEKTOR,  4, 0);

      if (! vmcomplete(vmblock)) {
        LogError ("No Memory", 0, __FILE__, __LINE__);
        return;
	  } 	

      printf(" ZUNK2:allocations Ok.\n"); getchar();

      ZEROR=0.0; ZEROI=0.0; CONER=1.0; CONEI=0.0;
      CR1R= 1.0;  CR1I= 1.73205080756887729;
      CR2R=-0.5; CR2I=-8.66025403784438647E-01;
      HPI=1.57079632679489662; 
      AIC=1.26551212348464539;
 
      CIPR[1]= 1.0; CIPI[1]=0.0; CIPR[2]=0.0; CIPI[2]=-1.0;
      CIPR[3]=-1.0; CIPI[3]=0.0; CIPR[4]=0.0; CIPI[4]= 1.0;

      KDFLG = 1;
      *NZ = 0;
/*----------------------------------------------------------------------
!     EXP(-ALIM)=EXP(-ELIM)/TOL=APPROX. ONE PRECISION GREATER THAN
!     THE UNDERFLOW LIMIT
!---------------------------------------------------------------------*/
      CSCL = 1.0/TOL;
      CRSC = TOL;
      CSSR[1] = CSCL;
      CSSR[2] = CONER;
      CSSR[3] = CRSC;
      CSRR[1] = CRSC;
      CSRR[2] = CONER;
      CSRR[3] = CSCL;
      BRY[1] = 1000.0*D1MACH(1)/TOL;
      BRY[2] = 1.0/BRY[1];
      BRY[3] = D1MACH(2);
      ZRR = ZR;
      ZRI = ZI;
      if (ZR >= 0.0) goto e10;
      ZRR = -ZR;
      ZRI = -ZI;
e10:  YY = ZRI;
      ZNR = ZRI;
      ZNI = -ZRR;
      ZBR = ZRR;
      ZBI = ZRI;
      INU = (int) floor(FNU);
      FNF = FNU - 1.0*INU;
      ANG = -HPI*FNF;
      CAR = COS(ANG);
      SAR = SIN(ANG);
      C2R = HPI*SAR;
      C2I = -HPI*CAR;
      KK = (INU % 4) + 1;
      STR = C2R*CIPR[KK] - C2I*CIPI[KK];
      STI = C2R*CIPI[KK] + C2I*CIPR[KK];
      CSR = CR1R*STR - CR1I*STI;
      CSI = CR1R*STI + CR1I*STR;
      if (YY > 0.0) goto e20;
      ZNR = -ZNR;
      ZBI = -ZBI;
/*----------------------------------------------------------------------
!     K(FNU,Z) IS COMPUTED FROM H(2,FNU,-I*Z) WHERE Z IS IN THE FIRST
!     QUADRANT. FOURTH QUADRANT VALUES (YY.LE.0.0E0) ARE COMPUTED BY
!     CONJUGATION SINCE THE K FUNCTION IS REAL ON THE POSITIVE REAL AXIS
!---------------------------------------------------------------------*/
e20:  J = 2;
      for (I=1; I<=N; I++) {
/*----------------------------------------------------------------------
!     J FLIP FLOPS BETWEEN 1 AND 2 IN J = 3 - J
!---------------------------------------------------------------------*/
        J = 3 - J;
        FN = FNU + 1.0*(I-1);
        ZUNHJ(ZNR, ZNI, FN, 0, TOL, &PHIR[J], &PHII[J], &ARGR[J], &ARGI[J],
              &ZETA1R[J], &ZETA1I[J], &ZETA2R[J], &ZETA2I[J], &ASUMR[J],
              &ASUMI[J], &BSUMR[J], &BSUMI[J]);
        if (KODE == 1) goto e30;
        STR = ZBR + ZETA2R[J];
        STI = ZBI + ZETA2I[J];
        RAST = FN/ZABS(STR,STI);
        STR = STR*RAST*RAST;
        STI = -STI*RAST*RAST;
        S1R = ZETA1R[J] - STR;
        S1I = ZETA1I[J] - STI;
        goto e40;
e30:    S1R = ZETA1R[J] - ZETA2R[J];
        S1I = ZETA1I[J] - ZETA2I[J];
/*----------------------------------------------------------------------
!     TEST FOR UNDERFLOW AND OVERFLOW
!---------------------------------------------------------------------*/
e40:    RS1 = S1R;
        if (ABS(RS1) > ELIM) goto e70;
        if (KDFLG == 1)  KFLAG = 2;
        if (ABS(RS1) < ALIM) goto e50;
/*----------------------------------------------------------------------
!     REFINE  TEST AND SCALE
!---------------------------------------------------------------------*/
        APHI = ZABS(PHIR[J],PHII[J]);
        AARG = ZABS(ARGR[J],ARGI[J]);
        RS1 = RS1 + log(APHI) - 0.25*log(AARG) - AIC;
        if (ABS(RS1) > ELIM) goto e70;
        if (KDFLG == 1)  KFLAG = 1;
        if (RS1 < 0.0) goto e50;
        if (KDFLG == 1)  KFLAG = 3;
/*----------------------------------------------------------------------
!     SCALE S1 TO KEEP INTERMEDIATE ARITHMETIC ON SCALE NEAR
!     EXPONENT EXTREMES
!---------------------------------------------------------------------*/
e50:    C2R = ARGR[J]*CR2R - ARGI[J]*CR2I;
        C2I = ARGR[J]*CR2I + ARGI[J]*CR2R;
        ZAIRY(C2R, C2I, 0, 2, &AIR, &AII, &NAI, &IDUM);
        ZAIRY(C2R, C2I, 1, 2, &DAIR, &DAII, &NDAI, &IDUM);
        STR = DAIR*BSUMR[J] - DAII*BSUMI[J];
        STI = DAIR*BSUMI[J] + DAII*BSUMR[J];
        PTR = STR*CR2R - STI*CR2I;
        PTI = STR*CR2I + STI*CR2R;
        STR = PTR + (AIR*ASUMR[J]-AII*ASUMI[J]);
        STI = PTI + (AIR*ASUMI[J]+AII*ASUMR[J]);
        PTR = STR*PHIR[J] - STI*PHII[J];
        PTI = STR*PHII[J] + STI*PHIR[J];
        S2R = PTR*CSR - PTI*CSI;
        S2I = PTR*CSI + PTI*CSR;
        STR = EXP(S1R)*CSSR[KFLAG];
        S1R = STR*COS(S1I);
        S1I = STR*SIN(S1I);
        STR = S2R*S1R - S2I*S1I;
        S2I = S1R*S2I + S2R*S1I;
        S2R = STR;
        if (KFLAG != 1) goto e60;
        ZUCHK(S2R, S2I, &NW, BRY[1], TOL);
        if (NW != 0) goto e70;
e60:    if (YY <= 0.0)  S2I = -S2I;
        CYR[KDFLG] = S2R;
        CYI[KDFLG] = S2I;
        YR[I] = S2R*CSRR[KFLAG];
        YI[I] = S2I*CSRR[KFLAG];
        STR = CSI;
        CSI = -CSR;
        CSR = STR;
        if (KDFLG == 2) goto e85;
        KDFLG = 2;
        goto e80;
e70:    if (RS1 > 0.0) goto e320;
/*----------------------------------------------------------------------
!     FOR ZR.LT.0.0, THE I FUNCTION TO BE ADDED WILL OVERFLOW
!---------------------------------------------------------------------*/
        if (ZR < 0.0) goto e320;
        KDFLG = 1;
        YR[I]=ZEROR;
        YI[I]=ZEROI;
        *NZ=*NZ+1;
        STR = CSI;
        CSI =-CSR;
        CSR = STR;
        if (I == 1) goto e80;
        if (YR[I-1] == ZEROR && YI[I-1] == ZEROI) goto e80;
        YR[I-1]=ZEROR;
        YI[I-1]=ZEROI;
        *NZ = *NZ + 1;
e80: ;} // I loop
      I = N;
e85:  RAZR = 1.0/ZABS(ZRR,ZRI);
      STR = ZRR*RAZR;
      STI = -ZRI*RAZR;
      RZR = (STR+STR)*RAZR;
      RZI = (STI+STI)*RAZR;
      CKR = FN*RZR;
      CKI = FN*RZI;
      IB = I + 1;
      if (N < IB) goto e180;
/*----------------------------------------------------------------------
!     TEST LAST MEMBER FOR UNDERFLOW AND OVERFLOW. SET SEQUENCE TO ZERO
!     ON UNDERFLOW.
!---------------------------------------------------------------------*/
      FN = FNU + 1.0*(N-1);
      IPARD = 1;
      if (MR != 0)  IPARD = 0;
      ZUNHJ(ZNR, ZNI, FN, IPARD, TOL, &PHIDR, &PHIDI, &ARGDR, &ARGDI,
            &ZET1DR, &ZET1DI, &ZET2DR, &ZET2DI, &ASUMDR, &ASUMDI, &BSUMDR, &BSUMDI);
      if (KODE == 1) goto e90;
      STR = ZBR + ZET2DR;
      STI = ZBI + ZET2DI;
      RAST = FN/ZABS(STR,STI);
      STR = STR*RAST*RAST;
      STI = -STI*RAST*RAST;
      S1R = ZET1DR - STR;
      S1I = ZET1DI - STI;
      goto e100;
e90:  S1R = ZET1DR - ZET2DR;
      S1I = ZET1DI - ZET2DI;
e100: RS1 = S1R;
      if (ABS(RS1) > ELIM) goto e105;
      if (ABS(RS1) < ALIM) goto e120;
/*----------------------------------------------------------------------
!     REFINE ESTIMATE AND TEST
!---------------------------------------------------------------------*/
      APHI = ZABS(PHIDR,PHIDI);
      RS1 = RS1+log(APHI);
      if (ABS(RS1) < ELIM) goto e120;
e105: if (RS1 > 0.0) goto e320;
/*----------------------------------------------------------------------
!     FOR ZR.LT.0.0, THE I FUNCTION TO BE ADDED WILL OVERFLOW
!---------------------------------------------------------------------*/
      if (ZR < 0.0) goto e320;
      *NZ = N;
      for (I=1; I<=N; I++) {
        YR[I] = ZEROR;
        YI[I] = ZEROI;
      }
      vmfree(vmblock);
      return;
e120: S1R = CYR[1];
      S1I = CYI[1];
      S2R = CYR[2];
      S2I = CYI[2];
      C1R = CSRR[KFLAG];
      ASCLE = BRY[KFLAG];
      for (I=IB; I<=N; I++) {
        C2R = S2R;
        C2I = S2I;
        S2R = CKR*C2R - CKI*C2I + S1R;
        S2I = CKR*C2I + CKI*C2R + S1I;
        S1R = C2R;
        S1I = C2I;
        CKR = CKR + RZR;
        CKI = CKI + RZI;
        C2R = S2R*C1R;
        C2I = S2I*C1R;
        YR[I] = C2R;
        YI[I] = C2I;
        if (KFLAG >= 3) goto e130;
        STR = ABS(C2R);
        STI = ABS(C2I);
        C2M = DMAX(STR,STI);
        if (C2M <= ASCLE) goto e130;
        KFLAG = KFLAG + 1;
        ASCLE = BRY[KFLAG];
        S1R = S1R*C1R;
        S1I = S1I*C1R;
        S2R = C2R;
        S2I = C2I;
        S1R = S1R*CSSR[KFLAG];
        S1I = S1I*CSSR[KFLAG];
        S2R = S2R*CSSR[KFLAG];
        S2I = S2I*CSSR[KFLAG];
        C1R = CSRR[KFLAG];
e130:;} // I loop
e180: if (MR == 0) {
        vmfree(vmblock);
	    return;
      }
/*----------------------------------------------------------------------
!     ANALYTIC CONTINUATION FOR RE(Z).LT.0.0D0
!---------------------------------------------------------------------*/
      *NZ = 0;
      FMR = 1.0*MR;
      SGN = -SIGN(PI,FMR);
/*----------------------------------------------------------------------
!     CSPN AND CSGN ARE COEFF OF K AND I FUNCIONS RESP.
!---------------------------------------------------------------------*/
      CSGNI = SGN;
      if (YY <= 0.0)  CSGNI = -CSGNI;
      IFN = INU + N - 1;
      ANG = FNF*SGN;
      CSPNR = COS(ANG);
      CSPNI = SIN(ANG);
      if ((IFN % 2) == 0) goto e190;
      CSPNR = -CSPNR;
      CSPNI = -CSPNI;
/*----------------------------------------------------------------------
!     CS=COEFF OF THE J FUNCTION TO GET THE I FUNCTION. I(FNU,Z) IS
!     COMPUTED FROM EXP(I*FNU*HPI)*J(FNU,-I*Z) WHERE Z IS IN THE FIRST
!     QUADRANT. FOURTH QUADRANT VALUES (YY.LE.0.0E0) ARE COMPUTED BY
!     CONJUGATION SINCE THE I FUNCTION IS REAL ON THE POSITIVE REAL AXIS
!---------------------------------------------------------------------*/
e190: CSR = SAR*CSGNI;
      CSI = CAR*CSGNI;
      IN0 = (IFN % 4) + 1;
      C2R = CIPR[IN0];
      C2I = CIPI[IN0];
      STR = CSR*C2R + CSI*C2I;
      CSI = -CSR*C2I + CSI*C2R;
      CSR = STR;
      ASC = BRY[1];
      IUF = 0;
      KK = N;
      KDFLG = 1;
      IB = IB - 1;
      IC = IB - 1;
      for (K=1; K<=N; K++) {
        FN = FNU + 1.0*(KK-1);
/*----------------------------------------------------------------------
!     LOGIC TO SORT OUT CASES WHOSE PARAMETERS WERE SET FOR THE K
!     FUNCTION ABOVE
!---------------------------------------------------------------------*/
        if (N > 2) goto e175;
e172:   PHIDR = PHIR[J];
        PHIDI = PHII[J];
        ARGDR = ARGR[J];
        ARGDI = ARGI[J];
        ZET1DR = ZETA1R[J];
        ZET1DI = ZETA1I[J];
        ZET2DR = ZETA2R[J];
        ZET2DI = ZETA2I[J];
        ASUMDR = ASUMR[J];
        ASUMDI = ASUMI[J];
        BSUMDR = BSUMR[J];
        BSUMDI = BSUMI[J];
        J = 3 - J;
        goto e210;
e175:   if (KK == N && IB < N) goto e210;
        if (KK == IB || KK == IC) goto e172;
        ZUNHJ(ZNR, ZNI, FN, 0, TOL, &PHIDR, &PHIDI, &ARGDR,
              &ARGDI, &ZET1DR, &ZET1DI, &ZET2DR, &ZET2DI, &ASUMDR,
              &ASUMDI, &BSUMDR, &BSUMDI);
e210:   if (KODE == 1) goto e220;
        STR = ZBR + ZET2DR;
        STI = ZBI + ZET2DI;
        RAST = FN/ZABS(STR,STI);
        STR = STR*RAST*RAST;
        STI = -STI*RAST*RAST;
        S1R = -ZET1DR + STR;
        S1I = -ZET1DI + STI;
        goto e230;
e220:   S1R = -ZET1DR + ZET2DR;
        S1I = -ZET1DI + ZET2DI;
/*----------------------------------------------------------------------
!     TEST FOR UNDERFLOW AND OVERFLOW
!---------------------------------------------------------------------*/
e230:   RS1 = S1R;
        if (ABS(RS1) > ELIM) goto e280;
        if (KDFLG == 1)  IFLAG = 2;
        if (ABS(RS1) < ALIM) goto e240;
/*----------------------------------------------------------------------
!     REFINE  TEST AND SCALE
!---------------------------------------------------------------------*/
        APHI = ZABS(PHIDR,PHIDI);
        AARG = ZABS(ARGDR,ARGDI);
        RS1 = RS1 + log(APHI) - 0.25*log(AARG) - AIC;
        if (ABS(RS1) > ELIM) goto e280;
        if (KDFLG == 1)  IFLAG = 1;
        if (RS1 < 0.0) goto e240;
        if (KDFLG == 1)  IFLAG = 3;
e240:   ZAIRY(ARGDR, ARGDI, 0, 2, &AIR, &AII, &NAI, &IDUM);
        ZAIRY(ARGDR, ARGDI, 1, 2, &DAIR, &DAII, &NDAI, &IDUM);
        STR = DAIR*BSUMDR - DAII*BSUMDI;
        STI = DAIR*BSUMDI + DAII*BSUMDR;
        STR = STR + (AIR*ASUMDR-AII*ASUMDI);
        STI = STI + (AIR*ASUMDI+AII*ASUMDR);
        PTR = STR*PHIDR - STI*PHIDI;
        PTI = STR*PHIDI + STI*PHIDR;
        S2R = PTR*CSR - PTI*CSI;
        S2I = PTR*CSI + PTI*CSR;
        STR = EXP(S1R)*CSSR[IFLAG];
        S1R = STR*COS(S1I);
        S1I = STR*SIN(S1I);
        STR = S2R*S1R - S2I*S1I;
        S2I = S2R*S1I + S2I*S1R;
        S2R = STR;
        if (IFLAG != 1) goto e250;
        ZUCHK(S2R, S2I, &NW, BRY[1], TOL);
        if (NW == 0) goto e250;
        S2R = ZEROR;
        S2I = ZEROI;
e250:   if (YY <= 0.0)  S2I = -S2I;
        CYR[KDFLG] = S2R;
        CYI[KDFLG] = S2I;
        C2R = S2R;
        C2I = S2I;
        S2R = S2R*CSRR[IFLAG];
        S2I = S2I*CSRR[IFLAG];
/*----------------------------------------------------------------------
!     ADD I AND K FUNCTIONS, K SEQUENCE IN Y(I), I=1,N
!---------------------------------------------------------------------*/
        S1R = YR[KK];
        S1I = YI[KK];
        if (KODE == 1) goto e270;
        ZS1S2(&ZRR, &ZRI, &S1R, &S1I, &S2R, &S2I, &NW, ASC, ALIM, &IUF);
        *NZ = *NZ + NW;
e270:   YR[KK] = S1R*CSPNR - S1I*CSPNI + S2R;
        YI[KK] = S1R*CSPNI + S1I*CSPNR + S2I;
        KK = KK - 1;
        CSPNR = -CSPNR;
        CSPNI = -CSPNI;
        STR = CSI;
        CSI = -CSR;
        CSR = STR;
        if (C2R != 0.0 || C2I != 0.0) goto e255;
        KDFLG = 1;
        goto e290;
e255:   if (KDFLG == 2) goto e295;
        KDFLG = 2;
        goto e290;
e280:   if (RS1 > 0.0) goto e320;
        S2R = ZEROR;
        S2I = ZEROI;
        goto e250;
e290:;} // K loop 
      K = N;
e295: IL = N - K;
      if (IL == 0) { 
		vmfree(vmblock);  
	    return;
      }
/*----------------------------------------------------------------------
!     RECUR BACKWARD FOR REMAINDER OF I SEQUENCE AND ADD IN THE
!     K FUNCTIONS, SCALING THE I SEQUENCE DURING RECURRENCE TO KEEP
!     INTERMEDIATE ARITHMETIC ON SCALE NEAR EXPONENT EXTREMES.
!---------------------------------------------------------------------*/
      S1R = CYR[1];
      S1I = CYI[1];
      S2R = CYR[2];
      S2I = CYI[2];
      CSR = CSRR[IFLAG];
      ASCLE = BRY[IFLAG];
      FN = 1.0*(INU+IL);
      for (I=1; I<=IL; I++) {
        C2R = S2R;
        C2I = S2I;
        S2R = S1R + (FN+FNF)*(RZR*C2R-RZI*C2I);
        S2I = S1I + (FN+FNF)*(RZR*C2I+RZI*C2R);
        S1R = C2R;
        S1I = C2I;
        FN = FN - 1.0;
        C2R = S2R*CSR;
        C2I = S2I*CSR;
        CKR = C2R;
        CKI = C2I;
        C1R = YR[KK];
        C1I = YI[KK];
        if (KODE == 1) goto e300;
        ZS1S2(&ZRR, &ZRI, &C1R, &C1I, &C2R, &C2I, &NW, ASC, ALIM, &IUF);
        *NZ = *NZ + NW;
e300:   YR[KK] = C1R*CSPNR - C1I*CSPNI + C2R;
        YI[KK] = C1R*CSPNI + C1I*CSPNR + C2I;
        KK--;
        CSPNR = -CSPNR;
        CSPNI = -CSPNI;
        if (IFLAG >= 3) goto e310;
        C2R = ABS(CKR);
        C2I = ABS(CKI);
        C2M = DMAX(C2R,C2I);
        if (C2M <= ASCLE) goto e310;
        IFLAG++;
        ASCLE = BRY[IFLAG];
        S1R = S1R*CSR;
        S1I = S1I*CSR;
        S2R = CKR;
        S2I = CKI;
        S1R = S1R*CSSR[IFLAG];
        S1I = S1I*CSSR[IFLAG];
        S2R = S2R*CSSR[IFLAG];
        S2I = S2I*CSSR[IFLAG];
        CSR = CSRR[IFLAG];
e310:;} // I loop
	  vmfree(vmblock);
      return;
e320: *NZ = -1;
} //ZUNK2()


void ZACON(REAL ZR, REAL ZI, REAL FNU, int KODE, int MR, int N, REAL *YR, REAL *YI, int *NZ,
           REAL RL, REAL FNUL, REAL TOL, REAL ELIM, REAL ALIM)  {
/***BEGIN PROLOGUE  ZACON
!***REFER TO  ZBESK,ZBESH
!
!     ZACON APPLIES THE ANALYTIC CONTINUATION FORMULA
!
!         K(FNU,ZN*EXP(MP))=K(FNU,ZN)*EXP(-MP*FNU) - MP*I(FNU,ZN)
!                 MP=PI*MR*CMPLX(0.0,1.0)
!
!     TO CONTINUE THE K FUNCTION FROM THE RIGHT HALF TO THE LEFT
!     HALF Z PLANE
!
!***ROUTINES CALLED  ZBINU,ZBKNU,ZS1S2,D1MACH,ZABS,ZMLT
!***END PROLOGUE  ZACON
!     COMPLEX CK,CONE,CSCL,CSCR,CSGN,CSPN,CY,CZERO,C1,C2,RZ,SC1,SC2,ST,
!     S1,S2,Y,Z,ZN */
//Labels: e10,e20,e30,e40,e50,e60,e70,e80,e90

      REAL ARG, ASCLE, AS2, AZN, BSCLE, CKI, CKR, CONEI, CONER, CPN, CSCL, CSCR,
      CSGNI, CSGNR, CSPNI, CSPNR, CSR, C1I, C1M, C1R, C2I, C2R, FMR, FN,
      PTI, PTR, RAZN, RZI, RZR, SC1I, SC1R, SC2I, SC2R, SGN, SPN, STI, STR,
      S1I, S1R, S2I, S2R, YY, ZEROI, ZEROR, ZNI, ZNR;
      int I, INU, IUF, KFLAG, NN, NW;
      REAL *CYR, *CYI, *CSSR, *CSRR, *BRY;
      void *vmblock = NULL;

//*** First executable statement ZACON

      //initialize pointers to vectors
      vmblock = vminit();  
      BRY   = (REAL *) vmalloc(vmblock, VEKTOR,  4, 0); //index 0 not used
      CYR  = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0);
      CYI  = (REAL *) vmalloc(vmblock, VEKTOR,  N+1, 0);
      CSSR = (REAL *) vmalloc(vmblock, VEKTOR,  4, 0);
      CSRR = (REAL *) vmalloc(vmblock, VEKTOR,  4, 0);

      if (! vmcomplete(vmblock)) {
        LogError ("No Memory", 0, __FILE__, __LINE__);
        return;
	  } 	

      ZEROR=0.0; ZEROI=0.0; CONER=1.0; CONEI=0.0;
      NZ = 0;
      ZNR = -ZR;
      ZNI = -ZI;
      NN = N;
      ZBINU(ZNR, ZNI, FNU, KODE, NN, YR, YI, &NW, RL, FNUL, TOL, ELIM, ALIM);
      if (NW < 0) goto e90;
/*----------------------------------------------------------------------
!     ANALYTIC CONTINUATION TO THE LEFT HALF PLANE FOR THE K FUNCTION
!---------------------------------------------------------------------*/
      NN = IMIN(2,N);
      ZBKNU(ZNR, ZNI, FNU, KODE, NN, CYR, CYI, &NW, TOL, ELIM, ALIM);
      if (NW != 0) goto e90;
      S1R = CYR[1];
      S1I = CYI[1];
      FMR = 1.0*MR;
      SGN = -SIGN(PI,FMR);
      CSGNR = ZEROR;
      CSGNI = SGN;
      if (KODE == 1) goto e10;
      YY = -ZNI;
      CPN = COS(YY);
      SPN = SIN(YY);
      ZMLT(CSGNR, CSGNI, CPN, SPN, &CSGNR, &CSGNI);
/*----------------------------------------------------------------------
!     CALCULATE CSPN=EXP(FNU*PI*I) TO MINIMIZE LOSSES OF SIGNIFICANCE
!     WHEN FNU IS LARGE
!---------------------------------------------------------------------*/
e10:  INU = (int) floor(FNU);
      ARG = (FNU-1.0*INU)*SGN;
      CPN = COS(ARG);
      SPN = SIN(ARG);
      CSPNR = CPN;
      CSPNI = SPN;
      if ((INU % 2) == 0) goto e20;
      CSPNR = -CSPNR;
      CSPNI = -CSPNI;
e20:  IUF = 0;
      C1R = S1R;
      C1I = S1I;
      C2R = YR[1];
      C2I = YI[1];
      ASCLE = 1000.0*D1MACH(1)/TOL;
      if (KODE == 1) goto e30;
      ZS1S2(&ZNR, &ZNI, &C1R, &C1I, &C2R, &C2I, &NW, ASCLE, ALIM, &IUF);
      *NZ = *NZ + NW;
      SC1R = C1R;
      SC1I = C1I;
e30:  ZMLT(CSPNR, CSPNI, C1R, C1I, &STR, &STI);
      ZMLT(CSGNR, CSGNI, C2R, C2I, &PTR, &PTI);
      YR[1] = STR + PTR;
      YI[1] = STI + PTI;
      if (N == 1) {
		vmfree(vmblock);  
	    return;
      }
      CSPNR = -CSPNR;
      CSPNI = -CSPNI;
      S2R = CYR[2];
      S2I = CYI[2];
      C1R = S2R;
      C1I = S2I;
      C2R = YR[2];
      C2I = YI[2];
      if (KODE == 1) goto e40;
      ZS1S2(&ZNR, &ZNI, &C1R, &C1I, &C2R, &C2I, &NW, ASCLE, ALIM, &IUF);
      *NZ = *NZ + NW;
      SC2R = C1R;
      SC2I = C1I;
e40:  ZMLT(CSPNR, CSPNI, C1R, C1I, &STR, &STI);
      ZMLT(CSGNR, CSGNI, C2R, C2I, &PTR, &PTI);
      YR[2] = STR + PTR;
      YI[2] = STI + PTI;
      if (N == 2) {
	    vmfree(vmblock);	  
	    return;
      }
      CSPNR = -CSPNR;
      CSPNI = -CSPNI;
      AZN = ZABS(ZNR,ZNI);
      RAZN = 1.0/AZN;
      STR = ZNR*RAZN;
      STI = -ZNI*RAZN;
      RZR = (STR+STR)*RAZN;
      RZI = (STI+STI)*RAZN;
      FN = FNU + 1.0;
      CKR = FN*RZR;
      CKI = FN*RZI;
/*---------------------------------------------------------------------
!     SCALE NEAR EXPONENT EXTREMES DURING RECURRENCE ON K FUNCTIONS
!---------------------------------------------------------------------*/
      CSCL = 1.0/TOL;
      CSCR = TOL;
      CSSR[1] = CSCL;
      CSSR[2] = CONER;
      CSSR[3] = CSCR;
      CSRR[1] = CSCR;
      CSRR[2] = CONER;
      CSRR[3] = CSCL;
      BRY[1] = ASCLE;
      BRY[2] = 1.0/ASCLE;
      BRY[3] = D1MACH(2);
      AS2 = ZABS(S2R,S2I);
      KFLAG = 2;
      if (AS2 > BRY[1]) goto e50;
      KFLAG = 1;
      goto e60;
e50:  if (AS2 < BRY[2]) goto e60;
      KFLAG = 3;
e60:  BSCLE = BRY[KFLAG];
      S1R = S1R*CSSR[KFLAG];
      S1I = S1I*CSSR[KFLAG];
      S2R = S2R*CSSR[KFLAG];
      S2I = S2I*CSSR[KFLAG];
      CSR = CSRR[KFLAG];
      for (I=3; I<=N; I++) {
        STR = S2R;
        STI = S2I;
        S2R = CKR*STR - CKI*STI + S1R;
        S2I = CKR*STI + CKI*STR + S1I;
        S1R = STR;
        S1I = STI;
        C1R = S2R*CSR;
        C1I = S2I*CSR;
        STR = C1R;
        STI = C1I;
        C2R = YR[I];
        C2I = YI[I];
        if (KODE == 1) goto e70;
        if (IUF < 0) goto e70;
        ZS1S2(&ZNR, &ZNI, &C1R, &C1I, &C2R, &C2I, &NW, ASCLE, ALIM, &IUF);
        *NZ = *NZ + NW;
        SC1R = SC2R;
        SC1I = SC2I;
        SC2R = C1R;
        SC2I = C1I;
        if (IUF != 3) goto e70;
        IUF = -4;
        S1R = SC1R*CSSR[KFLAG];
        S1I = SC1I*CSSR[KFLAG];
        S2R = SC2R*CSSR[KFLAG];
        S2I = SC2I*CSSR[KFLAG];
        STR = SC2R;
        STI = SC2I;
e70:    PTR = CSPNR*C1R - CSPNI*C1I;
        PTI = CSPNR*C1I + CSPNI*C1R;
        YR[I] = PTR + CSGNR*C2R - CSGNI*C2I;
        YI[I] = PTI + CSGNR*C2I + CSGNI*C2R;
        CKR = CKR + RZR;
        CKI = CKI + RZI;
        CSPNR = -CSPNR;
        CSPNI = -CSPNI;
        if (KFLAG >= 3) goto e80;
        PTR = ABS(C1R);
        PTI = ABS(C1I);
        C1M = DMAX(PTR,PTI);
        if (C1M <= BSCLE) goto e80;
        KFLAG++;
        BSCLE = BRY[KFLAG];
        S1R = S1R*CSR;
        S1I = S1I*CSR;
        S2R = STR;
        S2I = STI;
        S1R = S1R*CSSR[KFLAG];
        S1I = S1I*CSSR[KFLAG];
        S2R = S2R*CSSR[KFLAG];
        S2I = S2I*CSSR[KFLAG];
        CSR = CSRR[KFLAG];
e80: ;} // I loop
	  vmfree(vmblock);
      return;
e90:  *NZ = -1;
      if (NW = -2) *NZ=-2;
	  vmfree(vmblock);
} //ZACON()

//end of file Cbess3.cpp
