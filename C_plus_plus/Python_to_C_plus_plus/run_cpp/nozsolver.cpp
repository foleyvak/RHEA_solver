// Self-contained 2D convergent-divergent nozzle solver.
// Assembled from the curvilinear components validated bit-for-bit against the Python
// reference (grid + cell/face metrics + rotated-frame HLLC inviscid + curvilinear viscous
// + West Aitken inflow / East NSCBC outflow / South symmetry / North wall) plus ideal-gas
// thermodynamics and SSP-RK3 time integration. Ideal gas => no CoolProp/Boost needed.
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <hdf5.h>
using namespace std;

// ---------------- parameters ----------------
double r_t=0.8e-3,r_c=2.0e-3,R1_rt=10.0,R2_R1=3.0,Rexp_rt=30.0,theta=10.0,alpha=3.0,L_N=50.0e-3,L_c=3.0e-3;
double R1,R2,Rexp,th,al,x_c,r1,r2,x1,x2,x_t,x_exp,r_exp,L_x;
double x_0=0,y_0=0,z_0=0,L_y=1.0e-2,L_z=1.0e-4,A_x=0,A_y=-1.0,A_z=0;
const double Rg=296.8, gam=1.4, cv=296.8/0.4, cp=1.4*296.8/0.4;
const double mu_c=2.0e-5, kappa_c=2.0e-2;
const double CFL=0.1, eps=1e-10;
const double U_inlet=100.0, P_inlet=150.0e5, T_inlet=600.0, P_outlet=10.0e5;
const int gNx=32,gNy=32,gNz=1,NX=gNx+2,NY=gNy+2,NZ=gNz+2;

// ---------------- ideal gas ----------------
inline double sos_of(double P,double rho){return sqrt(gam*P/rho);}
// ---------------- fields ----------------
static double xf[NX][NY][NZ],yf[NX][NY][NZ],zf[NX][NY][NZ];
static double detJ[NX][NY][NZ],XI[NX][NY][NZ][3],ET[NX][NY][NZ][3],ZE[NX][NY][NZ][3];
static double FM[NX][NY][NZ][6][3][3],FJ[NX][NY][NZ][6];
static double rho[NX][NY][NZ],uu[NX][NY][NZ],vv[NX][NY][NZ],ww[NX][NY][NZ],PP[NX][NY][NZ],TT[NX][NY][NZ],EE[NX][NY][NZ],ss[NX][NY][NZ];
static double ru[NX][NY][NZ],rv[NX][NY][NZ],rw[NX][NY][NZ],rE[NX][NY][NZ]; // conserved (rho stored in rho[])
static double r0[NX][NY][NZ],ru0[NX][NY][NZ],rv0[NX][NY][NZ],rw0[NX][NY][NZ],rE0[NX][NY][NZ];
static double Ir[NX][NY][NZ],Iu[NX][NY][NZ],Iv[NX][NY][NZ],Iw[NX][NY][NZ],IE[NX][NY][NZ]; // inviscid flux div
static double Vu[NX][NY][NZ],Vv[NX][NY][NZ],Vw[NX][NY][NZ],VE[NX][NY][NZ];               // viscous flux div

double gfn(double e,double A){return e+A*(0.5-e)*(1.0-e)*e;}
double nozzleRadius(double x){double r;
  if(x<=x_c)r=r_c; else if(x<=x2)r=r_c-R2*(1.0-sqrt(1.0-pow((x-x_c)/R2,2.0)));
  else if(x<=x1)r=r1-(x-x1)*tan(th); else if(x<=x_t)r=r_t+R1*(1.0-sqrt(1.0-pow((x-x_t)/R1,2.0)));
  else if(x<=x_exp)r=r_t+Rexp*(1.0-sqrt(1.0-pow((x-x_t)/Rexp,2.0))); else r=r_exp+(x-x_exp)*tan(al); return r;}
double compC(int c,int i,int j,int k){if(c==0)return xf[i][j][k];if(c==1)return yf[i][j][k]/nozzleRadius(xf[i][j][k]);return zf[i][j][k];}
double cC(double F[NX][NY][NZ],int i,int j,int k,int d){int ip=i,jp=j,kp=k,im=i,jm=j,km=k;if(d==0){ip=i+1;im=i-1;}else if(d==1){jp=j+1;jm=j-1;}else{kp=k+1;km=k-1;}return (F[ip][jp][kp]-F[im][jm][km])/(compC(d,ip,jp,kp)-compC(d,im,jm,km));}
double fD(double F[NX][NY][NZ],int i,int j,int k,int nd,int s){int ni=i,nj=j,nk=k;if(nd==0)ni=i+s;else if(nd==1)nj=j+s;else nk=k+s;int hi,hj,hk,li,lj,lk;if(s>0){hi=ni;hj=nj;hk=nk;li=i;lj=j;lk=k;}else{hi=i;hj=j;hk=k;li=ni;lj=nj;lk=nk;}return (F[hi][hj][hk]-F[li][lj][lk])/(compC(nd,hi,hj,hk)-compC(nd,li,lj,lk));}
void inv3(double xm[3],double ym[3],double zm[3],double o[3][3],double&J){double d=xm[0]*ym[1]*zm[2]+xm[1]*ym[2]*zm[0]+xm[2]*ym[0]*zm[1]-xm[2]*ym[1]*zm[0]-xm[1]*ym[0]*zm[2]-xm[0]*ym[2]*zm[1];o[0][0]=(ym[1]*zm[2]-ym[2]*zm[1])/d;o[0][1]=(xm[2]*zm[1]-xm[1]*zm[2])/d;o[0][2]=(xm[1]*ym[2]-xm[2]*ym[1])/d;o[1][0]=(ym[2]*zm[0]-ym[0]*zm[2])/d;o[1][1]=(xm[0]*zm[2]-xm[2]*zm[0])/d;o[1][2]=(xm[2]*ym[0]-xm[0]*ym[2])/d;o[2][0]=(ym[0]*zm[1]-ym[1]*zm[0])/d;o[2][1]=(xm[1]*zm[0]-xm[0]*zm[1])/d;o[2][2]=(xm[0]*ym[1]-xm[1]*ym[0])/d;J=o[0][0]*(o[1][1]*o[2][2]-o[1][2]*o[2][1])-o[0][1]*(o[1][0]*o[2][2]-o[1][2]*o[2][0])+o[0][2]*(o[1][0]*o[2][1]-o[1][1]*o[2][0]);}

void buildGridMetrics(){
  const double pi=4.0*atan(1.0);R1=r_t*R1_rt;R2=R1*R2_R1;Rexp=r_t*Rexp_rt;th=theta*pi/180;al=alpha*pi/180;
  x_c=L_c;r2=r_c-R2*(1-cos(th));r1=r_t+R1*(1-cos(th));x2=x_c+R2*sin(th);x1=x2+(r2-r1)/tan(th);x_t=x1+R1*sin(th);x_exp=x_t+Rexp*sin(al);r_exp=r_t+Rexp*(1-cos(al));L_x=x_t+L_N+L_c;
  double gx[NX],gy[NY],gz[NZ];
  for(int i=0;i<NX;i++){double e=(i-0.5)/gNx;gx[i]=x_0+L_x*gfn(e,A_x);if(i==0){e=0.5/gNx;gx[i]=x_0-L_x*gfn(e,A_x);}if(i==gNx+1){e=(gNx-0.5)/gNx;gx[i]=x_0+2*L_x-L_x*gfn(e,A_x);}}
  for(int j=0;j<NY;j++){double e=(j-0.5)/gNy;gy[j]=y_0+L_y*gfn(e,A_y);if(j==0){e=0.5/gNy;gy[j]=y_0-L_y*gfn(e,A_y);}if(j==gNy+1){e=(gNy-0.5)/gNy;gy[j]=y_0+2*L_y-L_y*gfn(e,A_y);}}
  for(int k=0;k<NZ;k++){double e=(k-0.5)/gNz;gz[k]=z_0+L_z*gfn(e,A_z);if(k==0){e=0.5/gNz;gz[k]=z_0-L_z*gfn(e,A_z);}if(k==gNz+1){e=(gNz-0.5)/gNz;gz[k]=z_0+2*L_z-L_z*gfn(e,A_z);}}
  for(int i=0;i<NX;i++)for(int j=0;j<NY;j++)for(int k=0;k<NZ;k++){double Li=nozzleRadius(gx[i]);xf[i][j][k]=gx[i];yf[i][j][k]=y_0+(Li/L_y)*(gy[j]-y_0);zf[i][j][k]=gz[k];}
  for(int i=0;i<NX;i++)for(int j=0;j<NY;j++)for(int k=0;k<NZ;k++){int ix=(i==0)?1:(i==NX-1?-1:0),jy=(j==0)?1:(j==NY-1?-1:0),kz=(k==0)?1:(k==NZ-1?-1:0);double xm[3],ym[3],zm[3];double Ll=nozzleRadius(xf[i][j][k]);
    if(ix>0){double d=xf[i+1][j][k]-xf[i][j][k];xm[0]=(xf[i+1][j][k]-xf[i][j][k])/d;ym[0]=(yf[i+1][j][k]-yf[i][j][k])/d;zm[0]=(zf[i+1][j][k]-zf[i][j][k])/d;}else if(ix<0){double d=xf[i][j][k]-xf[i-1][j][k];xm[0]=(xf[i][j][k]-xf[i-1][j][k])/d;ym[0]=(yf[i][j][k]-yf[i-1][j][k])/d;zm[0]=(zf[i][j][k]-zf[i-1][j][k])/d;}else xm[0]=cC(xf,i,j,k,0),ym[0]=cC(yf,i,j,k,0),zm[0]=cC(zf,i,j,k,0);
    if(jy>0){double d=(yf[i][j+1][k]-yf[i][j][k])/Ll;xm[1]=(xf[i][j+1][k]-xf[i][j][k])/d;ym[1]=(yf[i][j+1][k]-yf[i][j][k])/d;zm[1]=(zf[i][j+1][k]-zf[i][j][k])/d;}else if(jy<0){double d=(yf[i][j][k]-yf[i][j-1][k])/Ll;xm[1]=(xf[i][j][k]-xf[i][j-1][k])/d;ym[1]=(yf[i][j][k]-yf[i][j-1][k])/d;zm[1]=(zf[i][j][k]-zf[i][j-1][k])/d;}else xm[1]=cC(xf,i,j,k,1),ym[1]=cC(yf,i,j,k,1),zm[1]=cC(zf,i,j,k,1);
    if(kz>0){double d=zf[i][j][k+1]-zf[i][j][k];xm[2]=(xf[i][j][k+1]-xf[i][j][k])/d;ym[2]=(yf[i][j][k+1]-yf[i][j][k])/d;zm[2]=(zf[i][j][k+1]-zf[i][j][k])/d;}else if(kz<0){double d=zf[i][j][k]-zf[i][j][k-1];xm[2]=(xf[i][j][k]-xf[i][j][k-1])/d;ym[2]=(yf[i][j][k]-yf[i][j][k-1])/d;zm[2]=(zf[i][j][k]-zf[i][j][k-1])/d;}else xm[2]=cC(xf,i,j,k,2),ym[2]=cC(yf,i,j,k,2),zm[2]=cC(zf,i,j,k,2);
    double o[3][3],J;inv3(xm,ym,zm,o,J);detJ[i][j][k]=J;for(int c=0;c<3;c++){XI[i][j][k][c]=o[0][c];ET[i][j][k][c]=o[1][c];ZE[i][j][k][c]=o[2][c];}}
  for(int i=1;i<=gNx;i++)for(int j=1;j<=gNy;j++)for(int k=1;k<=gNz;k++)for(int f=0;f<6;f++){int nd=f/2,s=(f%2==0)?1:-1,ni=i,nj=j,nk=k;if(nd==0)ni=i+s;else if(nd==1)nj=j+s;else nk=k+s;double xm[3],ym[3],zm[3];
    for(int c=0;c<3;c++){if(c==nd){xm[c]=fD(xf,i,j,k,nd,s);ym[c]=fD(yf,i,j,k,nd,s);zm[c]=fD(zf,i,j,k,nd,s);}else{xm[c]=0.5*(cC(xf,i,j,k,c)+cC(xf,ni,nj,nk,c));ym[c]=0.5*(cC(yf,i,j,k,c)+cC(yf,ni,nj,nk,c));zm[c]=0.5*(cC(zf,i,j,k,c)+cC(zf,ni,nj,nk,c));}}
    double o[3][3],J;inv3(xm,ym,zm,o,J);for(int c=0;c<3;c++)for(int d=0;d<3;d++)FM[i][j][k][f][c][d]=o[c][d];FJ[i][j][k][f]=J;}
}
// ---------------- HLLC (rotated) ----------------
void wavesSpeed(double rL,double rR,double uL,double uR,double PL,double PR,double aL,double aR,double&SL,double&SR){double sL=sqrt(rL),sR=sqrt(rR);double hu=(uL*sL+uR*sR)/(sL+sR);double ha=sqrt(((aL*aL*sL+aR*aR*sR)/(sL+sR))+0.5*((sL*sR)/((sL+sR)*(sL+sR)))*(uR-uL)*(uR-uL));SL=min(uL-aL,hu-ha);SR=max(uR+aR,hu+ha);}
double hllc(double rL,double rR,double uL,double uR,double vL,double vR,double wL,double wR,double EL,double ER,double PL,double PR,double aL,double aR,int vt){double FL=rL*uL,FR=rR*uR,UL=rL,UR=rR;if(vt==1){FL*=uL;FL+=PL;FR*=uR;FR+=PR;UL*=uL;UR*=uR;}else if(vt==2){FL*=vL;FR*=vR;UL*=vL;UR*=vR;}else if(vt==3){FL*=wL;FR*=wR;UL*=wL;UR*=wR;}else if(vt==4){FL*=EL;FL+=uL*PL;FR*=ER;FR+=uR*PR;UL*=EL;UR*=ER;}double SL,SR;wavesSpeed(rL,rR,uL,uR,PL,PR,aL,aR,SL,SR);double Ss=(PR-PL+rL*uL*(SL-uL)-rR*uR*(SR-uR))/(rL*(SL-uL)-rR*(SR-uR));double UsL=rL*((SL-uL)/(SL-Ss)),UsR=rR*((SR-uR)/(SR-Ss));if(vt==1){UsL*=Ss;UsR*=Ss;}else if(vt==2){UsL*=vL;UsR*=vR;}else if(vt==3){UsL*=wL;UsR*=wR;}else if(vt==4){UsL*=(EL+(Ss-uL)*(Ss+PL/(rL*(SL-uL))));UsR*=(ER+(Ss-uR)*(Ss+PR/(rR*(SR-uR))));}double FsL=FL+SL*(UsL-UL),FsR=FR+SR*(UsR-UR),F=0;if(0.0<=SL)F=FL;else if(SL<=0&&0<=Ss)F=FsL;else if(Ss<=0&&0<=SR)F=FsR;else if(0>=SR)F=FR;return F;}
void faceFlux(double mvx,double mvy,double mvz,int tt,double rL,double rR,double uL,double uR,double vL,double vR,double wL,double wR,double EL,double ER,double PL,double PR,double aL,double aR,double&rF,double&ruF,double&rvF,double&rwF,double&reF){
  double g=sqrt(mvx*mvx+mvy*mvy+mvz*mvz),n0=mvx/g,n1=mvy/g,n2=mvz/g,t10,t11,t12,t20,t21,t22;
  if(tt==0){t10=-n1;t11=n0;t12=0;t20=-n0*n2;t21=-n1*n2;t22=n0*n0+n1*n1;}else{t10=n2;t11=0;t12=-n0;t20=-n0*n1;t21=n0*n0+n2*n2;t22=-n1*n2;}
  double m1=sqrt(t10*t10+t11*t11+t12*t12),m2=sqrt(t20*t20+t21*t21+t22*t22);t10/=m1;t11/=m1;t12/=m1;t20/=m2;t21/=m2;t22/=m2;
  double VnL=uL*n0+vL*n1+wL*n2,VnR=uR*n0+vR*n1+wR*n2,V1L=uL*t10+vL*t11+wL*t12,V1R=uR*t10+vR*t11+wR*t12,V2L=uL*t20+vL*t21+wL*t22,V2R=uR*t20+vR*t21+wR*t22;
  double fr=hllc(rL,rR,VnL,VnR,V1L,V1R,V2L,V2R,EL,ER,PL,PR,aL,aR,0);
  double fn=hllc(rL,rR,VnL,VnR,V1L,V1R,V2L,V2R,EL,ER,PL,PR,aL,aR,1);
  double f1=hllc(rL,rR,VnL,VnR,V1L,V1R,V2L,V2R,EL,ER,PL,PR,aL,aR,2);
  double f2=hllc(rL,rR,VnL,VnR,V1L,V1R,V2L,V2R,EL,ER,PL,PR,aL,aR,3);
  double fe=hllc(rL,rR,VnL,VnR,V1L,V1R,V2L,V2R,EL,ER,PL,PR,aL,aR,4);
  rF=fr*g;ruF=(fn*n0+f1*t10+f2*t20)*g;rvF=(fn*n1+f1*t11+f2*t21)*g;rwF=(fn*n2+f1*t12+f2*t22)*g;reF=fe*g;}

void inviscid(){
  for(int i=1;i<=gNx;i++)for(int j=1;j<=gNy;j++)for(int k=1;k<=gNz;k++){
    double Ll=nozzleRadius(xf[i][j][k]);double dxi=0.5*(xf[i+1][j][k]-xf[i-1][j][k]);double deta=0.5*(yf[i][j+1][k]-yf[i][j-1][k])/Ll;double dze=0.5*(zf[i][j][k+1]-zf[i][j][k-1]);double dJ=detJ[i][j][k];double Fc[6][5];
    for(int f=0;f<6;f++){int comp=f/2,s=(f%2==0)?1:-1,di=(comp==0)?1:0,dj=(comp==1)?1:0,dk=(comp==2)?1:0,Li,Lj,Lk,Ri,Rj,Rk;if(s>0){Li=i;Lj=j;Lk=k;Ri=i+di;Rj=j+dj;Rk=k+dk;}else{Li=i-di;Lj=j-dj;Lk=k-dk;Ri=i;Rj=j;Rk=k;}
      double iJ=1.0/FJ[i][j][k][f];double mvx=FM[i][j][k][f][comp][0]*iJ,mvy=FM[i][j][k][f][comp][1]*iJ,mvz=FM[i][j][k][f][comp][2]*iJ;int tt=(comp==2)?1:0;double rF,ruF,rvF,rwF,reF;
      faceFlux(mvx,mvy,mvz,tt,rho[Li][Lj][Lk],rho[Ri][Rj][Rk],uu[Li][Lj][Lk],uu[Ri][Rj][Rk],vv[Li][Lj][Lk],vv[Ri][Rj][Rk],ww[Li][Lj][Lk],ww[Ri][Rj][Rk],EE[Li][Lj][Lk],EE[Ri][Rj][Rk],PP[Li][Lj][Lk],PP[Ri][Rj][Rk],ss[Li][Lj][Lk],ss[Ri][Rj][Rk],rF,ruF,rvF,rwF,reF);
      Fc[f][0]=rF;Fc[f][1]=ruF;Fc[f][2]=rvF;Fc[f][3]=rwF;Fc[f][4]=reF;}
    Ir[i][j][k]=(dJ/dxi)*(Fc[0][0]-Fc[1][0])+(dJ/deta)*(Fc[2][0]-Fc[3][0])+(dJ/dze)*(Fc[4][0]-Fc[5][0]);
    Iu[i][j][k]=(dJ/dxi)*(Fc[0][1]-Fc[1][1])+(dJ/deta)*(Fc[2][1]-Fc[3][1])+(dJ/dze)*(Fc[4][1]-Fc[5][1]);
    Iv[i][j][k]=(dJ/dxi)*(Fc[0][2]-Fc[1][2])+(dJ/deta)*(Fc[2][2]-Fc[3][2])+(dJ/dze)*(Fc[4][2]-Fc[5][2]);
    Iw[i][j][k]=(dJ/dxi)*(Fc[0][3]-Fc[1][3])+(dJ/deta)*(Fc[2][3]-Fc[3][3])+(dJ/dze)*(Fc[4][3]-Fc[5][3]);
    IE[i][j][k]=(dJ/dxi)*(Fc[0][4]-Fc[1][4])+(dJ/deta)*(Fc[2][4]-Fc[3][4])+(dJ/dze)*(Fc[4][4]-Fc[5][4]);}
}
void faceGrad(double F[NX][NY][NZ],int i,int j,int k,int f,int nd,int s,int ni,int nj,int nk,double g[3]){double dc[3];for(int c=0;c<3;c++)dc[c]=(c==nd)?fD(F,i,j,k,nd,s):0.5*(cC(F,i,j,k,c)+cC(F,ni,nj,nk,c));for(int d=0;d<3;d++)g[d]=FM[i][j][k][f][0][d]*dc[0]+FM[i][j][k][f][1][d]*dc[1]+FM[i][j][k][f][2][d]*dc[2];}
void viscous(){
  for(int i=1;i<=gNx;i++)for(int j=1;j<=gNy;j++)for(int k=1;k<=gNz;k++){
    double Ll=nozzleRadius(xf[i][j][k]);double dxi=0.5*(xf[i+1][j][k]-xf[i-1][j][k]);double deta=0.5*(yf[i][j+1][k]-yf[i][j-1][k])/Ll;double dze=0.5*(zf[i][j][k+1]-zf[i][j][k-1]);double dJ=detJ[i][j][k];
    double Tp[3][3],Tm[3][3],reP[3],reM[3];
    for(int axis=0;axis<3;axis++)for(int side=0;side<2;side++){int s=(side==0)?1:-1,f=2*axis+side,ni=i,nj=j,nk=k;if(axis==0)ni=i+s;else if(axis==1)nj=j+s;else nk=k+s;double gu[3],gv[3],gw[3],gT[3];faceGrad(uu,i,j,k,f,axis,s,ni,nj,nk,gu);faceGrad(vv,i,j,k,f,axis,s,ni,nj,nk,gv);faceGrad(ww,i,j,k,f,axis,s,ni,nj,nk,gw);faceGrad(TT,i,j,k,f,axis,s,ni,nj,nk,gT);
      double mua=mu_c,kaa=kappa_c,ua=0.5*(uu[i][j][k]+uu[ni][nj][nk]),va=0.5*(vv[i][j][k]+vv[ni][nj][nk]),wa=0.5*(ww[i][j][k]+ww[ni][nj][nk]),dv=gu[0]+gv[1]+gw[2],T0,T1,T2,q;
      if(axis==0){T0=2*mua*(gu[0]-dv/3);T1=mua*(gu[1]+gv[0]);T2=mua*(gu[2]+gw[0]);q=-kaa*gT[0];}else if(axis==1){T0=mua*(gv[0]+gu[1]);T1=2*mua*(gv[1]-dv/3);T2=mua*(gv[2]+gw[1]);q=-kaa*gT[1];}else{T0=mua*(gw[0]+gu[2]);T1=mua*(gw[1]+gv[2]);T2=2*mua*(gw[2]-dv/3);q=-kaa*gT[2];}
      double re=ua*T0+va*T1+wa*T2-q;if(side==0){Tp[axis][0]=T0;Tp[axis][1]=T1;Tp[axis][2]=T2;reP[axis]=re;}else{Tm[axis][0]=T0;Tm[axis][1]=T1;Tm[axis][2]=T2;reM[axis]=re;}}
    double mp[3][3],mm[3][3];for(int axis=0;axis<3;axis++){int fp=2*axis,fm=2*axis+1;double ip=1.0/FJ[i][j][k][fp],im=1.0/FJ[i][j][k][fm];for(int d=0;d<3;d++){mp[axis][d]=FM[i][j][k][fp][axis][d]*ip;mm[axis][d]=FM[i][j][k][fm][axis][d]*im;}}
    double delta[3]={dxi,deta,dze};double out[4];
    for(int comp=0;comp<4;comp++){double Cp[3],Cm[3];for(int axis=0;axis<3;axis++){if(comp<3){Cp[axis]=Tp[axis][comp];Cm[axis]=Tm[axis][comp];}else{Cp[axis]=reP[axis];Cm[axis]=reM[axis];}}double fl=0;for(int dir=0;dir<3;dir++){double dp=mp[dir][0]*Cp[0]+mp[dir][1]*Cp[1]+mp[dir][2]*Cp[2],dm=mm[dir][0]*Cm[0]+mm[dir][1]*Cm[1]+mm[dir][2]*Cm[2];fl+=(dJ/delta[dir])*(dp-dm);}out[comp]=fl;}
    Vu[i][j][k]=out[0];Vv[i][j][k]=out[1];Vw[i][j][k]=out[2];VE[i][j][k]=out[3];}
}
// ---------------- primitives / thermo from conserved ----------------
void consToPrim(){for(int i=1;i<=gNx;i++)for(int j=1;j<=gNy;j++)for(int k=1;k<=gNz;k++){double rr=rho[i][j][k];double u=ru[i][j][k]/rr,v=rv[i][j][k]/rr,w=rw[i][j][k]/rr,E=rE[i][j][k]/rr;double e=E-0.5*(u*u+v*v+w*w);double T=e/cv;double P=rr*Rg*T;uu[i][j][k]=u;vv[i][j][k]=v;ww[i][j][k]=w;EE[i][j][k]=E;TT[i][j][k]=T;PP[i][j][k]=P;ss[i][j][k]=sos_of(P,rr);}}
void primToCons(){for(int i=0;i<NX;i++)for(int j=0;j<NY;j++)for(int k=0;k<NZ;k++){double e=cv*TT[i][j][k];double E=e+0.5*(uu[i][j][k]*uu[i][j][k]+vv[i][j][k]*vv[i][j][k]+ww[i][j][k]*ww[i][j][k]);EE[i][j][k]=E;ss[i][j][k]=sos_of(PP[i][j][k],rho[i][j][k]);ru[i][j][k]=rho[i][j][k]*uu[i][j][k];rv[i][j][k]=rho[i][j][k]*vv[i][j][k];rw[i][j][k]=rho[i][j][k]*ww[i][j][k];rE[i][j][k]=rho[i][j][k]*E;}}

void setGhost(int i,int j,int k,double u,double v,double w,double P,double T){double rr=P/(Rg*T);double e=cv*T,ke=0.5*(u*u+v*v+w*w),E=e+ke;rho[i][j][k]=rr;uu[i][j][k]=u;vv[i][j][k]=v;ww[i][j][k]=w;PP[i][j][k]=P;TT[i][j][k]=T;EE[i][j][k]=E;ss[i][j][k]=sos_of(P,rr);ru[i][j][k]=rr*u;rv[i][j][k]=rr*v;rw[i][j][k]=rr*w;rE[i][j][k]=rr*E;}

void boundaries(){
  // West inflow (Aitken)
  double rho_ref=P_inlet/(Rg*T_inlet), e_ref=cv*T_inlet, h_ref=e_ref+P_inlet/rho_ref;
  for(int j=1;j<=gNy;j++)for(int k=1;k<=gNz;k++){int i=0;
    double wg_g=1.0-(x_0-xf[i][j][k])/(xf[i+1][j][k]-xf[i][j][k]);double wg_in=1.0-(xf[i+1][j][k]-x_0)/(xf[i+1][j][k]-xf[i][j][k]);
    double u_in=uu[i+1][j][k],v_in=vv[i+1][j][k],w_in=ww[i+1][j][k];double u_g=u_in,v_g=(0.0-wg_in*v_in)/wg_g,w_g=(0.0-wg_in*w_in)/wg_g;
    double P_g=PP[i][j][k],T_g=TT[i][j][k],rho_g=rho[i][j][k],x0A=rho_g;
    for(int it=0;it<100000;it++){P_g=P_inlet-(u_g*u_g)/(1.0/rho_ref+1.0/rho_g);double e_g=cv*T_g,h_g=h_ref-0.5*u_g*u_g;rho_g=P_g/(h_g-e_g);T_g=P_g/(rho_g*Rg);double x1A=rho_g;e_g=cv*T_g;P_g=P_inlet-(u_g*u_g)/(1.0/rho_ref+1.0/rho_g);rho_g=P_g/(h_g-e_g);T_g=P_g/(rho_g*Rg);double x2A=rho_g;double den=x2A-2*x1A+x0A;rho_g=x2A-((x2A-x1A)*(x2A-x1A))/(den+eps);T_g=P_g/(rho_g*Rg);if(fabs((rho_g-x2A)/rho_g)<1e-5){P_g=P_inlet-(u_g*u_g)/(1.0/rho_ref+1.0/rho_g);break;}x0A=rho_g;}
    setGhost(i,j,k,u_g,v_g,w_g,P_g,T_g);}
  // East outflow (NSCBC / supersonic switch)
  {int ie=gNx+1,je=1,ke=1;double u_i=0.5*(uu[ie][je][ke]+uu[ie-1][je][ke]),v_i=0.5*(vv[ie][je][ke]+vv[ie-1][je][ke]),w_i=0.5*(ww[ie][je][ke]+ww[ie-1][je][ke]);double V_i=sqrt(u_i*u_i+v_i*v_i+w_i*w_i),sos_i=0.5*(ss[ie][je][ke]+ss[ie-1][je][ke]),Ma=V_i/sos_i;
    for(int j=1;j<=gNy;j++)for(int k=1;k<=gNz;k++){int i=gNx+1;double u_in=uu[i-1][j][k],v_in=vv[i-1][j][k],w_in=ww[i-1][j][k],P_in=PP[i-1][j][k];
      if(Ma<1.2){double Dg=xf[i-1][j][k]-xf[i][j][k],Din=xf[i-2][j][k]-xf[i-1][j][k];double rin=rho[i-1][j][k],sin_=ss[i-1][j][k],Main=u_in/sin_,K=0.9*sin_*(1.0-Main*Main)/L_x;
        double dr=(rho[i-2][j][k]-rho[i-1][j][k])/Din,du=(uu[i-2][j][k]-uu[i-1][j][k])/Din,dv=(vv[i-2][j][k]-vv[i-1][j][k])/Din,dw=(ww[i-2][j][k]-ww[i-1][j][k])/Din,dP=(PP[i-2][j][k]-PP[i-1][j][k])/Din;
        double lam1=u_in-sin_,L1=K*(P_in-P_outlet)/lam1,L2=sin_*sin_*dr-dP,L3=dv,L4=dw,L5=dP+rin*sin_*du;
        double dQ1=(1.0/(sin_*sin_))*(L2+0.5*(L5+L1)),dQ2=(1.0/(2.0*rin*sin_))*(L5-L1),dQ3=L3,dQ4=L4,dQ5=0.5*(L5+L1);
        double rg=rin-Dg*dQ1,ug=u_in-Dg*dQ2,vg=v_in-Dg*dQ3,wg=w_in-Dg*dQ4,Pg=P_in-Dg*dQ5,Tg=Pg/(rg*Rg);setGhost(i,j,k,ug,vg,wg,Pg,Tg);}
      else setGhost(i,j,k,u_in,v_in,w_in,P_in,TT[i-1][j][k]);}}
  // South symmetry
  for(int i=1;i<=gNx;i++)for(int k=1;k<=gNz;k++){int j=0,jin=1;double Ly=nozzleRadius(xf[i][j][k]);double eg=yf[i][j][k]/Ly,ein=yf[i][jin][k]/Ly;double wg_g=1.0-(y_0-eg)/(ein-eg),wg_in=1.0-(ein-y_0)/(ein-eg);double v_in=vv[i][jin][k];setGhost(i,j,k,uu[i][jin][k],(0.0-wg_in*v_in)/wg_g,ww[i][jin][k],PP[i][jin][k],TT[i][jin][k]);}
  // North wall (curvilinear Neumann P/T, no-slip)
  for(int i=1;i<=gNx;i++)for(int k=1;k<=gNz;k++){int j=gNy+1,jin=gNy;double Ly=nozzleRadius(xf[i][j][k]);double eg=yf[i][j][k]/Ly,ein=yf[i][jin][k]/Ly;double Ln=0.5*(eg+ein),wg_g=1.0-(eg-Ln)/(eg-ein),wg_in=1.0-(Ln-ein)/(eg-ein);
    double xde=XI[i][j][k][0]*ET[i][j][k][0]+XI[i][j][k][1]*ET[i][j][k][1]+XI[i][j][k][2]*ET[i][j][k][2];
    double zde=ZE[i][j][k][0]*ET[i][j][k][0]+ZE[i][j][k][1]*ET[i][j][k][1]+ZE[i][j][k][2]*ET[i][j][k][2];
    double ede=ET[i][j][k][0]*ET[i][j][k][0]+ET[i][j][k][1]*ET[i][j][k][1]+ET[i][j][k][2]*ET[i][j][k][2];
    double Pxi=(PP[i+1][jin][k]-PP[i-1][jin][k])/(xf[i+1][jin][k]-xf[i-1][jin][k]),Pze=(PP[i][jin][k+1]-PP[i][jin][k-1])/(zf[i][jin][k+1]-zf[i][jin][k-1]);
    double Pg=PP[i][jin][k]+(-(xde*Pxi+zde*Pze)/ede)*(eg-ein);
    double Txi=(TT[i+1][jin][k]-TT[i-1][jin][k])/(xf[i+1][jin][k]-xf[i-1][jin][k]),Tze=(TT[i][jin][k+1]-TT[i][jin][k-1])/(zf[i][jin][k+1]-zf[i][jin][k-1]);
    double Tg=TT[i][jin][k]+(-(xde*Txi+zde*Tze)/ede)*(eg-ein);
    double u_in=uu[i][jin][k],v_in=vv[i][jin][k],w_in=ww[i][jin][k];
    setGhost(i,j,k,(0.0-wg_in*u_in)/wg_g,(0.0-wg_in*v_in)/wg_g,(0.0-wg_in*w_in)/wg_g,Pg,Tg);}
  // Z (homogeneous): Neumann copy so zeta faces contribute ~0
  for(int i=0;i<NX;i++)for(int j=0;j<NY;j++){for(int k=0;k<1;k++){}
    setGhost(i,j,0,uu[i][j][1],vv[i][j][1],ww[i][j][1],PP[i][j][1],TT[i][j][1]);
    setGhost(i,j,2,uu[i][j][1],vv[i][j][1],ww[i][j][1],PP[i][j][1],TT[i][j][1]);}
}
// Time step from the true PHYSICAL cell spacings (geometric CFL) — guarantees stability on
// the body-fitted grid (the throat cells are physically tiny). Note: this differs from the
// Jacobian-scaled delta_y in the faithful RHEA port (Task #8); here stability is the priority.
double timeStep(){double dt=1e30;for(int i=1;i<=gNx;i++)for(int j=1;j<=gNy;j++)for(int k=1;k<=gNz;k++){double sos=ss[i][j][k],c_p=cp;
  double dx=0.5*(xf[i+1][j][k]-xf[i-1][j][k]);double dy=0.5*(yf[i][j+1][k]-yf[i][j-1][k]);double dz=0.5*(zf[i][j][k+1]-zf[i][j][k-1]);
  double Sx=fabs(uu[i][j][k])+sos;dt=min(dt,CFL*fabs(dx)/Sx);dt=min(dt,CFL*rho[i][j][k]*dx*dx/max(mu_c,eps));dt=min(dt,CFL*rho[i][j][k]*c_p*dx*dx/max(kappa_c,eps));
  double Sy=fabs(vv[i][j][k])+sos;dt=min(dt,CFL*fabs(dy)/Sy);dt=min(dt,CFL*rho[i][j][k]*dy*dy/max(mu_c,eps));dt=min(dt,CFL*rho[i][j][k]*c_p*dy*dy/max(kappa_c,eps));}
  return dt;}

// ---------------- HDF5 output (RHEA-style: per-field datasets + XDMF sidecar, gzip) ----------------
static void h5_write2d(hid_t file,const char*name,double*buf){hsize_t dims[2]={(hsize_t)gNx,(hsize_t)gNy};hid_t sp=H5Screate_simple(2,dims,NULL);hid_t dcpl=H5Pcreate(H5P_DATASET_CREATE);hsize_t ch[2]={(hsize_t)gNx,(hsize_t)gNy};H5Pset_chunk(dcpl,2,ch);H5Pset_deflate(dcpl,6);hid_t ds=H5Dcreate2(file,name,H5T_NATIVE_DOUBLE,sp,H5P_DEFAULT,dcpl,H5P_DEFAULT);H5Dwrite(ds,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,buf);H5Dclose(ds);H5Pclose(dcpl);H5Sclose(sp);}
static void h5_attr_i(hid_t f,const char*n,int v){hid_t s=H5Screate(H5S_SCALAR);hid_t a=H5Acreate2(f,n,H5T_NATIVE_INT,s,H5P_DEFAULT,H5P_DEFAULT);H5Awrite(a,H5T_NATIVE_INT,&v);H5Aclose(a);H5Sclose(s);}
static void h5_attr_d(hid_t f,const char*n,double v){hid_t s=H5Screate(H5S_SCALAR);hid_t a=H5Acreate2(f,n,H5T_NATIVE_DOUBLE,s,H5P_DEFAULT,H5P_DEFAULT);H5Awrite(a,H5T_NATIVE_DOUBLE,&v);H5Aclose(a);H5Sclose(s);}
void writeOutput(int iter,double t){
  static double bx[gNx*gNy],by[gNx*gNy],br[gNx*gNy],bu[gNx*gNy],bv[gNx*gNy],bP[gNx*gNy],bT[gNx*gNy],bM[gNx*gNy];
  int n=0;for(int i=1;i<=gNx;i++)for(int j=1;j<=gNy;j++){int k=1;double V=sqrt(uu[i][j][k]*uu[i][j][k]+vv[i][j][k]*vv[i][j][k]);
    bx[n]=xf[i][j][k];by[n]=yf[i][j][k];br[n]=rho[i][j][k];bu[n]=uu[i][j][k];bv[n]=vv[i][j][k];bP[n]=PP[i][j][k];bT[n]=TT[i][j][k];bM[n]=V/ss[i][j][k];n++;}
  char fn[128];sprintf(fn,"nozzle_cd_%06d.h5",iter);
  hid_t f=H5Fcreate(fn,H5F_ACC_TRUNC,H5P_DEFAULT,H5P_DEFAULT);
  h5_attr_i(f,"Iteration",iter);h5_attr_d(f,"Time",t);
  h5_write2d(f,"x",bx);h5_write2d(f,"y",by);h5_write2d(f,"rho",br);h5_write2d(f,"u",bu);h5_write2d(f,"v",bv);h5_write2d(f,"P",bP);h5_write2d(f,"T",bT);h5_write2d(f,"Ma",bM);
  H5Fclose(f);
  char xn[128];sprintf(xn,"nozzle_cd_%06d.xmf",iter);FILE*x=fopen(xn,"w");
  fprintf(x,"<?xml version=\"1.0\" ?>\n<Xdmf Version=\"2.0\">\n <Domain>\n  <Grid Name=\"nozzle\" GridType=\"Uniform\">\n   <Time Value=\"%.9e\"/>\n",t);
  fprintf(x,"   <Topology TopologyType=\"2DSMesh\" Dimensions=\"%d %d\"/>\n   <Geometry GeometryType=\"X_Y\">\n    <DataItem Dimensions=\"%d %d\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">%s:/x</DataItem>\n    <DataItem Dimensions=\"%d %d\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">%s:/y</DataItem>\n   </Geometry>\n",gNx,gNy,gNx,gNy,fn,gNx,gNy,fn);
  const char*fl[]={"rho","u","v","P","T","Ma"};for(int q=0;q<6;q++)fprintf(x,"   <Attribute Name=\"%s\" AttributeType=\"Scalar\" Center=\"Node\">\n    <DataItem Dimensions=\"%d %d\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">%s:/%s</DataItem>\n   </Attribute>\n",fl[q],gNx,gNy,fn,fl[q]);
  fprintf(x,"  </Grid>\n </Domain>\n</Xdmf>\n");fclose(x);
}

int main(int argc,char**argv){
  int NITER = (argc>1)? atoi(argv[1]) : 20000;
  int OUT   = (argc>2)? atoi(argv[2]) : 2000;
  buildGridMetrics();
  // initial uniform
  for(int i=0;i<NX;i++)for(int j=0;j<NY;j++)for(int k=0;k<NZ;k++){uu[i][j][k]=U_inlet;vv[i][j][k]=0;ww[i][j][k]=0;PP[i][j][k]=P_inlet;TT[i][j][k]=T_inlet;rho[i][j][k]=P_inlet/(Rg*T_inlet);}
  primToCons(); boundaries();
  double t=0;
  for(int it=0;it<NITER;it++){
    double dt=timeStep();
    // store U0
    memcpy(r0,rho,sizeof(rho));memcpy(ru0,ru,sizeof(ru));memcpy(rv0,rv,sizeof(rv));memcpy(rw0,rw,sizeof(rw));memcpy(rE0,rE,sizeof(rE));
    for(int stage=0;stage<3;stage++){
      inviscid(); viscous();
      double a0,a1,a2; if(stage==0){a0=1;a1=0;a2=1;}else if(stage==1){a0=0.75;a1=0.25;a2=0.25;}else{a0=1.0/3.0;a1=2.0/3.0;a2=2.0/3.0;}
      for(int i=1;i<=gNx;i++)for(int j=1;j<=gNy;j++)for(int k=1;k<=gNz;k++){
        double Lr=-Ir[i][j][k], Lu=-Iu[i][j][k]+Vu[i][j][k], Lv=-Iv[i][j][k]+Vv[i][j][k], Lw=-Iw[i][j][k]+Vw[i][j][k], LE=-IE[i][j][k]+VE[i][j][k];
        rho[i][j][k]=a0*r0[i][j][k]+a1*rho[i][j][k]+a2*dt*Lr;
        ru[i][j][k]=a0*ru0[i][j][k]+a1*ru[i][j][k]+a2*dt*Lu;
        rv[i][j][k]=a0*rv0[i][j][k]+a1*rv[i][j][k]+a2*dt*Lv;
        rw[i][j][k]=a0*rw0[i][j][k]+a1*rw[i][j][k]+a2*dt*Lw;
        rE[i][j][k]=a0*rE0[i][j][k]+a1*rE[i][j][k]+a2*dt*LE;}
      consToPrim(); boundaries();
    }
    t+=dt;
    if(it%OUT==0 || it==NITER-1){
      // throat + exit Mach on centerline (j=1)
      int it_x=1;double rmin=1e9;for(int i=1;i<=gNx;i++){double rr=nozzleRadius(xf[i][1][1]);if(rr<rmin){rmin=rr;it_x=i;}}
      double Vth=sqrt(uu[it_x][1][1]*uu[it_x][1][1]+vv[it_x][1][1]*vv[it_x][1][1]),Mth=Vth/ss[it_x][1][1];
      double Vex=fabs(uu[gNx][1][1]),Mex=Vex/ss[gNx][1][1];
      printf("it=%6d t=%.4e dt=%.2e  Ma_throat=%.3f  Ma_exit=%.3f  P_exit=%.3f bar  rho_th=%.2f\n",it,t,dt,Mth,Mex,PP[gNx][1][1]/1e5,rho[it_x][1][1]);
      fflush(stdout);
      writeOutput(it,t);   // compact HDF5 snapshot (+ XDMF) every OUT iterations
    }
  }
  printf("done; wrote HDF5 snapshots nozzle_cd_<iter>.h5 (+ .xmf)\n");
  return 0;
}
