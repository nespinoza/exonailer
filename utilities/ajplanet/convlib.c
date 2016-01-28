#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "planet.h"

double clight = 299792.458;

double pl_gau(double x, double mu, double sigma)
{
  return (1.0 / sqrt(2.0 * 3.1415) * sigma) * 
    exp(-0.5 * ((x-mu) * (x-mu)) / (sigma * sigma));
}

void pl_Rconvolv(double * y,double * ys, long num, double st, double dw, double R)
{
  double w,s,t,nd, sigma;
  long i,n1,n2,n;

  /* End Effect */
  
  sigma = st / (2.35 * R);
  nd = (long) ceil(3 * sigma / dw);

  n1 = nd + 1;
  for(i=0;i<n1;i++) ys[i] = y[i];

  sigma = (st + ((double) num)*dw)/ (2.35 * R);
  nd = (long) ceil(3 * sigma / dw);

  n2 = num - nd - 1;
  for(i=n2;i<num;i++) ys[i] = y[i];


/* convolve with Gaussian */

   w = st + (n1-1)*dw;

   for(n=n1;n<n2;n++) {

     /* central wave */
     w = w+dw; 
     s = 0.0;
     t = 0.0;

     sigma = w / (2.35 * R);
     nd = (long) ceil(3 * sigma / dw);

     for(i=-nd;i<=nd;i++) {
       t = t + pl_gau((-(double) i) * dw, 0.0, sigma);
       s = s + pl_gau((-(double) i) * dw, 0.0, sigma) * y[n+i];     
     }
     ys[n] = s/t;
   }
   return;

}


void pl_convolv(double * y,double * ys,long num, double st, double dw, double vsini, double u)
{
  double beta,gam,w,s,t,dlc,c1,c2,dv,r2,f,v, nd, s2;
  long i,n1,n2,n;

  s2 = (st + ((double) num)*dw)*vsini/(dw*clight);
  nd = s2 + 5.5;

  beta = (1.0-u)/(1.0 - 0.333333*u);
  gam = u/(1.0 - 0.333333*u);  

/* End Effect */

  n1 = nd + 1;
  for(i=0;i<nd;i++) ys[i] = y[i];
  n2 = num - nd -1;
  for(i=n2;i<num;i++) ys[i] = y[i];
  if(vsini < 0.5) {
    for(i=0;i<num;i++) ys[i] = y[i];
    return;
  }

/* convolve with rotation profile */

   w = st + (n1-1)*dw;
   for(n=n1;n<n2;n++) {
     w = w+dw;
     s = 0.0;
     t = 0.0;
     dlc = w*vsini/clight;
     c1 = 0.63661977*beta/dlc;
     c2 = 0.5*gam/dlc;
     dv = dw/dlc;

     for(i=-nd;i<=nd;i++) {
       v = i*dv;
       r2 = 1.0 - v*v;
       if(r2 > 0.0) {
         f = c1*sqrt(r2) + c2*r2;
         t = t+f;
         s = s + f*y[n+i];
       }
     }
     ys[n] = s/t;
   }
   return;
}
