#include<stdio.h>
#include<math.h>
#include<gsl/gsl_math.h>
#include<gsl/gsl_roots.h>
#include<gsl/gsl_errno.h>

#include "planet.h"

/* This file contains code that solves for the true anomaly */

struct tE_params       {
         double e, M;       
};   

double pl_tE(double x, void * params){
  
  struct tE_params *p = (struct tE_params *) params;

  double M = p->M;
  double e = p->e;
  
  return (x - M - e * sin(x));
}

double pl_tE_deriv(double x, void * params){

  struct tE_params *p = (struct tE_params *) params;

  double e = p->e;

  return (1.0 - e * cos(x));

}

void pl_tE_fdf(double x, void * params, double *y, double *dy){

  struct tE_params *p = (struct tE_params *) params;

  double M = p->M;
  double e = p->e;

  *y = (x - M - e * sin(x));
  *dy = (1.0 - e * cos(x));

}


double pl_true_anomaly(double t, double t0, double e, double P){
  
  /* local variables */
  double M; /* mean anomaly */
  double f, cosf; /* return value */

  struct tE_params params;
  gsl_function_fdf FDF;
  const gsl_root_fdfsolver_type *T;
  gsl_root_fdfsolver *s;

  int iter=0, max_iter=10000;
  double x0;
  double x = M_PI;
  int status;
  double mod;
  
  /* TODO: catch errors with P <= 0 here */

  /* firts, get the mean anomaly */

  mod = fmod((t-t0)/P, 1.0);

  M = 2.0 * M_PI * mod;

  /* set first guess according to Charles and Totum 1997 */

  /*  x = M + e * ( pow((M_PI * M_PI * M), (1.0/3.0)) - (M_PI / 15.0) * sin(M) - M); */

  /* set root solver */
  params.e = e;
  params.M = M;

  FDF.f  = &pl_tE;
  FDF.df = &pl_tE_deriv;
  FDF.fdf = &pl_tE_fdf;
  FDF.params = &params;
  
  /* now, calculate eccentric anomaly */

  T = gsl_root_fdfsolver_newton;
  s = gsl_root_fdfsolver_alloc (T);
  gsl_root_fdfsolver_set (s, &FDF, x);  

  do{
    iter++;
    status = gsl_root_fdfsolver_iterate (s);
    x0 = x;           
    x = gsl_root_fdfsolver_root (s);
    status = gsl_root_test_delta (x, x0, 0, 1e-4);


    #ifdef DEBUG

    if (status == GSL_SUCCESS)
      printf ("Converged:\n");   

    printf ("%5d %10.7f %10.7f %10.7f %10.7f %10.7f%10.7f \n",
	    iter, x0, x, x - x0, M, mod, P);
    #endif
  }
  while (status == GSL_CONTINUE && iter < max_iter);
  
  /* die here if status is bad */
  gsl_root_fdfsolver_free (s);

  cosf = (cos(x) - e) / (1 - e * cos(x));
  f = acos(cosf);
  
  if ((M > M_PI) || ((M < 0) && (M > -M_PI)))
    f = 2*M_PI - f;

  return f;
}

