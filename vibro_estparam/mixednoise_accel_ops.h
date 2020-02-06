//# regenerate dqagse_fparams.c with:
//# f2c -a dqagse_fparams.f
//# patch -p0 <dqagse_fparams.patch

#include <omp.h>
#include <float.h>
#include <math.h>

// definitions for dqagse_fparams.c:
//#define M_PI_F M_PI


typedef double doublereal;
typedef float real;  // all calculations in double precision
typedef int logical;
typedef int integer;



typedef double (*D_fp)(double *fp0,double *fp1,double *fp2,double *fp3,double *fp4);  


double dabs(double p) { return fabs(p); }
double dmax(double p,double q) { if (p > q) return p;else return q; }
double dmin(double p,double q) { if (p < q) return p;else return q; }

static doublereal pow_dd(doublereal *arg1,doublereal *arg2)
{
  return pow(*arg1,*arg2);
}

/* C source for R1MACH -- remove the * in column 1 */
static doublereal d1mach_(integer *i)
{
	switch(*i){
	  case 1: return FLT_MIN;
	  case 2: return FLT_MAX;
	  case 3: return FLT_EPSILON/FLT_RADIX;
	  case 4: return FLT_EPSILON;
	  case 5: return log10((float)FLT_RADIX);
	  }
        printf("invalid argument: r1mach(%ld)\n",(long int) *i);
        return 0.0f;
//	assert(0); return 0; /* else complaint of missing return value */
}

#define TRUE_ 1
#define FALSE_ 0

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif
#ifdef abs
#undef abs
#endif

#define max dmax
#define min dmin
#define abs dabs

#define dqagse_ dqagse_mna
#define dqagie_ dqagie_mna
#define dqk15i_ dqk15i_mna
#define dqelg_ dqelg_mna
#define dqk21_ dqk21_mna
#define dqpsrt_ dqpsrt_mna


#include "dqagse_fparams.c"

#undef max
#undef min


// cachelookup and cacheadd are implemented in mixednoise_accel.pyx
double cachelookup(PyObject *evaluation_cache,double sigma_additive,double sigma_multiplicative, double prediction_indexed,double observed_indexed);
void cacheadd(PyObject *evaluation_cache,double sigma_additive,double sigma_multiplicative, double prediction_indexed,double observed_indexed,double p_value);
double evaluate_y_zero_to_eps(PyObject *integral_y_zero_to_eps,double sigma_additive,double sigma_multiplicative,double prediction_indexed,double observed_indexed,double eps);

static inline double lognormal_normal_convolution_kernel_core(double y,double sigma_additive,double sigma_multiplicative,double prediction_indexed,double observed_indexed)
{
  return (1.0/(y*sigma_multiplicative*sqrt(2.0*M_PI)))*exp(-(pow(log(y)-log(prediction_indexed),2.0))/(2.0*pow(sigma_multiplicative,2.0)))*(1.0/(sigma_additive*sqrt(2.0*M_PI)))*exp(-(pow(observed_indexed-y,2.0))/(2.0*pow(sigma_additive,2.0)));

}

static double lognormal_normal_convolution_kernel(double *yp,double *sigma_additivep,double *sigma_multiplicativep,double *prediction_indexedp,double *observed_indexedp)
{
  double y,sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,res;
  y=*yp;
  sigma_additive=*sigma_additivep;
  sigma_multiplicative=*sigma_multiplicativep;
  prediction_indexed=*prediction_indexedp;
  observed_indexed=*observed_indexedp;

  
  res = lognormal_normal_convolution_kernel_core(y,sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed);
  //fprintf(stderr,"kernel(%g,%g,%g,%g,%g) returns %g\n", y,sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,res);
  return res;
}

static double lognormal_normal_convolution_kernel_deriv_sigma_additive(double *yp,double *sigma_additivep,double *sigma_multiplicativep,double *prediction_indexedp,double *observed_indexedp)
{
  double y,sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,res;
  y=*yp;
  sigma_additive=*sigma_additivep;
  sigma_multiplicative=*sigma_multiplicativep;
  prediction_indexed=*prediction_indexedp;
  observed_indexed=*observed_indexedp;

  
  res = lognormal_normal_convolution_kernel_core(y,sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed)*( ((pow(observed_indexed-y,2.0))/(pow(sigma_additive,3.0))) - (1.0/sigma_additive));
  //fprintf(stderr,"kernel_dsa(%g,%g,%g,%g,%g) returns %g\n", y,sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,res);
  return res;
}


static double lognormal_normal_convolution_kernel_deriv_sigma_multiplicative(double *yp,double *sigma_additivep,double *sigma_multiplicativep,double *prediction_indexedp,double *observed_indexedp)
{
  double y,sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed;
  y=*yp;
  sigma_additive=*sigma_additivep;
  sigma_multiplicative=*sigma_multiplicativep;
  prediction_indexed=*prediction_indexedp;
  observed_indexed=*observed_indexedp;

  return lognormal_normal_convolution_kernel_core(y,sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed)*( ((pow(log(y) - log(prediction_indexed),2.0))/(pow(sigma_multiplicative,3.0))) - (1.0/sigma_multiplicative));
  
}


static double lognormal_normal_convolution_kernel_deriv_prediction(double *yp,double *sigma_additivep,double *sigma_multiplicativep,double *prediction_indexedp,double *observed_indexedp)
{
  double y,sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed;
  y=*yp;
  sigma_additive=*sigma_additivep;
  sigma_multiplicative=*sigma_multiplicativep;  
  prediction_indexed=*prediction_indexedp;
  observed_indexed=*observed_indexedp;

  return lognormal_normal_convolution_kernel_core(y,sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed)* ((log(y) - log(prediction_indexed))/(pow(sigma_multiplicative,2.0)*prediction_indexed));
}


int sort_bounds_compar(const void *bound1,const void *bound2)
{
  if (*(double *)bound1 > *(double*)bound2) return 1;
  if (*(double *)bound1 < *(double*)bound2) return -1;
  return 0;
}
static void sort_bounds(double *bounds,double minbound,unsigned n)
{
  unsigned cnt;
  qsort(bounds,n,sizeof(double),sort_bounds_compar);

  for (cnt=0;cnt < n;cnt++) {
    if (bounds[cnt] < minbound) {
      bounds[cnt]=minbound;
    }
  }
}

static double integrate_convolution_c_one(PyObject *lognormal_normal_convolution_integral_y_zero_to_eps,D_fp funct, double sigma_additive,double sigma_multiplicative, double prediction_indexed,double observed_indexed,double eps)
{
  double singular_portion;
  double epsabs=3e-25;
  double epsrel=1e-16;
  integer inf=1; // infinite integration range corresponding to (bound,+infinity)
  integer limit=50; // following scipy.integrate.quad() (!)
  double p1=0.0,p2=0.0,p3=0.0,p4=0.0,p5=0.0;
  double p1err=0.0,p2err=0.0,p3err=0.0,p4err=0.0,p5err=0.0;
  double bounds[5];
  integer neval=0,ier=0;
  double *alist;
  double *blist;
  double *rlist;
  double *elist;
  integer *iord;
  integer last=0;
  alist = malloc(sizeof(*alist)*limit);
  blist = malloc(sizeof(*blist)*limit);
  rlist = malloc(sizeof(*rlist)*limit);
  elist = malloc(sizeof(*elist)*limit);
  iord = malloc(sizeof(*iord)*limit);
  
  singular_portion = evaluate_y_zero_to_eps(lognormal_normal_convolution_integral_y_zero_to_eps,sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,eps);

  bounds[0]=eps;
  bounds[1]=observed_indexed-sigma_additive;
  bounds[2]=observed_indexed+sigma_additive;
  // also got ln y - ln a0 = +- sigma_multiplicative
  // or ln y = ln_a0 +- sigma_multiplicative
  // or y = exp(ln(a0) +- sigma_multiplicative
  // or y = a0*exp(+- sigma_multiplicative
  bounds[3]=prediction_indexed * exp(-sigma_multiplicative);
  bounds[4]=prediction_indexed * exp(sigma_multiplicative);

  sort_bounds(bounds,eps,sizeof(bounds)/sizeof(*bounds));
  
  //fprintf(stderr,"integrating from %g to %g\n",bounds[0],bounds[1]);

  if (bounds[0] < bounds[1]) {
    dqagse_mna(funct,
	       &sigma_additive,&sigma_multiplicative,&prediction_indexed,&observed_indexed,
	       &bounds[0],&bounds[1], // integration bounds
	       &epsabs,&epsrel,
	       &limit,
	       &p1,
	       &p1err,
	       &neval,
	       &ier,
	       alist,blist,rlist,elist,iord,
	       &last);
    
    
    //printf("ier=%d\n",ier);

    //printf("integral from %g to %g, sa=%g, sm=%g, pi=%g, oi=%g,ea=%g,er=%g\n",bounds[0],bounds[1],sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,epsabs,epsrel);
  }

  //fprintf(stderr,"integrating from %g to %g\n",bounds[1],bounds[2]);
  
  if (bounds[1] < bounds[2]) {
    dqagse_mna(funct,
	       &sigma_additive,&sigma_multiplicative,&prediction_indexed,&observed_indexed,
	       &bounds[1],&bounds[2], // integration bounds
	       &epsabs,&epsrel,
	       &limit,
	       &p2,
	       &p2err,
	       &neval,
	       &ier,
	       alist,blist,rlist,elist,iord,
	       &last);
    
  
    //printf("ier=%d\n",ier);
    
    //printf("integral from %g to %g, sa=%g, sm=%g, pi=%g, oi=%g,ea=%g,er=%g\n",bounds[1],bounds[2],sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,epsabs,epsrel);

  }

  
  //fprintf(stderr,"integrating from %g to %g\n",bounds[2],bounds[3]);
  if (bounds[2] < bounds[3]) {
    dqagse_mna(funct,
	       &sigma_additive,&sigma_multiplicative,&prediction_indexed,&observed_indexed,
	       &bounds[2],&bounds[3], // integration bounds
	       &epsabs,&epsrel,
	       &limit,
	       &p3,
	       &p3err,
	       &neval,
	       &ier,
	       alist,blist,rlist,elist,iord,
	       &last);
  
    //printf("ier=%d\n",ier);
  
    //printf("integral from %g to %g, sa=%g, sm=%g, pi=%g, oi=%g,ea=%g,er=%g\n",bounds[2],bounds[3],sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,epsabs,epsrel);
  }



    
  //fprintf(stderr,"integrating from %g to %g\n",bounds[3],bounds[4]);
  if (bounds[3] < bounds[4]) {
    dqagse_mna(funct,
	       &sigma_additive,&sigma_multiplicative,&prediction_indexed,&observed_indexed,
	       &bounds[3],&bounds[4], // integration bounds
	       &epsabs,&epsrel,
	       &limit,
	       &p4,
	       &p4err,
	       &neval,
	       &ier,
	       alist,blist,rlist,elist,iord,
	       &last);
    
    
    //printf("ier=%d\n",ier);
    //printf("integral from %g to %g, sa=%g, sm=%g, pi=%g, oi=%g,ea=%g,er=%g\n",bounds[3],bounds[4],sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,epsabs,epsrel);
  }
  





  
  //fprintf(stderr,"p1err=%g\n",p1err);
  // result from portion 1 stored in p1
  epsabs=1e-24; // following Python code... 
  //fprintf(stderr,"integrating from %g to Inf\n",bounds[4]);
  
  dqagie_mna(funct,
	     &sigma_additive,&sigma_multiplicative,&prediction_indexed,&observed_indexed,
	     &bounds[4],&inf, // integration bounds
	     &epsabs,&epsrel,
	     &limit,
	     &p5,
	     &p5err,
	     &neval,
	     &ier,
	     alist,blist,rlist,elist,iord,
	     &last);
  //fprintf(stderr,"p2err=%g\n",p2err);

  
  free(alist);
  free(blist);
  free(rlist);
  free(elist);
  free(iord);

  /***!!!!*** Should at least inspect ier... */
  //printf("ier=%d\n",ier);

  //printf("mnao_integrate_kernel returns %g from %g, %g, %g, %g, %g and %g; p1err=%g, p2err=%g,p3err=%g,p4err=%g,p5err=%g\n",(singular_portion+p1+p2+p3+p4+p5),(singular_portion),(p1),(p2),(p3),(p4),(p5),p1err,p2err,p3err,p4err,p5err);
  
  //fflush(stdout);

  return singular_portion + p1 + p2 + p3 + p4 + p5;
  
  
}

static void integrate_lognormal_normal_convolution_c(PyObject *lognormal_normal_convolution_integral_y_zero_to_eps,PyObject *evaluation_cache,double sigma_additive,double sigma_multiplicative, double *prediction,double *observed,double *p,unsigned n)
{
  unsigned itercnt;
#pragma omp parallel for shared(lognormal_normal_convolution_integral_y_zero_to_eps,evaluation_cache,sigma_additive,sigma_multiplicative,prediction,observed,p,n) default(none) private(itercnt)
  for(itercnt=0;itercnt < n;itercnt++) {
    double prediction_indexed,observed_indexed;
    double p_value;
    double eps;
    
    prediction_indexed = prediction[itercnt];
    observed_indexed = observed[itercnt];
    
    eps = observed_indexed/100.0;

    //double one=1.0;
    
    //printf("kernel(1,1,1,1,1)=%g\n",lognormal_normal_convolution_kernel(&one,&one,&one,&one,&one));
    
    p_value = cachelookup(evaluation_cache,sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed);
    if (isnan(p_value)) {
      // cache lookup failed: Calculate!
      p_value = integrate_convolution_c_one(lognormal_normal_convolution_integral_y_zero_to_eps,lognormal_normal_convolution_kernel,sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,eps);
      cacheadd(evaluation_cache,sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,p_value);
      
    }
    p[itercnt]=p_value;
    //printf("integrated value %g\n",p_value);
  }
}



static void integrate_deriv_sigma_additive_c(PyObject *lognormal_normal_convolution_integral_y_zero_to_eps_deriv_sigma_additive,double sigma_additive,double sigma_multiplicative, double *prediction,double *observed,double *dp,unsigned n)
{
  unsigned itercnt;
#pragma omp parallel for shared(lognormal_normal_convolution_integral_y_zero_to_eps_deriv_sigma_additive,sigma_additive,sigma_multiplicative,prediction,observed,dp,n,stderr) default(none) private(itercnt)
  for(itercnt=0;itercnt < n;itercnt++) {
    double prediction_indexed,observed_indexed;
    double dp_value;
    double eps;
    
    prediction_indexed = prediction[itercnt];
    observed_indexed = observed[itercnt];
    
    eps = observed_indexed/100.0;
    
    dp_value = integrate_convolution_c_one(lognormal_normal_convolution_integral_y_zero_to_eps_deriv_sigma_additive,lognormal_normal_convolution_kernel_deriv_sigma_additive,sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,eps);
    if (isnan(dp_value) || !isfinite(dp_value)) {
      fprintf(stderr,"idsac: got dp_value NaN, sa=%g sm=%g pi=%g oi=%g eps=%g\n",sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,eps);
      if (fabs(sigma_additive) < 1e-20 || fabs(sigma_multiplicative) < 1e-20)  {
	// off the meaningful domain... but sampler seems to get here sometimes during tuning
	dp_value=0.0;
      } else {
	assert(0);
      }
    }

    dp[itercnt]=dp_value;
  }
}


static void integrate_deriv_sigma_multiplicative_c(PyObject *lognormal_normal_convolution_integral_y_zero_to_eps_deriv_sigma_multiplicative,double sigma_additive,double sigma_multiplicative, double *prediction,double *observed,double *dp,unsigned n)
{
  unsigned itercnt;
#pragma omp parallel for shared(lognormal_normal_convolution_integral_y_zero_to_eps_deriv_sigma_multiplicative,sigma_additive,sigma_multiplicative,prediction,observed,dp,n,stderr) default(none) private(itercnt)
  for(itercnt=0;itercnt < n;itercnt++) {
    double prediction_indexed,observed_indexed;
    double dp_value;
    double eps;
    
    prediction_indexed = prediction[itercnt];
    observed_indexed = observed[itercnt];
    
    eps = observed_indexed/100.0;
    
    dp_value = integrate_convolution_c_one(lognormal_normal_convolution_integral_y_zero_to_eps_deriv_sigma_multiplicative,lognormal_normal_convolution_kernel_deriv_sigma_multiplicative,sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,eps);
    if (isnan(dp_value) || !isfinite(dp_value)) {
      if (prediction_indexed != 0.0) {
	fprintf(stderr,"idsmc: got dp_value NaN, sa=%g sm=%g pi=%g oi=%g eps=%g\n",sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,eps);
	if (fabs(sigma_additive) < 1e-20 || fabs(sigma_multiplicative) < 1e-20)  {
	  // off the meaningful domain... but sampler seems to get here sometimes during tuning
	  dp_value=0.0;
	} else {
	  assert(prediction_indexed==0.0); // know this happens in this case and it's OK because the exponential from the convolution makes the derivative zero
	  // fail!
	}

      }
      dp_value=0.0;
    }
    dp[itercnt]=dp_value;
  }
}



static void integrate_deriv_prediction_c(PyObject *lognormal_normal_convolution_integral_y_zero_to_eps_deriv_prediction,double sigma_additive,double sigma_multiplicative, double *prediction,double *observed,double *dp,unsigned n)
{
  unsigned itercnt;
#pragma omp parallel for shared(lognormal_normal_convolution_integral_y_zero_to_eps_deriv_prediction,sigma_additive,sigma_multiplicative,prediction,observed,dp,n,stderr) default(none) private(itercnt)
  for(itercnt=0;itercnt < n;itercnt++) {
    double prediction_indexed,observed_indexed;
    double dp_value;
    double eps;
    
    prediction_indexed = prediction[itercnt];
    observed_indexed = observed[itercnt];

    //printf("pid %d omp tid %d\n",getpid(),omp_get_thread_num());
    
    eps = observed_indexed/100.0;
    
    dp_value = integrate_convolution_c_one(lognormal_normal_convolution_integral_y_zero_to_eps_deriv_prediction,lognormal_normal_convolution_kernel_deriv_prediction,sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,eps);

    if (isnan(dp_value) || !isfinite(dp_value)) {
      if (prediction_indexed != 0.0) {
	fprintf(stderr,"idpc: got dp_value NaN, sa=%g sm=%g pi=%g oi=%g eps=%g\n",sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,eps);

	if (fabs(sigma_additive) < 1e-20 || fabs(sigma_multiplicative) < 1e-20)  {
	  // off the meaningful domain... but sampler seems to get here sometimes during tuning
	  dp_value=0.0;
	} else {
	  
	  assert(prediction_indexed==0.0); 
	}
      }
      dp_value=0.0; // know this happens in the prediction_indexed==0.0 case and it's OK because the exponential from the convolution makes the derivative zero
    }

    dp[itercnt]=dp_value;
  }
}
