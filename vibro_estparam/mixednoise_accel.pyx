from cpython cimport PyObject
from cpython.ref cimport Py_INCREF
#from cpython.ceval cimport PyEval_InitThreads
cdef extern from "Python.h":
     void PyEval_InitThreads()
     pass
import numpy as np
 
cimport numpy as np

np.import_array()
PyEval_InitThreads() # Required because we will get callbacks from other threads that don't hold the GIL

cdef extern from "mixednoise_accel_ops.h":
    void integrate_lognormal_normal_convolution_c(PyObject *lognormal_normal_convolution_integral_y_zero_to_eps,PyObject *evaluation_cache,double sigma_additive,double sigma_multiplicative, double *prediction,double *observed,double *p,unsigned n) nogil
    void integrate_deriv_sigma_additive_c(PyObject *lognormal_normal_convolution_integral_y_zero_to_eps_deriv_sigma_additive,double sigma_additive,double sigma_multiplicative, double *prediction,double *observed,double *dp,unsigned n) nogil
    void integrate_deriv_sigma_multiplicative_c(PyObject *lognormal_normal_convolution_integral_y_zero_to_eps_deriv_sigma_multiplicative,double sigma_additive,double sigma_multiplicative, double *prediction,double *observed,double *dp,unsigned n) nogil
    void integrate_deriv_prediction_c(PyObject *lognormal_normal_convolution_integral_y_zero_to_eps_deriv_prediction,double sigma_additive,double sigma_multiplicative, double *prediction,double *observed,double *dp,unsigned n) nogil
    pass

cdef public double evaluate_y_zero_to_eps(PyObject *integral_y_zero_to_eps,double sigma_additive,double sigma_multiplicative,double prediction_indexed,double observed_indexed,double eps) with gil:
    cdef object integral_y_zero_to_eps_obj
    cdef double singular_portion
    
    integral_y_zero_to_eps_obj = <object> integral_y_zero_to_eps
    singular_portion = integral_y_zero_to_eps_obj(sigma_additive,sigma_multiplicative,prediction_indexed,observed_indexed,eps)
    
    return singular_portion

cdef public double cachelookup(PyObject *evaluation_cache,double sigma_additive,double sigma_multiplicative, double prediction_indexed,double observed_indexed) with gil:
    # NOTE: Must check to make sure that Cython creates PyGilState_Ensure() with the "with gil":
    cdef object cache
    cdef double p_value
    
    cache = <object>evaluation_cache
    
    key = (float(sigma_additive),float(sigma_multiplicative),float(prediction_indexed),float(observed_indexed))
    if not key in cache:
         
        p_value = np.nan
        pass   
    else: 
        p_value = cache[key] 
        pass
    return p_value
 
cdef public void cacheadd(PyObject *evaluation_cache,double sigma_additive,double sigma_multiplicative, double prediction_indexed,double observed_indexed,double p_value) with gil:
    # NOTE: Must check to make sure that Cython creates PyGilState_Ensure() with the "with gil":
    cdef object cache
     
    cache = <object>evaluation_cache
    
    key = (float(sigma_additive),float(sigma_multiplicative),float(prediction_indexed),float(observed_indexed))
    cache[key] = float(p_value)
    pass

 
def integrate_lognormal_normal_convolution(object lognormal_normal_convolution_integral_y_zero_to_eps,
                                           object evaluation_cache,
                                           double sigma_additive,double sigma_multiplicative,
                                           np.ndarray[np.float64_t,ndim=1,mode="c"] prediction,
                                           np.ndarray[np.float64_t,ndim=1,mode="c"] observed):
    
    cdef unsigned cnt
    cdef unsigned lim
    cdef unsigned key_in_cache
    cdef np.ndarray[np.float64_t,ndim=1,mode="c"] p
    cdef PyObject *lognormal_normal_convolution_integral_y_zero_to_eps_c
    cdef PyObject *evaluation_cache_c

    lognormal_normal_convolution_integral_y_zero_to_eps_c = <PyObject *>lognormal_normal_convolution_integral_y_zero_to_eps
    evaluation_cache_c = <PyObject *>evaluation_cache

    if evaluation_cache is None:
        evaluation_cache_c = <PyObject *>NULL
        pass
    
    lim=prediction.shape[0]
    if lim != observed.shape[0]:
        raise ValueError("prediction and observed must be of same shape (%d vs %d)" % (prediction.shape[0],observed.shape[0]))
 
    p=np.zeros((lim,),dtype='d')

    with nogil:
        integrate_lognormal_normal_convolution_c(lognormal_normal_convolution_integral_y_zero_to_eps_c,evaluation_cache_c,sigma_additive,sigma_multiplicative,<double *>prediction.data,<double *>observed.data,<double *>p.data,lim)
        pass
    
    return p


def integrate_deriv_sigma_additive(object lognormal_normal_convolution_integral_y_zero_to_eps_deriv_sigma_additive,
                                   double sigma_additive,double sigma_multiplicative,
                                   np.ndarray[np.float64_t,ndim=1,mode="c"] prediction,
                                   np.ndarray[np.float64_t,ndim=1,mode="c"] observed):
    
    cdef unsigned cnt
    cdef unsigned lim
    cdef np.ndarray[np.float64_t,ndim=1,mode="c"] dp
    cdef PyObject *lognormal_normal_convolution_integral_y_zero_to_eps_deriv_sigma_additive_c

    lognormal_normal_convolution_integral_y_zero_to_eps_deriv_sigma_additive_c = <PyObject *>lognormal_normal_convolution_integral_y_zero_to_eps_deriv_sigma_additive

    lim=prediction.shape[0]
    if lim != observed.shape[0]:
        raise ValueError("prediction and observed must be of same shape (%d vs %d)" % (prediction.shape[0],observed.shape[0]))

    dp=np.zeros((lim,),dtype='d')

    with nogil:
        integrate_deriv_sigma_additive_c(lognormal_normal_convolution_integral_y_zero_to_eps_deriv_sigma_additive_c,sigma_additive,sigma_multiplicative,<double *>prediction.data,<double *>observed.data,<double *>dp.data,lim)
        pass
    
    return dp


def integrate_deriv_sigma_multiplicative(object lognormal_normal_convolution_integral_y_zero_to_eps_deriv_sigma_multiplicative,
                                         double sigma_additive,double sigma_multiplicative,
                                         np.ndarray[np.float64_t,ndim=1,mode="c"] prediction,
                                         np.ndarray[np.float64_t,ndim=1,mode="c"] observed):
    
    cdef unsigned cnt
    cdef unsigned lim
    cdef np.ndarray[np.float64_t,ndim=1,mode="c"] dp
    cdef PyObject *lognormal_normal_convolution_integral_y_zero_to_eps_deriv_sigma_multiplicative_c

    lognormal_normal_convolution_integral_y_zero_to_eps_deriv_sigma_multiplicative_c = <PyObject *>lognormal_normal_convolution_integral_y_zero_to_eps_deriv_sigma_multiplicative

    lim=prediction.shape[0]
    if lim != observed.shape[0]:
        raise ValueError("prediction and observed must be of same shape (%d vs %d)" % (prediction.shape[0],observed.shape[0]))

    dp=np.zeros((lim,),dtype='d')

    with nogil:
        integrate_deriv_sigma_multiplicative_c(lognormal_normal_convolution_integral_y_zero_to_eps_deriv_sigma_multiplicative_c,sigma_additive,sigma_multiplicative,<double *>prediction.data,<double *>observed.data,<double *>dp.data,lim)
        pass
    
    return dp


def integrate_deriv_prediction(object lognormal_normal_convolution_integral_y_zero_to_eps_deriv_prediction,
                                         double sigma_additive,double sigma_multiplicative,
                                         np.ndarray[np.float64_t,ndim=1,mode="c"] prediction,
                                         np.ndarray[np.float64_t,ndim=1,mode="c"] observed):
    
    cdef unsigned cnt
    cdef unsigned lim
    cdef np.ndarray[np.float64_t,ndim=1,mode="c"] dp
    cdef PyObject *lognormal_normal_convolution_integral_y_zero_to_eps_deriv_prediction_c

    lognormal_normal_convolution_integral_y_zero_to_eps_deriv_prediction_c = <PyObject *>lognormal_normal_convolution_integral_y_zero_to_eps_deriv_prediction

    lim=prediction.shape[0]
    if lim != observed.shape[0]:
        raise ValueError("prediction and observed must be of same shape (%d vs %d)" % (prediction.shape[0],observed.shape[0]))

    dp=np.zeros((lim,),dtype='d')

    with nogil:
        integrate_deriv_prediction_c(lognormal_normal_convolution_integral_y_zero_to_eps_deriv_prediction_c,sigma_additive,sigma_multiplicative,<double *>prediction.data,<double *>observed.data,<double *>dp.data,lim)
        pass

    return dp
