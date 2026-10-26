"""Positive joint Gaussian-mixture strip extraction, independent of RNG/FFT.

The source mixture is normalized over the full image plane. Width is conditioned
on a finite spatial strip, while its finite dispersion-window mass is retained.
PSF/LSF roles are explicit hypotheses, not an instrument calibration.
"""
import math
from numbers import Integral, Real
import warnings

import numpy as np
from scipy.integrate import quad_vec
from scipy.special import erf, ndtr


class StripConvergenceError(RuntimeError):
    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = {} if diagnostics is None else diagnostics


def _real(value, name, positive=True):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real):
        raise ValueError(name+' must be a finite real number, excluding booleans')
    value=float(value)
    if not math.isfinite(value) or (value<=0 if positive else value<0):
        raise ValueError(name+' must be finite and '+('positive' if positive else 'nonnegative'))
    return value


def _integer(value,name,minimum,maximum=None):
    if (isinstance(value,(bool,np.bool_)) or not isinstance(value,Integral) or value<minimum or
            (maximum is not None and value>maximum)):
        raise ValueError(name+' is outside its integer range')
    return int(value)


def _positive_vector(value,name):
    if np.ma.isMaskedArray(value) or hasattr(value,'unit'):
        raise ValueError(name+' must be plain unmasked values')
    raw=np.asarray(value)
    if raw.ndim!=1 or raw.size==0 or raw.dtype.kind not in 'fiu':
        raise ValueError(name+' must be a nonempty real vector')
    if any(isinstance(v,(bool,np.bool_)) for v in np.asarray(value,dtype=object)):
        raise ValueError(name+' must exclude booleans')
    result=np.array(raw,dtype=float,copy=True)
    if np.any(~np.isfinite(result)) or np.any(result<=0):
        raise ValueError(name+' must be finite and positive')
    return result


def _normal_interval(lower,upper):
    """Positive Gaussian interval mass; central erf avoids narrow-bin loss."""
    lo,hi=np.broadcast_arrays(np.asarray(lower,dtype=float),np.asarray(upper,dtype=float))
    result=np.empty(lo.shape,dtype=float)
    crossing=(lo<0)&(hi>0)
    result[crossing]=.5*(erf(hi[crossing]/math.sqrt(2))-erf(lo[crossing]/math.sqrt(2)))
    same=~crossing
    a=np.where(hi[same]<=0,-hi[same],lo[same])
    b=np.where(hi[same]<=0,-lo[same],hi[same])
    mass=np.empty(a.shape,dtype=float);central=b<=1.
    mass[central]=.5*(erf(b[central]/math.sqrt(2))-erf(a[central]/math.sqrt(2)))
    mass[~central]=ndtr(-a[~central])-ndtr(-b[~central])
    result[same]=mass
    return result


def mge_strip_kernel(sigma_over_re,weights,re_pixel,pa_deg,axis_ratio,
                     psf_fwhm_arcsec,pixel_scale_arcsec,spatial_pixels,kernel_radius_pixels,
                     lsf_convention,*,absolute_tolerance=1.e-9,max_evaluations=100000,
                     apply_morphology=True):
    """Return width/height probabilities for a joint, centered spatial strip.

    sigma_over_re/weights are positive component vectors. Re [pixel] is the
    circularized half-light radius used to scale those source Gaussian sigmas;
    q=b/a is an area-preserving shear and PA is image-x degrees. FWHM [arcsec]
    and pixel scale [arcsec/pixel] define a circular Gaussian imaging PSF.

    point_source adds that PSF variance in y only: the upstream dispersion LSF
    already includes its x contribution. pre_psf adds it in both coordinates.
    Neither convention subtracts a PSF or certifies an actual instrumental LSF.
    Height is absolute full-source y-bin mass. Width is P(xbin,ystrip)/F where
    F=P(ystrip) for the entire normalized mixture, and is NOT window-normalized.
    Turning morphology off returns width=[1] while preserving F and height.

    Correlated components use positive conditional-Gaussian quadrature. Its
    error and optional standardized-y tail omission are in conditional units;
    neither includes source-fit error nor is a certified numerical error bound.
    Invalid inputs raise ValueError; warnings, invalid results, nonconvergence
    or a global evaluation-budget failure raise StripConvergenceError.
    """
    sigmas=_positive_vector(sigma_over_re,'sigma_over_re')
    supplied_weights=_positive_vector(weights,'weights')
    if sigmas.shape!=supplied_weights.shape:raise ValueError('Component vectors must have equal length')
    radius=_real(re_pixel,'re_pixel');q=_real(axis_ratio,'axis_ratio')
    if q>1:raise ValueError('axis_ratio must lie in (0,1]')
    if isinstance(pa_deg,(bool,np.bool_)) or not isinstance(pa_deg,Real) or not math.isfinite(float(pa_deg)):
        raise ValueError('pa_deg must be a finite real angle')
    angle=float(pa_deg)%180.
    if angle==0.:cosine,sine=1.,0.
    elif angle==90.:cosine,sine=0.,1.
    else:cosine,sine=math.cos(math.radians(angle)),math.sin(math.radians(angle))
    fwhm=_real(psf_fwhm_arcsec,'psf_fwhm_arcsec',positive=False)
    pixel=_real(pixel_scale_arcsec,'pixel_scale_arcsec')
    count=_integer(spatial_pixels,'spatial_pixels',1,1001)
    half=_integer(kernel_radius_pixels,'kernel_radius_pixels',0,1024)
    tolerance=_real(absolute_tolerance,'absolute_tolerance')
    maximum=_integer(max_evaluations,'max_evaluations',1)
    if lsf_convention not in ('point_source','pre_psf'):raise ValueError('An explicit supported lsf_convention is required')
    if not isinstance(apply_morphology,(bool,np.bool_)):raise ValueError('apply_morphology must be boolean')
    with np.errstate(over='ignore',under='ignore',invalid='ignore'):
        total=float(np.sum(supplied_weights,dtype=float))
        w=supplied_weights/total
    if not math.isfinite(total) or total<=0 or np.any(~np.isfinite(w)) or np.any(w<=0):
        raise ValueError('Full-source weight normalization is not representable')
    p=fwhm/pixel/math.sqrt(8.*math.log(2.));p2=p*p
    if not math.isfinite(p2) or (fwhm>0 and p2==0):raise ValueError('PSF variance is not representable')
    dx=p2 if lsf_convention=='pre_psf' else 0.;dy=p2
    xedges=np.arange(-half,half+2,dtype=float)-.5
    yedges=np.arange(count+1,dtype=float)-count/2.
    components=[];fy=[];oy=[]
    rootq=math.sqrt(q)
    for scaled in sigmas:
        source_sigma=float(scaled)*radius
        major=source_sigma/rootq;minor=source_sigma*rootq
        mc,ms,bc,bs=major*cosine,major*sine,minor*cosine,minor*sine
        source_x=mc*mc+bs*bs;source_y=ms*ms+bc*bc
        covariance=mc*ms-bs*bc;var_y=source_y+dy;source_variance=source_sigma*source_sigma
        if not all(math.isfinite(v) for v in (source_x,source_y,covariance,var_y,source_variance)) or min(var_y,source_variance)<=0:
            raise ValueError('Derived source covariance is not representable')
        sy=math.sqrt(var_y)
        base=source_variance/sy
        conditional_variance=base*base+dx*(source_y/var_y)+source_x*(dy/var_y)+dx*(dy/var_y)
        if not math.isfinite(conditional_variance) or conditional_variance<=0:
            raise ValueError('Positive determinant conditional variance is not representable')
        fraction=float(_normal_interval(-count/(2.*sy),count/(2.*sy)))
        outside=2.*float(ndtr(-count/(2.*sy)))
        components.append((sy,covariance/sy,math.sqrt(conditional_variance),covariance,source_x+dx))
        fy.append(fraction);oy.append(outside)
    inside=math.fsum(float(a)*b for a,b in zip(w,fy))
    outside=math.fsum(float(a)*b for a,b in zip(w,oy))
    fraction=(1.-outside) if inside>.5 else inside
    diagnostics={'kernel_applied':bool(apply_morphology),'lsf_convention':lsf_convention,
        'covariance_role':('source plus spatial Gaussian PSF only' if lsf_convention=='point_source' else
                           'source plus Gaussian PSF in both dispersion and spatial coordinates'),
        'source_weight_sum_input':total,'source_weight_sum_normalized':float(w.sum()),
        'absolute_tolerance':tolerance,'max_evaluations':maximum,'evaluations':0,
        'standardized_y_truncation':12.,'separable_components':0,'quadrature_components':0,
        'quadrature_error_estimate':0.,'omitted_probability_bound':0.,
        'quadrature_error_norm':'max-bin error estimate; component estimates summed in conditional units',
        'error_interpretation':'conditional quadrature estimates, not certified bounds; source-fit error excluded',
        'assumptions':['centered strip and source; orthogonal x integral is infinite for height',
                       'positive full-source mixture normalized before joint strip conditioning',
                       'PSF/LSF convention is an explicit adopted hypothesis, not instrument calibration']}
    def fail(message):raise StripConvergenceError(message,diagnostics)
    if not math.isfinite(fraction) or not 0.<fraction<=1.:fail('Invalid or unrepresentable spatial fraction')
    height=np.zeros(count,dtype=float)
    for weight,(sy,_,_,_,_) in zip(w,components):
        height+=weight*_normal_interval(yedges[:-1]/sy,yedges[1:]/sy)
    width=np.zeros(2*half+1,dtype=float);error_total=0.;omitted=0.
    if apply_morphology:
        for weight,fi,(sy,beta,conditional_sigma,covariance,var_x) in zip(w,fy,components):
            if covariance==0.:
                diagnostics['separable_components']+=1
                sx=math.sqrt(var_x)
                width+=(float(weight)*fi/fraction)*_normal_interval(xedges[:-1]/sx,xedges[1:]/sx)
                continue
            diagnostics['quadrature_components']+=1
            bound=count/(2.*sy);left,right=-min(bound,12.),min(bound,12.)
            if bound>12.:omitted+=(float(weight)/fraction)*2.*float(ndtr(-12.))
            points=[]
            if conditional_sigma/abs(beta)<.25*(right-left):
                transition=conditional_sigma/beta
                for edge in xedges:
                    center=float(edge)/beta
                    for point in (center-transition,center,center+transition):
                        if left<point<right:points.append(point)
            def integrand(z):
                if diagnostics['evaluations']>=maximum:fail('Strip quadrature evaluation budget exceeded')
                diagnostics['evaluations']+=1
                mean=beta*z
                return (float(weight)/fraction/math.sqrt(2.*math.pi)*math.exp(-.5*z*z))*_normal_interval(
                    (xedges[:-1]-mean)/conditional_sigma,(xedges[1:]-mean)/conditional_sigma)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                try:
                    value,error,info=quad_vec(integrand,left,right,epsabs=tolerance/(4.*len(w)),epsrel=0.,
                        norm='max',limit=min(maximum,4096),points=sorted(set(points)),workers=1,full_output=True)
                except StripConvergenceError:raise
                except Exception as problem:fail('Strip quadrature failed: '+str(problem))
            if caught:
                diagnostics['warnings']=[str(item.message) for item in caught]
                fail('Strip quadrature emitted a warning')
            if not info.success or not math.isfinite(error) or error<0:
                fail('Strip quadrature did not converge')
            width+=value;error_total+=float(error)
    else:width=np.ones(1,dtype=float)
    diagnostics['quadrature_error_estimate']=error_total
    diagnostics['omitted_probability_bound']=omitted
    budget=len(width)*error_total+omitted+64.*np.finfo(float).eps*(len(w)+count+len(width))
    if error_total>tolerance:fail('Combined conditional quadrature error exceeds tolerance')
    if np.any(~np.isfinite(height)) or np.any(height<0.) or abs(float(height.sum())-fraction)>budget:
        fail('Spatial probability accounting failed')
    mass=float(width.sum())
    if np.any(~np.isfinite(width)) or np.any(width<0.) or np.any(width>1.+budget) or mass>1.+budget:
        fail('Conditional finite kernel probability is invalid; no clipping applied')
    diagnostics.update(kernel_window_fraction=mass,kernel_absolute_fraction=fraction*mass,
        kernel_n_bins=int(len(width)),accounting={'height_sum':float(height.sum()),
            'source_fraction':fraction,'numerical_consistency_tolerance':budget,
            'interpretation':'roundoff/accounting tolerance, not a scientific error bound'})
    return {'width':width,'height':height,'source_fraction':fraction,'spatial_pixels':count,'diagnostics':diagnostics}
