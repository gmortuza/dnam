# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 08:44:55 2019

@author: WILLIAMCLAY

Code copied from Jungmann lab Picasso
"""

import matplotlib.pyplot as _plt
import numpy as _np
from numpy import fft as _fft
import lmfit as _lmfit
from tqdm import tqdm as _tqdm
from tqdm import trange as _trange
import os.path as _ospath
import yaml as _yaml
from scipy import interpolate as _interpolate

import numba as _numba
import h5py as _h5py

import gc

#_plt.style.use("ggplot")

_DRAW_MAX_SIGMA = 3

def xcorr(imageA, imageB):
    FimageA = _fft.fft2(imageA)
    CFimageB = _np.conj(_fft.fft2(imageB))
    return _fft.fftshift(
        _np.real(_fft.ifft2((FimageA * CFimageB)))
    ) / _np.sqrt(imageA.size)


def get_image_shift(imageA, imageB, box, roi=None, display=False):
    """ Computes the shift from imageA to imageB """
    #print(_np.sum(imageA))
    if (_np.sum(imageA) == 0) or (_np.sum(imageB) == 0):
        return 0, 0
    # Compute image correlation
    XCorr = xcorr(imageA, imageB)
    # Cut out center roi
    Y, X = imageA.shape
    if roi is not None:
        Y_ = int((Y - roi) / 2)
        X_ = int((X - roi) / 2)
        if Y_ > 0:
            XCorr = XCorr[Y_:-Y_, :]
        else:
            Y_ = 0
        if X_ > 0:
            XCorr = XCorr[:, X_:-X_]
        else:
            X_ = 0
    else:
        Y_ = X_ = 0
    # A quarter of the fit ROI
    fit_X = int(box / 2)
    # A coordinate grid for the fitting ROI
    y, x = _np.mgrid[-fit_X: fit_X + 1, -fit_X: fit_X + 1]
    # Find the brightest pixel and cut out the fit ROI
    y_max_, x_max_ = _np.unravel_index(XCorr.argmax(), XCorr.shape)
    FitROI = XCorr[
        y_max_ - fit_X: y_max_ + fit_X + 1,
        x_max_ - fit_X: x_max_ + fit_X + 1,
    ]

    # The fit model
    def flat_2d_gaussian(a, xc, yc, s, b):
        A = a * _np.exp(-0.5 * ((x - xc) ** 2 + (y - yc) ** 2) / s ** 2) + b
        return A.flatten()

    gaussian2d = _lmfit.Model(
        flat_2d_gaussian, name="2D Gaussian", independent_vars=[]
    )

    # Set up initial parameters and fit
    params = _lmfit.Parameters()
    params.add("a", value=FitROI.max(), vary=True, min=0)
    params.add("xc", value=0, vary=True)
    params.add("yc", value=0, vary=True)
    params.add("s", value=1, vary=True, min=0)
    params.add("b", value=FitROI.min(), vary=True, min=0)
    results = gaussian2d.fit(FitROI.flatten(), params)

    # Get maximum coordinates and add offsets
    xc = results.best_values["xc"]
    yc = results.best_values["yc"]
    xc += X_ + x_max_
    yc += Y_ + y_max_

    if display:
        _plt.figure(figsize=(17, 10))
        _plt.subplot(1, 3, 1)
        _plt.imshow(imageA, interpolation="none")
        _plt.subplot(1, 3, 2)
        _plt.imshow(imageB, interpolation="none")
        _plt.subplot(1, 3, 3)
        _plt.imshow(XCorr, interpolation="none")
        _plt.plot(xc, yc, "x")
        _plt.show()

    xc -= _np.floor(X / 2)
    yc -= _np.floor(Y / 2)
    return -yc, -xc


def rcc(segments, max_shift=None, callback=None):
    n_segments = len(segments)
    shifts_x = _np.zeros((n_segments, n_segments))
    shifts_y = _np.zeros((n_segments, n_segments))
    n_pairs = int(n_segments * (n_segments - 1) / 2)
    flag = 0
    with _tqdm(
        total=n_pairs, desc="Correlating image pairs", unit="pairs"
    ) as progress_bar:
        if callback is not None:
            callback(0)
        for i in range(n_segments - 1):
            for j in range(i + 1, n_segments):
                progress_bar.update()
                shifts_y[i, j], shifts_x[i, j] = get_image_shift(
                    segments[i], segments[j], 5, max_shift
                )
                flag += 1
                if callback is not None:
                    callback(flag)
    return minimize_shifts(shifts_x, shifts_y)

def minimize_shifts(shifts_x, shifts_y, shifts_z=None):
    n_channels = shifts_x.shape[0]
    n_pairs = int(n_channels * (n_channels - 1) / 2)
    n_dims = 2 if shifts_z is None else 3
    rij = _np.zeros((n_pairs, n_dims))
    A = _np.zeros((n_pairs, n_channels - 1))
    flag = 0
    for i in range(n_channels - 1):
        for j in range(i + 1, n_channels):
            rij[flag, 0] = shifts_y[i, j]
            rij[flag, 1] = shifts_x[i, j]
            if n_dims == 3:
                rij[flag, 2] = shifts_z[i, j]
            A[flag, i:j] = 1
            flag += 1
    Dj = _np.dot(_np.linalg.pinv(A), rij)
    shift_y = _np.insert(_np.cumsum(Dj[:, 0]), 0, 0)
    shift_x = _np.insert(_np.cumsum(Dj[:, 1]), 0, 0)
    if n_dims == 2:
        return shift_y, shift_x
    else:
        shift_z = _np.insert(_np.cumsum(Dj[:, 2]), 0, 0)
        return shift_y, shift_x, shift_z
    
def segment(locs, info, segmentation, kwargs={}, callback=None):
    Y = info[0]["Height"]
    X = info[0]["Width"]
    n_frames = info[0]["Frames"]
    n_seg = n_segments(info, segmentation)
    bounds = _np.linspace(0, n_frames - 1, n_seg + 1, dtype=_np.uint32)
    segments = _np.zeros((n_seg, Y, X))
    if callback is not None:
        callback(0)
    for i in _trange(n_seg, desc="Generating segments", unit="segments"):
        #print("Segment",i)
        segment_locs = locs[
            (locs.frame >= bounds[i]) & (locs.frame < bounds[i + 1])
        ]
        _, segments[i] = render(segment_locs, info, **kwargs)
        if callback is not None:
            callback(i + 1)
    return bounds, segments


def n_segments(info, segmentation):
    n_frames = info[0]["Frames"]
    return int(_np.round(n_frames / segmentation))

def render(
    locs,
    info=None,
    oversampling=1,
    viewport=None,
    blur_method=None,
    min_blur_width=0,
):
    if viewport is None:
        try:
            viewport = [(0, 0), (info[0]["Height"], info[0]["Width"])]
        except TypeError:
            raise ValueError("Need info if no viewport is provided.")
    (y_min, x_min), (y_max, x_max) = viewport
    
    if blur_method == "gaussian":
        return render_gaussian(
            locs, oversampling, y_min, x_min, y_max, x_max, min_blur_width
        )

    else:
        raise Exception("blur_method not understood.")

@_numba.jit(nopython=True, nogil=True)
def render_gaussian(
    locs, oversampling, y_min, x_min, y_max, x_max, min_blur_width
):
    image, n_pixel_y, n_pixel_x, x, y, in_view = _render_setup(
        locs, oversampling, y_min, x_min, y_max, x_max
    )
    blur_width = oversampling * _np.maximum(locs.lpx, min_blur_width)
    blur_height = oversampling * _np.maximum(locs.lpy, min_blur_width)
    sy = blur_height[in_view]
    sx = blur_width[in_view]
    for x_, y_, sx_, sy_ in zip(x, y, sx, sy):
        max_y = _DRAW_MAX_SIGMA * sy_
        i_min = _np.int32(y_ - max_y)
        if i_min < 0:
            i_min = 0
        i_max = _np.int32(y_ + max_y + 1)
        if i_max > n_pixel_y:
            i_max = n_pixel_y
        max_x = _DRAW_MAX_SIGMA * sx_
        j_min = _np.int32(x_ - max_x)
        if j_min < 0:
            j_min = 0
        j_max = _np.int32(x_ + max_x) + 1
        if j_max > n_pixel_x:
            j_max = n_pixel_x
        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                image[i, j] += _np.exp(
                    -(
                        (j - x_ + 0.5) ** 2 / (2 * sx_ ** 2)
                        + (i - y_ + 0.5) ** 2 / (2 * sy_ ** 2)
                    )
                ) / (2 * _np.pi * sx_ * sy_)
    return len(x), image

@_numba.jit(nopython=True, nogil=True)
def _render_setup(locs, oversampling, y_min, x_min, y_max, x_max):
    n_pixel_y = int(_np.ceil(oversampling * (y_max - y_min)))
    n_pixel_x = int(_np.ceil(oversampling * (x_max - x_min)))
    x = locs.x
    y = locs.y
    in_view = (x > x_min) & (y > y_min) & (x < x_max) & (y < y_max)
    x = x[in_view]
    y = y[in_view]
    x = oversampling * (x - x_min)
    y = oversampling * (y - y_min)
    image = _np.zeros((n_pixel_y, n_pixel_x), dtype=_np.float32)
    return image, n_pixel_y, n_pixel_x, x, y, in_view

def load_info(path, qt_parent=None):
    path_base, path_extension = _ospath.splitext(path)
    filename = path_base + ".yaml"

    with open(filename, "r") as info_file:
        info = list(_yaml.load_all(info_file, Loader = _yaml.SafeLoader))

    return info

def undrift(
    locs,
    info,
    segmentation,
    display=False,
    segmentation_callback=None,
    rcc_callback=None,
):
    bounds, segments = segment(
        locs,
        info,
        segmentation,
        {"blur_method": "gaussian", "min_blur_width": 1},
        segmentation_callback,
    )
    shift_y, shift_x = rcc(segments, 32, rcc_callback)
    t = (bounds[1:] + bounds[:-1]) / 2
    drift_x_pol = _interpolate.InterpolatedUnivariateSpline(t, shift_x, k=3)
    drift_y_pol = _interpolate.InterpolatedUnivariateSpline(t, shift_y, k=3)
    t_inter = _np.arange(info[0]["Frames"])
    drift = (drift_x_pol(t_inter), drift_y_pol(t_inter))
    drift = _np.rec.array(drift, dtype=[("x", "f"), ("y", "f")])
    
    locs.x -= drift.x[locs.frame]
    locs.y -= drift.y[locs.frame]
    return drift, locs

def load_locs(path, qt_parent=None):
    with _h5py.File(path, "r") as locs_file:
        locs = locs_file["locs"][...]
    locs = _np.rec.array(
        locs, dtype=locs.dtype
    )  # Convert to rec array with fields as attributes
    info = load_info(path, qt_parent=qt_parent)
    return locs, info


def save_locs(path, locs, info):
    locs = ensure_sanity(locs, info)
    with _h5py.File(path, "w") as locs_file:
        locs_file.create_dataset("locs", data=locs)
    base, ext = _ospath.splitext(path)
    info_path = base + ".yaml"
    save_info(info_path, info)
    
def ensure_sanity(locs, info):
    # no inf or nan:
    locs = locs[
        _np.all(
            _np.array([_np.isfinite(locs[_]) for _ in locs.dtype.names]),
            axis=0,
        )
    ]
    # other sanity checks:
    locs = locs[locs.x > 0]
    locs = locs[locs.y > 0]
    locs = locs[locs.x < info[0]["Width"]]
    locs = locs[locs.y < info[0]["Height"]]
    locs = locs[locs.lpx > 0]
    locs = locs[locs.lpy > 0]
    return locs

def save_info(path, info, default_flow_style=False):
    with open(path, "w") as file:
        _yaml.dump_all(info, file, default_flow_style=default_flow_style)
        
def undrift_from_picked(locs,picked_locs,info):


    drift_x = _undrift_from_picked_coordinate_light(
        picked_locs, "x",info
    )
    drift_y = _undrift_from_picked_coordinate_light(
        picked_locs, "y",info
    )

    # Apply drift
    locs.x -= drift_x[locs.frame]
    locs.y -= drift_y[locs.frame]

    # A rec array to store the applied drift
    drift = (drift_x, drift_y)
    drift = _np.rec.array(drift, dtype=[("x", "f"), ("y", "f")])

    return drift, locs

def _undrift_from_picked_coordinate_light(
    picked_locs, coordinate, info
):
    """Should be identical to _undrift_from_picked_coordinate but with lower
    memory usage."""
    n_picks = len(picked_locs)
    n_frames = info[0]["Frames"]

    # Drift per pick per frame
    #drift = _np.empty((n_picks, n_frames))
    #drift.fill(_np.nan)
    
    drift = _np.empty(n_picks)
    drift.fill(_np.nan)
    
    drift_mean = _np.empty(n_frames)
    drift_mean.fill(_np.nan)
    
    msd = _np.empty(n_picks)
    msd.fill(_np.nan)
    
    drift_counts = _np.zeros(n_frames)

    # Remove center of mass offset, compute mean drift
    for i, locs in enumerate(picked_locs):
        drift_temp = _np.empty(n_frames)
        drift_temp.fill(_np.nan)
        coordinates = getattr(locs, coordinate)
        drift_temp[locs.frame] = coordinates - _np.mean(coordinates)
        drift_counts[~_np.isnan(drift_temp)] += 1.0
        drift_mean[(~_np.isnan(drift_temp)) & (~_np.isnan(drift_mean))] += drift_temp[(~_np.isnan(drift_temp)) & (~_np.isnan(drift_mean))]
        drift_mean[(~_np.isnan(drift_temp)) & (_np.isnan(drift_mean))] = drift_temp[(~_np.isnan(drift_temp)) & (_np.isnan(drift_mean))]

    drift_mean = drift_mean/drift_counts    
    
    #compute msd
    for i, locs in enumerate(picked_locs):
        drift_temp = _np.empty(n_frames)
        drift_temp.fill(_np.nan)
        coordinates = getattr(locs, coordinate)
        drift_temp[locs.frame] = coordinates - _np.mean(coordinates)
        sd_temp = (drift_temp - drift_mean)**2
        msd[i] = _np.nanmean(sd_temp)
        
        
    drift_mean = _np.empty(n_frames)
    drift_mean.fill(_np.nan)
    #Compute weighted mean drift
    for i, locs in enumerate(picked_locs):
        drift_temp = _np.empty(n_frames)
        drift_temp.fill(_np.nan)
        coordinates = getattr(locs, coordinate)
        drift_temp[locs.frame] = coordinates - _np.mean(coordinates)
        drift_temp /= msd[i]
        drift_counts[(~_np.isnan(drift_temp))] += 1.0/msd[i]
        drift_mean[(~_np.isnan(drift_temp)) & (~_np.isnan(drift_mean))] += drift_temp[(~ _np.isnan(drift_temp)) & (~_np.isnan(drift_mean))]
        drift_mean[(~_np.isnan(drift_temp)) & (_np.isnan(drift_mean))] = drift_temp[(~_np.isnan(drift_temp)) & (_np.isnan(drift_mean))]

    drift_mean = drift_mean/drift_counts   

    # Mean drift over picks
    #drift_mean = _np.nanmean(drift, 0)
    # Square deviation of each pick's drift to mean drift along frames
    #sd = (drift - drift_mean) ** 2
    # Mean of square deviation for each pick
    #msd = _np.nanmean(sd, 1)
    # New mean drift over picks
    # where each pick is weighted according to its msd
    #nan_mask = _np.isnan(drift)
    #drift = _np.ma.MaskedArray(drift, mask=nan_mask)
    #drift_mean = _np.ma.average(drift, axis=0, weights=1 / msd)
    #drift_mean = drift_mean.filled(_np.nan)

    # Linear interpolation for frames without localizations
    def nan_helper(y):
        return _np.isnan(y), lambda z: z.nonzero()[0]

    nans, nonzero = nan_helper(drift_mean)
    drift_mean[nans] = _np.interp(
        nonzero(nans), nonzero(~nans), drift_mean[~nans]
    )

    return drift_mean

def _undrift_from_picked_coordinate(
    picked_locs, coordinate, info
):
    n_picks = len(picked_locs)
    n_frames = info[0]["Frames"]

    # Drift per pick per frame
    drift = _np.empty((n_picks, n_frames))
    drift.fill(_np.nan)

    # Remove center of mass offset
    for i, locs in enumerate(picked_locs):
        coordinates = getattr(locs, coordinate)
        drift[i, locs.frame] = coordinates - _np.mean(coordinates)

    # Mean drift over picks
    drift_mean = _np.nanmean(drift, 0)
    # Square deviation of each pick's drift to mean drift along frames
    sd = (drift - drift_mean) ** 2
    # Mean of square deviation for each pick
    msd = _np.nanmean(sd, 1)
    # New mean drift over picks
    # where each pick is weighted according to its msd
    nan_mask = _np.isnan(drift)
    drift = _np.ma.MaskedArray(drift, mask=nan_mask)
    drift_mean = _np.ma.average(drift, axis=0, weights=1 / msd)
    drift_mean = drift_mean.filled(_np.nan)

    # Linear interpolation for frames without localizations
    def nan_helper(y):
        return _np.isnan(y), lambda z: z.nonzero()[0]

    nans, nonzero = nan_helper(drift_mean)
    drift_mean[nans] = _np.interp(
        nonzero(nans), nonzero(~nans), drift_mean[~nans]
    )

    return drift_mean

def csv2locs(path, pixelsize):  
    print("Converting {}".format(path))

    data = _np.genfromtxt(path, dtype=float, delimiter=",", names=True)

    try:
        frames = data["frame"].astype(int)
        # make sure frames start at zero:
        frames = frames - _np.min(frames)
        x = data["x_nm"] / pixelsize
        y = data["y_nm"] / pixelsize
        photons = data["intensity_photon"].astype(int)

        bg = data["offset_photon"].astype(int)
        if "uncertainty_xy_nm" in data.dtype.names:
            lpx = data["uncertainty_xy_nm"] / pixelsize
            lpy = data["uncertainty_xy_nm"] / pixelsize
        else:
            lpx = data["uncertainty_nm"] / pixelsize
            lpy = data["uncertainty_nm"] / pixelsize

        if "z_nm" in data.dtype.names:
            z = data["z_nm"] / pixelsize
            sx = data["sigma1_nm"] / pixelsize
            sy = data["sigma2_nm"] / pixelsize
            
            del data
            gc.collect()

            LOCS_DTYPE = [
                ("frame", "u4"),
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("photons", "f4"),
                ("sx", "f4"),
                ("sy", "f4"),
                ("bg", "f4"),
                ("lpx", "f4"),
                ("lpy", "f4"),
            ]

            locs = _np.rec.array(
                (frames, x, y, z, photons, sx, sy, bg, lpx, lpy),
                dtype=LOCS_DTYPE,
            )

        else:
            sx = data["sigma_nm"] / pixelsize
            sy = data["sigma_nm"] / pixelsize
            
            del data
            gc.collect()

            LOCS_DTYPE = [
                ("frame", "u4"),
                ("x", "f4"),
                ("y", "f4"),
                ("photons", "f4"),
                ("sx", "f4"),
                ("sy", "f4"),
                ("bg", "f4"),
                ("lpx", "f4"),
                ("lpy", "f4"),
            ]

            locs = _np.rec.array(
                (frames, x, y, photons, sx, sy, bg, lpx, lpy),
                dtype=LOCS_DTYPE,
            )

        locs.sort(kind="mergesort", order="frame")

        img_info = {}
        img_info["Generated by"] = "picasso_utils csv2locs"
        img_info["Frames"] = int(_np.max(frames)) + 1
        img_info["Height"] = int(_np.ceil(_np.max(y)))
        img_info["Width"] = int(_np.ceil(_np.max(x)))

        info = []
        info.append(img_info)

        return locs, info
    except Exception as e:
        print(e)
        print("Error. Datatype not understood.")
        
def txt2locs(path):  
    """Reads DaoStorm .txt format"""
    print("Converting {}".format(path))

    data = _np.genfromtxt(path, dtype=float, delimiter=",", names=True)
    data = data[data["background"] > 0]

    if "frame" in data.dtype.names:
        frames = data["frame"].astype(int)
    else:
        frames = data["frame_number"].astype(int)
    # make sure frames start at zero:
    frames = frames - _np.min(frames)
    x = data["x"]
    y = data["y"]
    photons = data["sum"]
    photons[photons <= 0] = 1

    bg = data["background"]


    sx = data["xsigma"]
    sy = data["xsigma"]
    
    error = data["error"]
    
    if "track_length" in data.dtype.names:
        trackLength = data["track_length"]
    else:
        trackLength = 1
        
    del data
    gc.collect()
        
    #corrects bug in frame tracking for current version of DaoStorm. May need to be changed if DaoStorm fixed
    sx /= trackLength
    sy /= trackLength
    bg /= trackLength
    error /= trackLength
    
    #bkgstd = _np.sqrt(error+bg)
    
    #lpx = _np.sqrt((2*sx**2+1/12)/photons+8*_np.pi*sx**4*bkgstd**2/photons**2)
    lpx = _np.sqrt((2*sx**2+1/12)/photons+8*_np.pi*sx**4*bg/photons**2)
    lpy = lpx
    
    #del bkgstd
    del error
    gc.collect()

    LOCS_DTYPE = [
        ("frame", "u4"),
        ("x", "f4"),
        ("y", "f4"),
        ("photons", "f4"),
        ("sx", "f4"),
        ("sy", "f4"),
        ("bg", "f4"),
        ("lpx", "f4"),
        ("lpy", "f4"),
    ]

    locs = _np.rec.array(
        (frames, x, y, photons, sx, sy, bg, lpx, lpy),
        dtype=LOCS_DTYPE,
    )

    locs.sort(kind="mergesort", order="frame")

    img_info = {}
    img_info["Generated by"] = "picasso_utils txt2locs"
    img_info["Frames"] = int(_np.max(frames)) + 1
    img_info["Height"] = int(_np.ceil(_np.max(y)))
    img_info["Width"] = int(_np.ceil(_np.max(x)))

    info = []
    info.append(img_info)

    return locs, info

def load_salocs(path):
    import storm_analysis.sa_library.sa_h5py as sa_h5py
    salocs = sa_h5py.SAH5Py(filename=path)
    #maxFrame = salocs.getMovieLength()
    nLocs = salocs.getNLocalizations()
    
    LOCS_DTYPE = [
        ("frame", "u4"),
        ("x", "f4"),
        ("y", "f4"),
        ("photons", "f4"),
        ("sx", "f4"),
        ("sy", "f4"),
        ("bg", "f4"),
        ("lpx", "f4"),
        ("lpy", "f4"),
    ]

    zc = _np.empty((nLocs,))
    zc.fill(_np.nan)
    locs = _np.rec.array(
        (zc,zc,zc,zc,zc,zc,zc,zc,zc),
        dtype=LOCS_DTYPE,
    )
    
    locPtr = 0
    
    for frameId,frameLocs in salocs.localizationsIterator():
        #Filter out negative background localizations
        idx = frameLocs["background"] > 0
        x = frameLocs["x"][idx]
        y = frameLocs["y"][idx]
        photons = frameLocs["sum"][idx]
        photons[photons <= 0] = 1
        
        frames = frameId*_np.ones(x.shape)
        frames = frames.astype(int)
        
        nFrameLocs = len(frames)
    
        bg = frameLocs["background"][idx]
    
    
        sx = frameLocs["xsigma"][idx]
        sy = frameLocs["xsigma"][idx]
        
        error = frameLocs["error"][idx]
        
        if "track_length" in frameLocs:
            trackLength = frameLocs["track_length"][idx]
        else:
            trackLength = 1
            
        #corrects bug in frame tracking for current version of DaoStorm. May need to be changed if DaoStorm fixed
        sx /= trackLength
        sy /= trackLength
        bg /= trackLength
        error /= trackLength
        
        #bkgstd = _np.sqrt(error+bg)
        
        #lpx = _np.sqrt((2*sx**2+1/12)/photons+8*_np.pi*sx**4*bkgstd**2/photons**2)
        lpx = _np.sqrt((2*sx**2+1/12)/photons+8*_np.pi*sx**4*bg/photons**2)
        lpy = lpx
        
        frameTable = _np.rec.array(
            (frames, x, y, photons, sx, sy, bg, lpx, lpy),
            dtype=LOCS_DTYPE,
        )
        
        #nFrameLocs = len(frameTable)
        
# =============================================================================
#         if _np.isnan(_np.average(frameTable['lpx'])):
#             idx = [_np.isnan(frameTable['lpx'])]
#             print("!",frameId,frameTable[tuple(idx)])
#         if min(frameTable['bg']) < 0:
#             idx = [(frameTable['bg']<0)]
#             print(frameId,frameTable[tuple(idx)])
# =============================================================================
        
        locs[locPtr:(locPtr+nFrameLocs)] = frameTable
        
        locPtr+=nFrameLocs
         
    #Remove unused rows in pre-allocated table
    locs = locs[~_np.isnan(locs['lpx'])]
    locs.sort(kind="mergesort", order="frame")

    img_info = {}
    img_info["Generated by"] = "picasso_utils load_salocs"
    img_info["Frames"] = int(_np.max(frames)) + 1
    img_info["Height"] = int(_np.ceil(_np.max(locs['y'])))
    img_info["Width"] = int(_np.ceil(_np.max(locs['x'])))

    info = []
    info.append(img_info)

    return locs, info