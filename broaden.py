"""
Routine to apply instumental, rotational, and macroturbulent broadening.
Script from  Michael Abdul Masih, Calum Hawcroft (2020).
Altered by Sarah Brands to account for convolving large wavelength
ranges as might be needed for usage i.c.w. FW v11 (Feb 2021).

Usage:
> python broaden.py -f <filename> -r <resolving power> -v <vrot> -m <vmacro>
Needs PyAstronomy on Lisa, Cartesius to install this:
> module load python
> pip install --user PyAstronomy
"""

# Suppress the standard matplotlib warning
import warnings
warnings.filterwarnings("ignore")

# Import needed modules
import sys
import argparse
import numpy as np
from math import sin, pi, ceil
from scipy.interpolate import interp1d
from scipy.special import erf
from scipy.signal import fftconvolve
from PyAstronomy.pyasl import rotBroad, instrBroadGaussFast

# Settings for rebinning
binSize = 0.01    # Size of wavelength bins resampled spectrum (Angstrom)
finalBins = 0.01  # Size of wavelength bins of broadened spectrum (Angstrom)

# Physical values
limbDark = 0.6    # Limb darkening coefficient (rotational broadening)
c = 299792.458    # Speed of light in km/s

# Settings for large wavelength intervals
# An interval of 100.0 A gives a precicion of about 0.001 % compared to
# using the exact appropiate kernel fore ach wavelength
range_fast = 100.  # Angstrom. Maximum width of range for using fast broadening
overlap = 20.0     # Extend the range to account for edges

def main():
    # Read in the needed values from the command line
    arguments = parseArguments()
    # Read in the spectrum
    try:
        #wlc, flux = np.loadtxt(arguments.fileName, unpack=True, skiprows=1)
        wlc, flux = np.genfromtxt(arguments.fileName).T
    except IOError as ioe:
        print(ioe, "Input spectrum " + arguments.fileName + " not found!")
        sys.exit()

    # Resample input spectrum to even wavelength bins of <binSize> Angstrom
    newWlc = np.arange(wlc[0]+binSize, wlc[-1]-binSize, binSize)
    flux = np.interp(newWlc, wlc, flux)
    wlc = newWlc

    # Check the total width of the wavelength interval
    # If this is smaller than a certain value, apply the fast broadening,
    # i.e. one kernel for all wavelengths
    # If the wavelength interval is large, then cut the interval into
    # pieces, use an appropiate interval for each piece, and stitch together.
    totalwidth = wlc[-1]-wlc[0]
    if totalwidth < range_fast:
        # Apply instrumental broadening
        flux = instrBroadGaussFast(wlc, flux, arguments.res,
            maxsig=5.0, edgeHandling="firstlast")
        # Apply rotational broadening
        flux = rotBroad(wlc, flux, limbDark, arguments.vrot)
        # Apply macroturbulent broadening
        if arguments.vmacro not in (-1, None):
            flux = macroBroad(wlc, flux, arguments.vmacro)
        # Resample to <finalBins> Angstrom
        if finalBins != binSize:
            newWlc = np.arange(wlc[0]+finalBins, wlc[-1]-finalBins, finalBins)
            flux = np.interp(newWlc, wlc, flux)
        convolved_flux_all = flux
        convolved_wave_all = newWlc
    else:
        nparts = int(ceil(totalwidth/range_fast))
        partwidth = totalwidth / nparts
        lenbins = len(wlc[wlc < wlc[0] + partwidth])
        partwidthext = partwidth + overlap
        lenbinsext = len(wlc[wlc < wlc[0] + partwidthext])
        extrabins = lenbinsext - lenbins

        idxlow_list = []
        idxhigh_list = []
        for i in range(nparts):
            idxlow = max(0, i*lenbins - extrabins)
            idxhigh = min(i*lenbins + lenbins + extrabins, len(wlc))

            idxlow_list.append(idxlow)
            idxhigh_list.append(idxhigh)

        convolved_flux_all = np.array([])
        convolved_wave_all = np.array([])

        for startarg, endarg, counter in zip(idxlow_list, idxhigh_list, range(nparts)):
            wlcpart = wlc[startarg:endarg]
            fluxpart = flux[startarg:endarg]

            fluxpart = instrBroadGaussFast(wlcpart, fluxpart, arguments.res,
                maxsig=5.0, edgeHandling="firstlast")

            # Apply rotational broadening
            fluxpart = rotBroad(wlcpart, fluxpart, limbDark, arguments.vrot)
            # Apply macroturbulent broadening
            if arguments.vmacro not in (-1, None):
                fluxpart = macroBroad(wlcpart, fluxpart, arguments.vmacro)
            # Resample to <finalBins> Angstrom
            if finalBins != binSize:
                newWlcpart = np.arange(wlcpart[0]+finalBins, wlcpart[-1]-finalBins, finalBins)
                fluxpart = np.interp(newWlcpart, wlcpart, fluxpart)
                wlcpart = newWlcpart

            if counter == 0:
                wlcpart = wlcpart[0:-extrabins]
                fluxpart = fluxpart[0:-extrabins]
            elif counter == nparts - 1:
                wlcpart = wlcpart[extrabins:]
                fluxpart = fluxpart[extrabins:]
            else:
                wlcpart = wlcpart[extrabins:-extrabins]
                fluxpart = fluxpart[extrabins:-extrabins]

            convolved_flux_all = np.concatenate((convolved_flux_all, fluxpart))
            convolved_wave_all = np.concatenate((convolved_wave_all, wlcpart))
    
    np.savetxt(arguments.fileName + ".fin", np.array([convolved_wave_all, convolved_flux_all]).T)

    exit()

def parseArguments():
    """
    Reads in the values from the command line.
    """
    parser = argparse.ArgumentParser(description="Applies all broadening.")
    parser.add_argument("-f", "--filename", type=str, dest="fileName", \
                        help="Filename of the input spectrum.")
    parser.add_argument("-r", "--resolution", type=float, dest="res", \
                        help="Resolving power.")
    parser.add_argument("-v", "--vrot", type=float, dest="vrot", \
                        help="Rotational velocity.")
    parser.add_argument("-m", "--vmacro", type=float, dest="vmacro", \
                        help="Macroturbulent velocity.")
    object = parser.parse_args()
    return object


def interpolate(newWlc, oldWlc, oldFlux):
    """
    Simple linear interpolation.
    Interpolates the flux to match the given new wavelength array.
    """
    f = interp1d(oldWlc, oldFlux)
    newFlux = f(newWlc)
    return newWlc, newFlux


def macroBroad(xdata, ydata, vmacro):
    """
    Edited broadening routine from http://dx.doi.org/10.5281/zenodo.10013
      This broadens the data by a given macroturbulent velocity.
    It works for small wavelength ranges. I need to make a better
    version that is accurate for large wavelength ranges! Sorry
    for the terrible variable names, it was copied from
    convol.pro in AnalyseBstar (Karolien Lefever)
    """
    # Make the kernel
    sq_pi = np.sqrt(np.pi)
    lambda0 = np.median(xdata)
    xspacing = xdata[1] - xdata[0]
    mr = vmacro * lambda0 / c
    ccr = 2 / (sq_pi * mr)

    px = np.arange(-len(xdata) / 2, len(xdata) / 2 + 1) * xspacing
    pxmr = abs(px) / mr
    profile = ccr * (np.exp(-pxmr ** 2) + sq_pi * pxmr * (erf(pxmr) - 1.0))

    # Extend the xy axes to avoid edge-effects
    before = ydata[int(-profile.size / 2 + 1):]
    after = ydata[:int(profile.size / 2)]
    extended = np.r_[before, ydata, after]

    first = xdata[0] - float(int(profile.size / 2.0 + 0.5)) * xspacing
    last = xdata[-1] + float(int(profile.size / 2.0 + 0.5)) * xspacing
    x2 = np.linspace(first, last, extended.size)

    conv_mode = "same"

    # Do the convolution
    newydata = fftconvolve(extended, profile / profile.sum(), mode=conv_mode)
    newydata = newydata[len(before):len(before)+len(ydata)]

    return newydata


if __name__ == "__main__":
    main()
