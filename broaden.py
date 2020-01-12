"""
Routine to apply instumental, rotational, and macroturbulent broadening
for the GA.
Usage:
> python broaden.py -f <filename> -r <resolving power> -v <vrot> -m <vmacro>
Needs PyAstronomy on Lisa, to install this:
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
from math import sin, pi
from scipy.interpolate import interp1d                      # Interpolation routine
from scipy.special import erf                               # Error function
from scipy.signal import fftconvolve                        # Convolution routine
from PyAstronomy.pyasl import rotBroad, instrBroadGaussFast # Rotational and instumental broadening routines

# Settings & constants
binSize = 0.01     # Size of wavelength bins to resample input spectrum to
finalBins = 0.01   # Size of wavelength bins of output spectrum after broadening
limbDark = 0.6     # Limb darkening coefficient to be used by rotational broadening
c = 299792.458     # Speed of light in km/s


def main():
    # Read in the needed values from the command line
    arguments = parseArguments()
    # Read in the spectrum
    try:
        wlc, flux = np.loadtxt(arguments.fileName, unpack=True, skiprows=1)
    except IOError as ioe:
        print(ioe, "Input spectrum " + arguments.fileName + " not found!")
        sys.exit()
    # Resample the input spectrum to even wavelength bins of <binSize> Angstrom
    newWlc = np.arange(wlc[0]+binSize, wlc[-1]-binSize, binSize)
    flux = np.interp(newWlc, wlc, flux)
    wlc = newWlc
    # Apply instrumental broadening
    flux = instrBroadGaussFast(wlc, flux, arguments.res, maxsig=5.0, edgeHandling="firstlast")
    # Apply rotational broadening
    flux = rotBroad(wlc, flux, limbDark, arguments.vrot)
    # Apply macroturbulent broadening
    if arguments.vmacro != -1:
        flux = macroBroad(wlc, flux, arguments.vmacro)
    # Resample to <finalBins> Angstrom
    if finalBins != binSize:
        newWlc = np.arange(wlc[0]+finalBins, wlc[-1]-finalBins, finalBins)
        flux = np.interp(newWlc, wlc, flux)
    # Write output file
    out_f = open(arguments.fileName + ".fin", 'w')
    out_f.write("#" + str(len(wlc)) + "\t" + "#0 \n")
    for i in range(len(wlc)-1):
        out_f.write(str(wlc[i]) + "\t" + str(flux[i]) + "\n")
    out_f.close()
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

    conv_mode = "valid"

    # Do the convolution
    newydata = fftconvolve(extended, profile / profile.sum(), mode=conv_mode)

    return newydata


if __name__ == "__main__":
    main()
