#!/usr/bin/python3

import numpy as np
import argparse
import sys
from numba import jit
import matplotlib.pyplot as plt

@jit(nogil=True,nopython=True,cache=True)
def mandelcalc(z,maxiter,exp):
    '''
    Iterate the mandelbrot calculation until maxiter for given z.
    Exponent of the calculation can be varied using exp.
    '''
    c = z
    z = 0

    for i in range(maxiter):
        z = z**exp + c
        if z.real**2 + z.imag**2 > 4:
            return i

    return 0

@jit(nogil=True,nopython=True,cache=True)
def mandelbrot(x,y,maxiter,exp,skiptest):
    '''
    Generate the mandelbrot figure of resolution y,x
    x,y:        Resolution
    maxiter:    Max. number of iterations per point
    exp:        Alternative exponent for recursive iteration
    skiptest:   Boolean for skipping cardiodi checking
    '''
    array = np.zeros( (len(y), len(x)) )
    for X in range(len(x)):
        for Y in range(len(y)):

            # Iterate over all points
            if skiptest:
                z = x[X] + 1j*y[Y]
                array[Y,X] = mandelcalc(z,maxiter,exp)

            # Leave out cardioid
            else:
                if not test_inner(x[X],y[Y]):
                    z = x[X] + 1j*y[Y]
                    array[Y,X] = mandelcalc(z,maxiter,exp)

    return array

@jit(nogil=True,nopython=True,cache=True)
def test_inner(x,y):
    '''
    Check if (x,y) lies within cardioid
    '''
    p = np.sqrt( (x-.25)**2 + y**2 )
    q = ( x+1 )**2 + y**2

    if x < p - 2*p**2 + .25 or q < 1/16:return 1
    else:return 0

def saturate(array, saturation=None, offset=0):
    '''
    Nonlinearly push high values by
      * Normalize array to [0,1]
      * Apply n-th root

    Inputs:
        array:      Numpy array
        saturation: Integer
        offset:     float
    '''
    if not saturation == None:

        # Compensate an offset due to limited floating point precision
        array = array - np.min(array[array > 0])

        # Saturate
        array = normalize(array)
        saturated_array = np.where(array>0, np.power(array, 1/(2**saturation)), 0)

        # Cut off precision artefacts
        min_value = np.min(saturated_array[saturated_array > 0])
        saturated_array = saturated_array - (1 + offset/10) * min_value

        # Set x=0 for x in array with x<0
        saturated_array = np.where( saturated_array >=0, saturated_array, 0)

        return saturated_array

    else:
        return normalize(array)

def normalize(array):
    '''
    Normalize numpy array to unit interval [0,1]
    '''
    array = array - np.min(array)
    array = array / ( np.max(array)-np.min(array) )

    return array

parser = argparse.ArgumentParser(description='Generate Mandelbrot Fraktal', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-v', '--verbose', action = 'store_const', const = 1,
                    default = 0, help = 'Show additional messages')
parser.add_argument('-m', '--maxiter', help='Maximum number of iterations per point', default=100, type=int)
parser.add_argument('-o', '--outfile', help='Filename for image', default=None, type=str)
parser.add_argument('--colormap', help='Matplotlib colormap', default="viridis", type=str)
parser.add_argument('--coloroffset', help='Offset for colors. Typical: -10->10', default=0, type=float)
parser.add_argument('-f', '--flip', action = 'store_const', const = 1,
                    default = 0, help = 'Turn appleman upright / transpose image')
parser.add_argument('--histogram', action = 'store_const', const = 1,
                    default = 0, help = 'Save histogram of colordistribution to histogram.pdf')
parser.add_argument('-s', '--skiptest', action = 'store_const', const = 1,
                    default = 0, help = 'Turn off pretest for inner of appleman (cardioid)')
parser.add_argument('--saturation', help='Increase colorfulness. Typical: 1, 2, 3,...', default=None, type=int)
parser.add_argument('--exponent', help='Alter exponent for mandelbrot recursion', default=2, type=float)

subparsers = parser.add_subparsers(title='Functions',
                    description='description',
                    help='help',
                    dest='operation')

parser_range = subparsers.add_parser('range', help='Range in xy plane squeezed into square', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_range.add_argument('--xyrange', help='Choose exact sector: xmin:xmax,ymin:ymax', default='-1.5:.5,-1:1', type=str)
parser_range.add_argument('-P', '--pixel', help='Resolution/Length of square image', default=1000, type=int)

parser_point = subparsers.add_parser('point', help='Center on point in xy-plane', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_point.add_argument('-F', '--factor', help='Linear zoom factor', default=1, type=float)
parser_point.add_argument('--power', help='Exponential zoom factor', default=2, type=float)
parser_point.add_argument('-p', '--point', help='Center point. x:y for x+iy', default='-.5:0', type=str)
parser_point.add_argument('-R', '--resolution', help='widthxheight', default=None, type=str)
parser_point.add_argument('-P', '--pixel', help='Resolution/Length of square image', default=1000, type=int)

cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn', 'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink', 'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper', 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic', 'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c', 'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv', 'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']

args = parser.parse_args()

if not args.colormap in cmaps and not args.colormap.replace('_r','') in cmaps:
    raise ValueError("Not a matplotlib colormap! Choose one of\n{}\n{}".format(",".join(cmaps),'See https://matplotlib.org/examples/color/colormaps_reference.html'))

#################### RANGE ####################

if args.operation == "range":

    xmin = float(args.xyrange.split(',')[0].split(':')[0])
    xmax = float(args.xyrange.split(',')[0].split(':')[1])
    ymin = float(args.xyrange.split(',')[1].split(':')[0])
    ymax = float(args.xyrange.split(',')[1].split(':')[1])

    Dx = xmax - xmin
    Dy = ymax - ymin

    # Detect orientation
    if Dy >= Dx:
        height = int(np.round(args.pixel * (Dy / Dx)))
        width = args.pixel
    else:
        width = int(np.round(args.pixel * (Dx / Dy)))
        height = args.pixel

#################### POINT ####################

elif args.operation == "point":

    point = float(args.point.split(':')[0]) + 1j * float(args.point.split(':')[1])

    if not args.resolution == None:

        width  = int(args.resolution.split('x')[0])
        height = int(args.resolution.split('x')[1])

        # Generate boundaries from resolution
        if width >= height:
            Dy = 2/(args.factor)**args.power
            Dx = 2/(args.factor)**args.power * (width/height)
            xmin = point.real - Dx/2
            xmax = point.real + Dx/2
            ymin = point.imag - Dy/2
            ymax = point.imag + Dy/2

        else:
            Dx = (2/args.factor)**args.power
            Dy = (2/args.factor)**args.power * (height/width)
            xmin = point.real - Dx/2
            xmax = point.real + Dx/2
            ymin = point.imag - Dy/2
            ymax = point.imag + Dy/2

    else:
        # No resolution given, choose square
        width = args.pixel
        height = args.pixel
        Dx = 2/(args.factor)**args.power
        Dy = 2/(args.factor)**args.power
        xmin = point.real - Dx/2
        xmax = point.real + Dx/2
        ymin = point.imag - Dy/2
        ymax = point.imag + Dy/2

###############################################

# Warnung high resolution
if width * height > 7000 * 7000:
    while True:
        print('WARNING: width: {}, height: {}. This might consume a lot of RAM! Continue? [y/N]'.format(width,height))
        sure = input()
        if sure in ['','n','N']:exit(0)
        elif sure in ['y','Y']:break
        else:print('Continue? [y/N]')

if args.outfile == None:

    if args.verbose:
        print('No filename given, taking {} instead.'.format(args.outfile))

    args.outfile = 'mandelbrot_x{}a{}_y{}a{}_{}x{}_{}.png'.format(xmin,xmax,ymin,ymax,width,height,args.maxiter)

#################### ACTUAL CALCULATION ####################

x = np.linspace(xmin, xmax, width)
y = np.linspace(ymin, ymax, height)

array = mandelbrot(x,y,maxiter=args.maxiter,exp=args.exponent,skiptest=args.skiptest)

# Flip image
if args.flip: array = array.T

array = saturate(array, saturation=args.saturation, offset=args.coloroffset)

if args.histogram:

    if args.verbose:
        print('Generating histogram to histogram.pdf')

    hist = np.histogram(array.flatten(), bins='auto')
    plt.plot(hist[1][:-1], hist[0])
    plt.suptitle('Color distribution')
    plt.title('np.max(array) = {}'.format(np.max(array)))
    plt.xlabel('# Pixel')
    plt.xlabel('Color value')
    plt.savefig('histogram.pdf', bbox_inches="tight")

if args.verbose: print('Writing to: {}'.format(args.outfile))
plt.imsave(args.outfile,array,cmap=plt.get_cmap(args.colormap))
