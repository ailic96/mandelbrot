#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy


def mandelbrot(x, y, maxit):
    """
    Computes real and imaginary numbers, number of iterations
    returns value of current iteration.
    Takes input of complex numbers:
     x - real value
     y - imaginary value
     maxit -  maximum iteration.
    z is a part of Mandelbrot set only if the absolute value for
    z is greater than 2 and if the function doesn't go to infinity
    (does not diverge when iterated)
    """
    c = x + y * 1j
    z = 0 + 0j
    it = 0
    while abs(z) < 2 and it < maxit:
        z = z*z + c
        it += 1
    return it

help_text = """HELP MENU:\n\n
This code implements compution of Mandelbrot set\n\n
||||||Data types:
\t    w  = int\t          Image width,\n
\t    h  = int\t          Image height,\n
\t    x1 = float\t        minimum X-Axis value,\n
\t    x2 = float\t        maximum X-Axis value,\n
\t    y1 = float\t        minimum Y-Axis value,\n
\t    y2 = float\t        maximum Y-Axis value,\n
\t    maxit = int\t       maximum interation,\n\n
\t    color_mode=int      (0-3) color modes, 0 - default\n

||||||Running Instructions:\n\t    
        Serial: python3 mandelbrot_s.py width height x1 x2 y1 y2 maxIt color_mode
        Parallel: mpirun -np numProcc mandelbrot_p.py width height x1 x2 y1 y2 maxIt color_mode
\nExample:\n\t    
        python3 mandelbrot_s 512 512 -2.0 1.0 -1.0  1.0  250 0
        mpirun -np 3 python3 mandelbrot_s 512 512 -2.0 1.0 -1.0  1.0  250 0
"""


def help_menu():
    """ Help function with instructions for running the code.
     Enables using -h as a help argument"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                        help=help_text)
    parser.parse_args()


def change_colors(c, C):
    """ Argument c calls a mathematical operation used
    on argument C and returns a value. Used for graphing"""
    if c == 0:
        return C            #defualt
    elif c == 1:
        return numpy.sin(numpy.abs(C))
    elif c == 2:
        return numpy.cos(numpy.abs(C))
    elif c == 3:
        return numpy.log(numpy.abs(C))
    else:
        print("Invalid color input! Assigning default color mode...")
        return C

import sys
def progressbar(it, prefix="", size=60, file=sys.stdout):
    """
    Code progress bar, not implemented (yet)
    """
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()
