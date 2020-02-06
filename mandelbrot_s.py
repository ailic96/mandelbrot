# coding=utf-8
""""
Authors:
        Bruno Beljo
        Antonio Boban
        Anton Ilic

Description:
        Computing of Mandelbrot set using serial computing

Data types:
    width  = int      Image width
    height  = int     Image height
    x1 = float        minimum X-Axis value
    x2 = float        maximum X-Axis value
    y1 = float        minimum Y-Axis value
    y2 = float        maximum Y-Axis value
    maxit = int       maximum interation

Test values:
    width,  height  = 500, 250
    x1, x2 = -2.0, 1.0
    y1, y2 = -1.0, 1.0
    maxit  = 141

Running instruction:
    Code runs in Linux terminal
    Structure:
        python3 mandelbrot_s.py width height x1 x2 y1 y2 maxit change_color
    Example:
        python3 mandelbrot_p.py 1024 1024 -0.74877 -0.74872 0.065053 0.065103 2048 3
"""
from functions import *
import time
import numpy as np
from matplotlib import pyplot as plt
import os
import sys

"""makes using -h parameter possible by ignoring other arguments.
If there are more arguments, takes their input into compution
"""
if len(sys.argv) == 2:
   help_menu()
else:
    width =  int(sys.argv[1])       #Image width
    height = int(sys.argv[2])       #Image height
    x1 = float(sys.argv[3])     #x axis minimum
    x2 = float(sys.argv[4])     #x axis maximum
    y1 = float(sys.argv[5])     #y axis minimum
    y2 = float(sys.argv[6])     #x axis maximum
    maxit = int(sys.argv[7])    #maximum number of iterations
    c = int(sys.argv[8])        #color mode

#Printing arguments used code was ran
print("Parameters:")
print("\tOutput: ", os.path.relpath('mandelbrot_serial.png', start="./file.txt"))
print("\tImage width: ", width)
print("\tImage height:", height)
print("\tX-Axis minimum:", x1)
print("\tX-Axis maximum:", x2)
print("\tY-Axis minimum: ", y1)
print("\tY-Axis maximum: ", y2)
print("\tIterations: ", maxit)

#Calculating execution time
start = time.time ()
print("Computing...")

C = np.zeros([height, width], dtype='i')
dx = (x2-x1)/width
dy = (y2-y1)/height

for i in range(height):
    y = y1 + i * dy
    for j in range(width):
        x = x1 + j * dx
        C[i,j] = mandelbrot(x, y, maxit)

# Time calculation result
end = time.time()
print("Computing finished in ", end - start, "seconds.")

#Graphing
print("Building image...")
plt.imshow(change_colors(c,C), aspect='equal', cmap=plt.cm.gnuplot2, interpolation='bilinear', extent=(x1, x2, y1, y2))
plt.title('Mandelbrot set using serial computing')
plt.xlabel('Real')
plt.ylabel('Imaginary')

plt.savefig('mandelbrot_serial.png', dpi=1000)
plt.show()

#Relative path
path = os.path.relpath('mandelbrot_serial.png', start="./file.txt")
print("Image save path: ", path)