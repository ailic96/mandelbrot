# coding=utf-8
""""
Authors:
        Bruno Beljo
        Antonio Boban
        Anton Ilic

Description:
        Computing of Mandelbrot set using mpi4py parallel computing
        Method:
            Blocking communication
Data types:
    width  = int        Image width
    height  = int        Image height
    x1 = float       minimum X-Axis value
    x2 = float        maximum X-Axis value
    y1 = float        minimum Y-Axis value
    y2 = float        maximum Y-Axis value
    maxit = int       maximum interation

Test values:
    width,  height  = 500, 250
    x1, x2 = -2.0, 1.0
    y1, y2 = -1.0, 1.0
    maxit  = 141

Running instructions:
    Code runs in Linux terminal
    Structure:
        mpirun -np numProcesses mandelbrot_p.py width height x1 x2 y1 y2 maxit change_color
    Example:
        mpirun -np 3 python3 mandelbrot_p.py 512 512 -1.5 1.0 -1 1 200 1
If parallel code shows some Darth Vader error:
    echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope

"""

from functions import *
from mpi4py import MPI                  #Used for parallel computing
from matplotlib import pyplot as plt    #Used for graphing
import numpy as np                      #Used for arrays
import os                               #Used for printing path
import sys                              #Used for argument input

#makes using -h parameter possible by ignoring other parameters
if len(sys.argv) == 2:
   help_menu()
else:
    width =  int(sys.argv[1])   #Image width
    height = int(sys.argv[2])   #Image height
    x1 = float(sys.argv[3])     #x axis minimum
    x2 = float(sys.argv[4])     #x axis maximum
    y1 = float(sys.argv[5])     #y axis minimum
    y2 = float(sys.argv[6])     #x axis maximum
    maxit = int(sys.argv[7])    #maximum number of iterations
    c = int(sys.argv[8])        # maximum number of iterations

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

#Calculating code execution time in MPI module
t0 = MPI.Wtime()

# Rows to compute
N = height // size + (height % size > rank)
# Positioning to the first line
start = comm.scan(N)-N
# Local result
Cl = np.empty([N, width], dtype='i')
# Translating
dx = (x2 - x1) / width
dy = (y2 - y1) / height

for i in range(N):
    y = y1 + (start+i) * dy
    for j in range(width):
        x = x1 + j * dx
        Cl[i, j] = mandelbrot(x, y, maxit)
#gathering data at root
C = None
counts = comm.gather(N, root=0)

if rank == 0:
    print("Parameters:")
    print("\tIterations: ", maxit)
    print("\tOutput: ", os.path.relpath('mandelbrot_parallel.png', start="./file.txt"))
    print("\tImage width: ", width)
    print("\tImage height: ", height)
    print("\tx1 value: ", x1)
    print("\ty1 value: ", y1)
    print("\tx2 value: ", x2)
    print("\ty2 value: ", y2)
    print("\nComputing...")

    C = np.empty([height, width], dtype='i')

# Setting width border
rowtype = MPI.INT.Create_contiguous(width)
rowtype.Commit()

#Gathering vector arrays
sendbuf=[Cl, MPI.INT]
recvbuf=[C, (counts, None), rowtype]
comm.Gatherv(sendbuf, recvbuf, root=0)

#Freeing memory
rowtype.Free()

if rank == 0:
    # Time calculation result
    t1 = MPI.Wtime() - t0
    print("Computing finished in ", t1, "seconds.")

    # Graphing
    print("Building image...")
    #https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imshow.html
    #konfiguracija grafa, pozivanje funkcije za promjenu grafa, učitavanje varijabli x1,...y1 u graf
    plt.imshow(change_colors(c,C), aspect='equal',cmap=plt.cm.gnuplot2, interpolation='none', extent=(x1, x2, y1, y2))

    plt.title('Mandelbrot set using parallel computing')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')

    plt.savefig('mandelbrot_parallel.png', dpi=1000)     #dpi - dots per inch
    plt.show()

    #Relative path
    path = os.path.relpath('mandelbrot_parallel.png', start="./file.txt")
    print("Image saved in: ", path)

MPI.COMM_WORLD.Barrier()