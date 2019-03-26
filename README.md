# Mandelbrot-set-in-Python
Mandelbrot set with high speed multicore computation and simple GUI for zoom

Lib requirements:
  numba
  numpy
  PIL (pillow)
  Tkinter
  
GIL is lifted using numba compiled functions  
Resulting image is rendered at double resolution (default) and downscaled using antialiasing
Default resolution is 1280x720

At the moment, the only way to adjust resolution and turn of antialiasing is by changing variables at the top of the file

mandelbrot_set.py can be used as a library for plain output of image 
 
mandelbrot_set(xmin,xmax,ymin,ymax,maxiter)|
xmin, xmax - x axis range|
ymin, ymax - y axis range|
maxiter - maximum number of iterations per pixel in computation