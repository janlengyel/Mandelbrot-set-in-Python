import numpy as np
from numba import jit
from PIL import Image,ImageColor,ImageTk
import tkinter as t
import threading


width=1280
height=720
superres=2
colorres=2**8

res_image=None
palette=[ImageColor.getrgb(f"hsl({((i/colorres)/256*360)%360},100%,{0.000000000045*(i/colorres-256)**5+50}%)") for i in range(256*colorres)]


def mandelbrot_set(xmin,xmax,ymin,ymax,maxiter):
    global res_image
    r1 = np.linspace(xmin, xmax, width*superres)
    r2 = np.linspace(ymin, ymax, height*superres)
    res=np.empty(width*height*(superres**2))

    @jit(nogil=1, parallel=True, nopython=True)
    def mandelbrot(c, maxiter):
        z = c
        for n in range(maxiter):
            if abs(z) > 2:
                return n, z
            z = z * z + c
        return 0, 0

    @jit(nogil=True, parallel=True)
    def write(x, y, linx, liny, arr, iter):
        res = mandelbrot(linx[x] + 1j * liny[y], iter)
        if res[0] == iter or res[0] == 0:
            num = res[0]
        else:
            num = (res[0] + 1 - np.log(np.log2(abs(res[1]))))
        arr[y * width * superres + x] = num % 256

    @jit(nogil=1, parallel=True, nopython=True)
    def calculate(res, slice, slices, r1, r2, iter): 
        for j in range(int(height * superres * slice / slices), int(height * superres * (slice + 1) / slices)):
            for i in range(width * superres):
                write(i, j, r1, r2, res, iter)
        #print(slice, "done")

    threads=[]
    for k in range(16):
        thread=threading.Thread(target=lambda:calculate(res,k,16,r1,r2,maxiter))
        #print(f"thread {k} created")
        thread.start()
        threads.append(thread)

    for l in threads:
        l.join()

    @jit(nogil=True)
    def convert(img_arr,palette):
        new_arr=np.empty((len(img_arr),3),np.uint8)
        for i in range(len(img_arr)):
            new_arr[i]=palette[int((img_arr[i]*colorres)//1)]
        return new_arr

    res=convert(res,palette)
    res=res.reshape((height*superres,width*superres,3))
    img=Image.fromarray(res)
    img=img.resize((width,height),resample=Image.ANTIALIAS)
    #img=img.filter(ImageFilter.EMBOSS)
    return img


def set_image(canvas_image,canvas,center,size,iter):
    global mandelbrot
    root.config(cursor="circle")
    mandelbrot = ImageTk.PhotoImage(mandelbrot_set(center[0]-size[0]/2, center[0]+size[0]/2,
                                                   center[1]-size[1]/2, center[1]+size[1]/2, iter))
    root.config(cursor="")
    canvas.itemconfigure(canvas_image,image=mandelbrot)


def click(event):
    global iter,zoom,center
    x,y=event.x,event.y
    center[0] = center[0]-(5/zoom)/2+x/width*5/zoom # center[0] + (-5+10*x/width)/(2*zoom)
    center[1] = center[1]-(3/zoom)/2+y/height*3/zoom # center[1] + (-3+10*y/height)/(2*zoom)
    zoom*=2
    iter*=1.5
    threading.Thread(None, lambda: set_image(image, canvas, center, list(map(lambda x:x/zoom,(5, 3))), iter)).start()
    print(center,zoom,iter)

root=t.Tk()
canvas=t.Canvas(root,width=width,height=height,background="black")
canvas.pack()
mandelbrot=None
image=canvas.create_image(width/2,height/2,image=None)
iter=80
zoom=1
center=[-0.5,0]
threading.Thread(None,lambda:set_image(image,canvas,center,(5,3),iter)).start()
canvas.bind("<1>",click)
root.mainloop()



