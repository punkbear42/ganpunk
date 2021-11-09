import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import ImageColor

im = cv2.imread('./results/generated_plot_e486_000.png') # read input image

def getClosestColor(pixel,color_set_rgb): # Get the closest color for the pixel
    closest_color = None
    cost_init = 10000
    pixel = np.array(pixel)
    for color in color_set_rgb:
        color = np.array(color)
        cost = np.sum((color - pixel)**2)
        if cost < cost_init:
            cost_init = cost
            closest_color = color
    return closest_color

def getClosestImage(im): # Get the closest image
    color_set = ['#1D1D21','#B02E26', '#5E7C16', '#835432', '#3C44AA', '#8932B8', '#169C9C', '#9D9D97', '#474F52', '#F38BAA',
     '#80C71F', '#FED83D','#3AB3DA' ,'#C74EBD' ,'#F9801D' ,'#F9FFFE'] # Given Colorset
    color_set_rgb= [ImageColor.getrgb(color) for color in color_set] # RGB Colorset

    height, width, channels = im.shape
    im_out = np.zeros((height,width,channels))

    for y in range(0, height):
        for x in range(0, width):
            closest_color = getClosestColor(im[y, x],color_set_rgb)
            im_out[y,x,:] = closest_color
    return im_out


im_out = getClosestImage(im)

# plt.imshow(im_out.astype(np.uint8))
plt.imsave('im_out.png',im_out/255)
# plt.show()