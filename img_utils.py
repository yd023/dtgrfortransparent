import OpenEXR
import Imath
from PIL import Image
import array
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pyvredner

def convertEXR2PNG(exrfile, pngfile, exposure_scale):
	File = OpenEXR.InputFile(exrfile)
	PixType = Imath.PixelType(Imath.PixelType.FLOAT)
	DW = File.header()['dataWindow']
	Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)

	RedStr = File.channel('R', PixType)
	GreenStr = File.channel('G', PixType)
	BlueStr = File.channel('B', PixType)

	Red = array.array('f', RedStr)
	Green = array.array('f', GreenStr)
	Blue = array.array('f', BlueStr)

	for I in range(len(Red)):
		Red[I] = encode2SRGB(Red[I] * exposure_scale)
		# Red[I] = MyToneMap(Red[I])

	for I in range(len(Green)):
		Green[I] = encode2SRGB(Green[I] * exposure_scale)
		# Green[I] = MyToneMap(Green[I])

	for I in range(len(Blue)):
		Blue[I] = encode2SRGB(Blue[I] * exposure_scale)
		# Blue[I] = MyToneMap(Blue[I])

	rgbf = [Image.frombytes("F", Size, Red.tobytes())]
	rgbf.append(Image.frombytes("F", Size, Green.tobytes()))
	rgbf.append(Image.frombytes("F", Size, Blue.tobytes()))

	rgb8 = [im.convert("L") for im in rgbf]
	Image.merge("RGB", rgb8).save(pngfile, "PNG")

def encode2SRGB(v):
	if (v <= 0.0031308):
		return (v * 12.92) * 255.0
	else:
		return (1.055*(v**(1.0/2.4))-0.055) * 255.0

def MyToneMap(v):
	v = v**(1.0/1.5)*2**(1.0)
	v = v*255.0
	return v

def convertEXR2ColorMap(exrfile, pngfile, vmin, vmax, withColorBar = False, amp = False, amp_value = 1):
	cubicL = LinearSegmentedColormap.from_list("cubicL", np.loadtxt("CubicL.txt"), N=256)

	def remap(img, amp, amp_value):
		if amp:
			return np.multiply(np.sign(img), np.log1p(np.abs(amp_value*img)))
			# return np.multiply(np.sign(img), np.log1p(np.abs(np.log1p(np.abs(100*img)))))
		else:
			return img

	def rgb2gray(img):
		return 0.2989*img[:, :, 0] + 0.5870*img[:, :, 1] + 0.1140*img[:, :, 2]

	img_input = rgb2gray(pyvredner.imread(exrfile))
	img_input = remap(img_input, amp, amp_value)
	plt.imshow(img_input, interpolation='bilinear', vmin=vmin, vmax=vmax, cmap=cubicL)
	plt.axis('off')
	if withColorBar:
		plt.colorbar()
	plt.savefig(pngfile, bbox_inches='tight', pad_inches=0)
	plt.close()