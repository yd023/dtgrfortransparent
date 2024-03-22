import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pyvredner
import argparse, os
from skimage.transform import resize

def convertEXR2ColorMap(exrfile, pngfile, vmin, vmax, amp, resize_scale, withColorBar = False):
	cubicL = LinearSegmentedColormap.from_list("cubicL", np.loadtxt("CubicL.txt"), N=256)

	def remap(img):
		if amp > 0:
			return np.multiply(np.sign(img), np.log1p(np.abs(amp*img)))
		else:
			return img

	def rgb2gray(img):
		return 0.2989*img[:, :, 0] + 0.5870*img[:, :, 1] + 0.1140*img[:, :, 2]

	img_input = rgb2gray(pyvredner.imread(exrfile)).numpy()
	img_input = remap(img_input)
	val_min = np.min(img_input)
	val_max = np.max(img_input)
	print(val_min, val_max)

	ratio = img_input.shape[0]/img_input.shape[1]
	if withColorBar:
		fig = plt.figure(figsize=(5, 5*ratio/1.1))
	else:
		fig = plt.figure(figsize=(5, 5*ratio))

	plt.imshow(img_input, interpolation='bilinear', vmin=vmin, vmax=vmax, cmap=cubicL)
	plt.axis('off')
	if withColorBar:
		plt.colorbar()
		plt.savefig(pngfile, bbox_inches='tight', pad_inches=0)
	else:
		plt.subplots_adjust(bottom=0.0, left=0.0, top=1.0, right=1.0)
		plt.savefig(pngfile, dpi=100)
	plt.close()

def main(args):
	if os.path.isdir(args.path):
		for root, dirs, files in os.walk(args.path):
			for file in files:
				if file.endswith(".exr"):
					path_in = os.path.join(root, file)
					path_out = os.path.splitext(path_in)[0] + '.png'
					convertEXR2ColorMap(path_in, path_out, args.vmin, args.vmax, args.amp, args.resize, args.colorbar > 0)

	else:
		path_in = args.path
		path_out = os.path.splitext(path_in)[0] + '.png'
		convertEXR2ColorMap(path_in, path_out, args.vmin, args.vmax, args.amp, args.resize, args.colorbar > 0)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
			description='Script for simple EXR utilities',
			epilog='Cheng Zhang (chengz20@uci.edu)')

	parser.add_argument('path', metavar='path', type=str, help='input path to convert to color map')
	parser.add_argument('-vmin', metavar='vmin', type=str, help='minimum value for color map')
	parser.add_argument('-vmax', metavar='vmax', type=str, help='maximum value for color map')
	parser.add_argument('-amp', metavar='amp', type=float, default=-1.0, help='amount to amplify the difference')
	parser.add_argument('-resize', metavar='resize', type=float, default=1.0, help='rescale scalar for error image')
	parser.add_argument('-colorbar', metavar='colorbar', type=int, default=0, help='add colorbar to figure or not')

	args = parser.parse_args()
	main(args)