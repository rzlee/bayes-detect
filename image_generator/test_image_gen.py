import argparse
import numpy as np
from numpy.random import uniform, randint
import test_image

parser = argparse.ArgumentParser(description="Generate test image as a numpy file. It will write three files: one with the x,y, amplitude, and radius values, one image without noise, and one with.")

#sources
parser.add_argument("--n", type=int, help="number of sources", required=True)

#xrange
parser.add_argument("--width", type=int, help="width", required=True)

#yrange
parser.add_argument("--height", type=int, help="height", required=True)

#max amplitude
parser.add_argument("--maxamp", type=float, help="max amplitude", required=True)

#min amplitude
parser.add_argument("--minamp", type=float, help="min amplitude", required=True)

#max radius
parser.add_argument("--maxrad", type=float, help="max radius", required=True)

#max radius
parser.add_argument("--minrad", type=float, help="min radius", required=True)
#noise amt
parser.add_argument("--noise", type=float, help="noise level", required=True)

parser.add_argument("--output_dir", help = "output directory")

args = parser.parse_args()

num_sources = args.n
width = args.width
height = args.height
amp_max = args.maxamp
amp_min = args.minamp
rad_max = args.maxrad
rad_min = args.minrad
noise = args.noise

#make a src array [[x,y,a,r], ...]
src_array = np.zeros((num_sources, 4))
for source in range(src_array.shape[0]):
    src_array[source, 0] = randint(width)
    src_array[source, 1] = randint(height)
    src_array[source, 2] = uniform(low = amp_min, high = amp_max)
    src_array[source, 3] = uniform(low = rad_min, high = rad_max)

non_noisy_sources = test_image.make_source(src_array, height, width)

noised = test_image.add_gaussian_noise(0, noise, non_noisy_sources)

name = "num_sources-" + str(num_sources) + "-width-" + str(width) + "-height-" + str(height) + "-a_min-" + str(int(amp_min)) + "-a_max-" + str(int(amp_max))
name = name + "-rad_min-" + str(int(rad_min)) + "-rad_max-" + str(int(rad_max)) + "-noise-" + str(int(noise))

normal = "noised_" + name + ".npy"
clean = "no_noise_" + name + ".npy"
srcs = "src_array_" + name + ".npy"

if args.output_dir:
    normal = args.output_dir + "/" + normal
    clean = args.output_dir + "/" + clean
    srcs = args.output_dir + "/" + srcs

with open(normal, "wb") as f:
    np.save(f, noised)

with open(clean, "wb") as f:
    np.save(f, non_noisy_sources)

with open(srcs, "wb") as f:
    np.save(f, src_array)
