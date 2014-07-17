import plot
import sources

SrcArray = [[43.71, 22.91, 10.54, 3.34],
            [101.62, 40.60, 1.37, 3.40],
            [92.63, 110.56, 1.81, 3.66],
            [183.60, 85.90, 1.23, 5.06],
            [34.12, 162.54, 1.95, 6.02],
            [153.87, 169.18, 1.06, 6.61],
            [155.54, 32.14, 1.46, 4.05],
            [130.56, 183.48, 1.63, 4.11]]

data_map = plot.make_source(src_array = SrcArray,height=200, width=200)
noise = 2.0
data_map = plot.add_gaussian_noise(mean=0,sd=noise,data=data_map)
sources.write(data_map, "simulated_images\multinest_noised")

prior_array = [[0.0,200.0],[0.0,200.0],[1.0,12.5],[2.0,9.0]]

sources.run_source_detect(samples = 4000, iterations = 25000, sample_method = "metropolis", prior = prior_array, noise_rms = noise)