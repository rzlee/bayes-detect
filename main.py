# Main toy example
from Src import plot


SrcArray = [[43.71, 22.91, 10.54, 3.34],
            [101.62, 40.60, 1.37, 3.40],
            [92.63, 110.56, 1.81, 3.66],
            [183.60, 85.90, 1.23, 5.06],
            [34.12, 162.54, 1.95, 6.02],
            [153.87, 169.18, 1.06, 6.61],
            [155.54, 32.14, 1.46, 4.05],
            [130.56, 183.48, 1.63, 4.11]]

#prior_array = [[0.0,100.0],[0.0,100.0],[1.0,12.5],[2.0,9.0]]

# When making a fake image with new prior limits. Don't forget update the prior values in config.cfg before running. 
data_map = plot.make_source(SrcArray, 200, 200)
noise = 2.0
data_map = plot.add_gaussian_noise(mean=0,sd=noise,data=data_map)
plot.write(data_map, "assets/simulated_images/multinest_toy_noised")

#Import sources should be here because we are loading the simulated image globally inside it
from Src import sources
#[X,Y,A,R]
prior_array = [[0.0,200.0],[0.0,200.0],[1.0,12.5],[2.0,9.0]]

sources.run_source_detect(mode = "Manual")
