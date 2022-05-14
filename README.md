# Experiments with PCA on fashion mnist dataset
I will use 3 datasets (pullovers, sneakers and trousers) from mnist fashion dataset and observe how PCA method will affect the data.

# Primary Components
Visualization of the 25 first primary components (new vector basis for our data). First vectors carry most information. <br><br>
![components](out/primaryComponents.png)

# Covariance Matrices
PCA minimized feature covariance <br>
![covariance](out/covariance.png)

# Feature Variances
After PCA the variance is cumulated in the few first features - the rest could be considered noise and discarded. <br>
![variances](out/variances.png)

# How dimension reduction affects data in original base
Few examples of how the data looks after dimension reduction (by discarding features with the least variance) after converting back to the original basis <br>
## Pullover
![Pullover](out/PulloverExample.png)
## Sneaker
![Sneaker](out/SneakerExample.png)
## Trouser
![Trouser](out/TrouserExample.png)

# First two features plotted on 2D pane
After plotting the first two features of the data set, we  can see that first 2 they 
(after applying PCA - before that would be values of 2 first pixels on the picture) grouped the data into discernable clusters. <br>
![2D](out/first2features.png)

