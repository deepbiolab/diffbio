
We provide the example training set that was used to generate Figures 1a and 5. 

# Structure of the Data
There are three folders
* `training_historic` (16 runs from each of the historic products HP1 ... HP5)
* `training_novel` (4 runs from the novel product NP)
* `testing` (100 runs from the novel product NP) 

Each folder contains a file `experimental_conditions.csv` that lists the process conditions. The subfolders `HP1` ... `HP5` or `NP` then contain experimental runs obtained with these conditions for the products. 

For example `training_historic/HP1/HP1_ExpHist00.csv` contains the measurements obtained for the experimental condition `ExpHist00` (which is found in `training_historic/experimental_conditions.csv`) with the product `HP1`. I.e. measurements for 6 quantities once per day. 
The file `HP1_ExpHist00_details.csv` further contains intermediate measurements throughout each day. This is NOT used for model training and included here to provide a better understanding of the bioprocess emulator.

# Measured Quantities
* VCD (Viable Cell Density)
* Glc (Glucose Concentration)
* Gln (Glutamine Concentration)
* Amm (Ammonium Concentration)
* Lac (Lactate Concentration)
* product (concentration of the desired product, also known as titer)

# Experimental Conditions
* Xv_0, Glc_0, Gln_0, Amm_0, Lac_0 (initial concentrations)
* pH_0, T_0 (initial pH and Temperature)
* pH_switch_time, T_switch_time (time at which pH and Temperature switch)
* pH_switch, T_switch (Temperature and pH after switch)
* Glc_feed, Gln_feed (mass of Glucose and Glutamine in the feed)
* feed_start, feed_end (day at which the feeding starts and ends)
* feed_volume (volume of the feed)
* measurement_volume (once per day this amount of volume is removed from the bioreactor, simulates a measurement)
* stirring_rate 
* initial_volume (initial reactor volume, changes through feeds and measurements)
* total_days (duration of the experiment)


