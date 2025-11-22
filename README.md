# Spatiotemporal Analysis of Weather Extremes and Power Quality and Reliability in Accra, Ghana
Kwame Donkor, June Lukuyu 

## Overview 
This repository contains code for pre-processing and analyzing Power Quality and Reliability (PQR) data alongside weather extremes in the Greater Accra region, Ghana.
It includes workflows for data cleaning, analysis, index creation, and visualization.

## Repository Structure 

### `weather_data` 
* Pre-processing of weather variables (temperature, precipitation, wind, lightning)
* Calculation of weather extremes
* Spatial distribution and mapping of extreme weather events

### `pqr_data_processing`
* Pre-processing of PQR sensor data 
* Calculation of outages and undervoltages for various durations

### `co_occurrence_analysis`
* Analysis of the co-occurrence between weather extremes and PQR metrics 

### `cesi_index`
* Creation of the custom Climate Exposure and Sensitivity Index (CESI) 

### `bivariate analysis_spatial_co_occurrence_plots`
* Visualization of CESI versus grid performance metrics
* Spatial co-occurrence visualizations
