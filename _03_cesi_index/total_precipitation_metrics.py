import pandas as pd
import numpy as np 
import geopandas as gpd 
import json


### Calculates total precipitation per EA (in mm) over the study period 

### read in geodataframes with the precipitation interpolations for all 5019 EAs 
precip_all_22 = gpd.read_file('/work/pi_jtaneja_umass_edu/kdonkor_umass_edu/Geo_Interpolation/Geo_Interpolation_Outputs/precip_idw_all_EAs_2022_5019_EAs.geojson')

precip_all_23 = gpd.read_file('/work/pi_jtaneja_umass_edu/kdonkor_umass_edu/Geo_Interpolation/Geo_Interpolation_Outputs/precip_idw_all_EAs_2023_5019_EAs.geojson')

# concatenate 2022 & 2023 dataframes 
precip_all = pd.concat([precip_all_22, precip_all_23], ignore_index=True).rename(columns = {'interp_val_Precip_Hour':'Precip'})


# calculate the total rainfall per EA 
total_rainfall = (
    precip_all
    .groupby('ea_code9ch', as_index=False)['Precip']
    .sum()
    .rename(columns={'Precip': 'Total_Rainfall'})
)

# drop duplicates if any 
total_rainfall = total_rainfall.drop_duplicates()

## Save dataframe  
total_rainfall.to_csv('/work/pi_jtaneja_umass_edu/kdonkor_umass_edu/Geo_Interpolation/Geo_Interpolation_Outputs/total_rainfall_5019_EAs.csv')