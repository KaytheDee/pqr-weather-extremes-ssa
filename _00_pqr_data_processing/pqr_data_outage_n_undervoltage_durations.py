#Import packages 

import numpy as np 
import pandas as pd 
import json 
import glob
import os
import csv
from statsmodels.tsa.seasonal import seasonal_decompose


####### 


## this code is to process the PQR data to calculate the number of occurrences of 1+hour and 4+ hour UNDERVOLTAGES PER SITE 
## Outputs of this goes into the bivariate analysis 


####### 


min_volt_thresh = 20 # there were values THIS low in the dataset 

# Frequency Threshold 
freq_thresh = 30

## Normal Undervoltage Threshold (10% below) 
under_volt_thresh = 230 - (0.1 * 230)

## Very low Undervoltage Threshold (20% below) 
very_low_undervolt_thresh = 230 - (0.2 * 230) 

## Normal Overvoltage Threshold (10% above) 
over_volt_thresh = 230 + (0.1 * 230)

## Very high overvoltage threshold (20% above)
very_high_overvolt_thresh = 230 + (0.2 * 230) 


#### Helper Functions #### 


def filter_by_site(df, site_id):
    new_df = df[df['site_id'] == site_id]
    return new_df 


## clean up sensor dataframe 

def clean_up_df(df):
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values(by='time').reset_index(drop=True)

    # Select relevant columns
    df = df[['time', 'respondent_id', 'site_latitude', 'site_longitude', 
             'voltage', 'frequency', 'site_id', 'district', 'region', 'is_powered']]

    return df 


def read_and_concatenate_csvs(folder_path):
    # Find all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    
    # Read and concatenate
    df_list = [pd.read_csv(file, low_memory=False) for file in csv_files]
    concatenated_df = pd.concat(df_list, ignore_index=True)
    
    print(f"Concatenated {len(csv_files)} files.")
    return concatenated_df



####### 

## Process sensor data (for all sites) and output more compact 2-minute version 

def process_data_voltage_2min_version(folder_path, min_volt_thresh, under_volt_thresh, over_volt_thresh, site_ids=None, buffer_seconds=45):
    
    # Read and concatenate CSVs
    print("Concatenating files in folder...")
    df = read_and_concatenate_csvs(folder_path)
    
    # Clean up the data
    print("Cleaning up merged folder files...")
    df = clean_up_df(df)

    # Initialize an empty list to store results
    all_stats = []
    
    # If site_ids are provided, process for each site
    if site_ids is not None:
        for site_id in site_ids:

            #### Run the below per site_id #### 
            site_df = filter_by_site(df, site_id=site_id)
            
            # If no data for that site 
            if site_df.empty:
                print(f"Site {site_id} not found or no data available. Skipping...")
                continue

            #### VOLTAGE SECTION #### 
            # Perform resampling to 2-min 
            print(f"Resampling for site {site_id}...")
            df_voltage_2_min = analyze_sensor_voltage_flags(
                site_df, 
                min_volt_thresh, 
                under_volt_thresh, 
                over_volt_thresh
            )
            
            #### APPEND TO LIST #### 
            all_stats.append(df_voltage_2_min)
    
    else:
        # If no specific sites are provided 
        print("No site list provided...")

    # Concatenate results if there are any
    if all_stats:
        print("Concatenating 2-min data for all sites")
        final_df = pd.concat(all_stats, ignore_index=True)
    else:
        final_df = pd.DataFrame()  # Return an empty DataFrame if no data is processed
    
    return final_df



## process sensor data into 2-minute version for voltage metrics 

def analyze_sensor_voltage_flags(
    df, 
    min_volt_thresh, 
    under_volt_thresh, 
    over_volt_thresh
):
    df = df.copy()
    unique_sensors = df['respondent_id'].unique()[:3]
    result_df = df[['time']].drop_duplicates().sort_values('time').reset_index(drop=True)

    for i, sensor in enumerate(unique_sensors, start=1):
        sensor_df = df[df['respondent_id'] == sensor].sort_values('time').copy()

        # Voltage flag checks
        sensor_df[f'sensor_{i}_undervolt'] = (
            (sensor_df['voltage'] > min_volt_thresh) &
            (sensor_df['voltage'] < under_volt_thresh) &
            (sensor_df['is_powered'])
        ).astype(int)

        sensor_df[f'sensor_{i}_overvolt'] = (
            (sensor_df['voltage'] > over_volt_thresh) &
            (sensor_df['is_powered'])
        ).astype(int)

        # Merge into result
        result_df = result_df.merge(
            sensor_df[[
                'time', 
                f'sensor_{i}_undervolt', 
                f'sensor_{i}_overvolt', 
            ]],
            on='time', how='left'
        )

    # Add site_id column (assumes all rows in df have the same site_id)
    result_df['site_id'] = df['site_id'].iloc[0]

    # Reorder columns to place 'site_id' after 'time'
    columns = ['time', 'site_id'] + [col for col in result_df.columns if col not in ['time', 'site_id']]
    result_df = result_df[columns]

    return result_df



### Resample data for hourly voltage metrics 

def process_hourly_voltage_data_only(df_voltage_2_min, min_undervolt_duration, min_overvolt_duration):

    all_stats = []

    # run this for each unique site_id 
    print("Extracting unique site_ids from 2-min file")
    site_ids = df_voltage_2_min['site_id'].unique()

    for site_id in site_ids:
        
        site_df = df_voltage_2_min[df_voltage_2_min['site_id'] == site_id]

        # hourly per sensor  
        print(f"Perfoming individual sensor resampling for site {site_id}")
        per_sensor_hour_resampled = summarize_voltage_events_per_hour_v2(site_df, 
                                                                      min_undervolt_duration, 
                                                                      min_overvolt_duration)

        ## total hourly 
        print(f"Perfoming TOTAL sensor resampling for site {site_id}")
        total_volt_hour_resampled = average_voltage_events_all_sensors_v2(per_sensor_hour_resampled)
        
        # append to list 
        all_stats.append(total_volt_hour_resampled)

    ## Concatenate dfs for all sites 
    print("Concatenating all 1 hourly resampled outage data")
    final_df = pd.concat(all_stats, ignore_index=True)

    return final_df



## helper function (embedded in previous function)
# specifies minimum duration for an undervoltage to be flagged 
# finds undervoltage events & duration per hour (for individual sensors) 

# ignores momentary blips in voltage restoration 
# voltage has to be within limits for at least 5 minutes to be considered valid 

def summarize_voltage_events_per_hour_v2(
    df_flags,
    min_undervolt_duration=5,
    min_overvolt_duration=2,
    smooth_gap_duration=5  # in minutes
):
    df = df_flags.copy()
    df = df.sort_values('time').reset_index(drop=True)
    df['hour'] = df['time'].dt.floor('h')

    sensor_cols = [col for col in df.columns if col.startswith('sensor_') and any(k in col for k in ['undervolt', 'overvolt'])]
    sensor_ids = sorted(set(col.split('_')[1] for col in sensor_cols))

    all_hours = pd.date_range(start=df['hour'].min(), end=df['hour'].max(), freq='h')
    result = pd.DataFrame({'time': all_hours})
    result['site_id'] = df['site_id'].iloc[0]

    for sensor in sensor_ids:
        for event_type in ['undervolt', 'overvolt']:
            flag_col = f'sensor_{sensor}_{event_type}'

            if flag_col not in df.columns:
                df[flag_col] = 0

            temp_df = df[['time', 'hour', flag_col]].copy()
            temp_df['is_event_raw'] = temp_df[flag_col] == 1

            # Step 1: Group based on transitions
            temp_df['group'] = (temp_df['is_event_raw'] != temp_df['is_event_raw'].shift(1)).cumsum()

            # Step 2: Compute duration and status for each group
            group_info = temp_df.groupby('group').agg(
                start_time=('time', 'first'),
                end_time=('time', 'last'),
                is_event=('is_event_raw', 'first')
            )
            group_info['duration_min'] = (group_info['end_time'] - group_info['start_time']).dt.total_seconds() / 60

            # Step 3: Smooth short non-event gaps
            smoothed_event = group_info['is_event'].copy()
            for i in range(1, len(group_info) - 1):
                if (
                    not group_info.iloc[i]['is_event'] and
                    group_info.iloc[i - 1]['is_event'] and
                    group_info.iloc[i + 1]['is_event'] and
                    group_info.iloc[i]['duration_min'] <= smooth_gap_duration
                ):
                    smoothed_event.iloc[i] = True

            group_info['is_event_smoothed'] = smoothed_event

            # Step 4: Map smoothed group status back to rows
            smoothed_map = group_info['is_event_smoothed'].to_dict()
            temp_df['is_event'] = temp_df['group'].map(smoothed_map)

            # Step 5: Identify event starts and assign event group
            temp_df['event_start'] = temp_df['is_event'] & ~temp_df['is_event'].shift(1, fill_value=False)
            temp_df['event_group'] = temp_df['event_start'].cumsum() * temp_df['is_event']

            # Step 6: Get valid events using actual duration
            event_durations = temp_df[temp_df['is_event']].groupby('event_group').agg(
                start=('time', 'first'),
                end=('time', 'last')
            )
            event_durations['duration_min'] = (event_durations['end'] - event_durations['start']).dt.total_seconds() / 60

            threshold = min_undervolt_duration if event_type == 'undervolt' else min_overvolt_duration
            valid_events = event_durations[event_durations['duration_min'] >= threshold]
            valid_groups = valid_events.index

            # Step 7: Assign valid rows and summarize by hour
            valid_df = temp_df[temp_df['event_group'].isin(valid_groups)].copy()
            valid_df['event_duration_min'] = valid_df['event_group'].map(valid_events['duration_min'])

            # First, assign the hour of each event based on its start time
            valid_events = valid_events.copy()
            valid_events['hour'] = valid_events['start'].dt.floor('h')
            
            # Then group and summarize durations directly from valid_events
            summary = valid_events.groupby('hour').agg(
                **{
                    f'{event_type}_sensor_{sensor}_events': ('duration_min', 'count'),
                    f'{event_type}_sensor_{sensor}_duration': ('duration_min', 'sum')
                }
            ).reset_index()

            result = pd.merge(result, summary, left_on='time', right_on='hour', how='left')
            result.drop(columns='hour', inplace=True)

    # Fill missing values
    event_cols = [col for col in result.columns if col.startswith(('undervolt', 'overvolt'))]
    result[event_cols] = result[event_cols].fillna(0)

    return result




## Helper function embedded in main hourly voltage metrics function
## Aggregates metrics for individual sensors per site 

def average_voltage_events_all_sensors_v2(hourly_summary_df):
    df = hourly_summary_df.copy()

    # Identify the correct columns based on the naming pattern
    undervolt_event_cols = [col for col in df.columns if col.startswith('undervolt_sensor_') and col.endswith('_events')]
    undervolt_duration_cols = [col for col in df.columns if col.startswith('undervolt_sensor_') and col.endswith('_duration')]
    overvolt_event_cols = [col for col in df.columns if col.startswith('overvolt_sensor_') and col.endswith('_events')]
    overvolt_duration_cols = [col for col in df.columns if col.startswith('overvolt_sensor_') and col.endswith('_duration')]

    # Compute row-wise averages and assign with "total_*" naming     ## Average instead of sum here ## 
    df['total_undervolt_events'] = df[undervolt_event_cols].mean(axis=1)
    df['total_undervolt_duration'] = df[undervolt_duration_cols].mean(axis=1)
    df['total_overvolt_events'] = df[overvolt_event_cols].mean(axis=1)
    df['total_overvolt_duration'] = df[overvolt_duration_cols].mean(axis=1)

    # Keep only relevant columns
    result = df[['time', 'site_id',
                 'total_undervolt_events', 'total_undervolt_duration',
                 'total_overvolt_events', 'total_overvolt_duration'
                ]]

    return result



### Merge hourly outage and voltage metrics data 


def process_merge_voltage_n_outage_hourly(outage_hr_resampled, voltage_hr_resampled):

    all_stats = []

    # run this for each unique site_id 
    print("Extracting unique site_ids from file")
    site_ids = outage_hr_resampled['site_id'].unique()

    for site_id in site_ids:
        
        # Per site 
        outage_hr_resampled_site = outage_hr_resampled[outage_hr_resampled['site_id'] == site_id]
        
        voltage_hr_resampled_site = voltage_hr_resampled[voltage_hr_resampled['site_id'] == site_id]

        
        #### MERGE OUTAGE & VOLTAGE HOURLY DFs #### 
        print(f"Merging voltage & outage data for site {site_id}")
        merged_outage_n_voltage_df = outage_hr_resampled_site.merge(voltage_hr_resampled_site, on=['time', 'site_id'], how='left')

            
        #### APPEND TO LIST #### 
        all_stats.append(merged_outage_n_voltage_df)

    ## Concatenate dfs for all sites 
    print("Concatenating all merged voltage & outage metrics")
    final_df = pd.concat(all_stats, ignore_index=True)

    return final_df


####### 


### List of ALL sites 

full_site_df = pd.read_csv('/work/pi_jtaneja_umass_edu/kdonkor_umass_edu/common_sites_merged_df_final.csv')
list_full = list(np.sort(full_site_df['site_id'].unique()))


# ------------ # 


###### 2022 


# Folder Path 
fold_path = '/work/pi_jtaneja_umass_edu/kdonkor_umass_edu/2022_PQR_Data/2022'


#### SPECIFY DURATIONS 

# minimum outage duration 
min_outage_duration = 5

# overvolt, very_high_overvolt
min_overvolt_duration = 5           ### 5_min 


### Keeping the outage duration & overvolt duration constant because the FOCUS is only on the undervoltages here 


# ------------ # 

## Process 2-min voltage data 
test_2min_volt = process_data_voltage_2min_version(fold_path, min_volt_thresh, under_volt_thresh, over_volt_thresh, site_ids=list_full, buffer_seconds=45)

# ------------ # 


###### 60 min 

# undervolt
min_undervolt_duration = 60 

## Voltage hourly resample 
voltage_hourly_22 = process_hourly_voltage_data_only(test_2min_volt, min_undervolt_duration, min_overvolt_duration)

## Save to file 
voltage_hourly_22.to_csv(f'/work/pi_jtaneja_umass_edu/kdonkor_umass_edu/Geospatial_Files/Geospatial_Voltage_Durations/Files/voltage_hourly_22_und_60.csv')


# ------------ # 


###### 240 min 

# undervolt
min_undervolt_duration = 240 

## Voltage hourly resample 
voltage_hourly_22 = process_hourly_voltage_data_only(test_2min_volt, min_undervolt_duration, min_overvolt_duration)

## Save to file 
voltage_hourly_22.to_csv(f'/work/pi_jtaneja_umass_edu/kdonkor_umass_edu/Geospatial_Files/Geospatial_Voltage_Durations/Files/voltage_hourly_22_und_240.csv')






# ------------ # 





###### 2023 


# Folder Path 
fold_path = '/work/pi_jtaneja_umass_edu/kdonkor_umass_edu/2023_PQR_Data/2023'


# ------------ # 

## Process 2-min voltage data 
test_2min_volt = process_data_voltage_2min_version(fold_path, min_volt_thresh, under_volt_thresh, over_volt_thresh, site_ids=list_full, buffer_seconds=45)

# ------------ # 


###### 60 min 

# undervolt
min_undervolt_duration = 60 

## Voltage hourly resample 
voltage_hourly_23 = process_hourly_voltage_data_only(test_2min_volt, min_undervolt_duration, min_overvolt_duration)

## Save to file 
voltage_hourly_23.to_csv(f'/work/pi_jtaneja_umass_edu/kdonkor_umass_edu/Geospatial_Files/Geospatial_Voltage_Durations/Files/voltage_hourly_23_und_60.csv')


# ------------ # 


###### 240 min 

# undervolt
min_undervolt_duration = 240 

## Voltage hourly resample 
voltage_hourly_23 = process_hourly_voltage_data_only(test_2min_volt, min_undervolt_duration, min_overvolt_duration)

## Save to file 
voltage_hourly_23.to_csv(f'/work/pi_jtaneja_umass_edu/kdonkor_umass_edu/Geospatial_Files/Geospatial_Voltage_Durations/Files/voltage_hourly_23_und_240.csv')