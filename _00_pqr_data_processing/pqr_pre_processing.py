#Import packages 

import numpy as np 
import pandas as pd 
import json 
import glob
import os
import csv
from statsmodels.tsa.seasonal import seasonal_decompose


###### 


# Process raw PQR sensor data into hourly metrics 
# Metrics -> 
# Outage events & duration per hour 
# Undervoltage events & duration per hour 


###### 

# if sensor voltage is below this, it will be considered an outage 
min_volt_thresh = 20  

# Low frequency Threshold 
freq_thresh = 30

## Normal Undervoltage Threshold (10% below) 
under_volt_thresh = 230 - (0.1 * 230)

## Very low Undervoltage Threshold (20% below) 
very_low_undervolt_thresh = 230 - (0.2 * 230) 

## Normal Overvoltage Threshold (10% above) 
over_volt_thresh = 230 + (0.1 * 230)

## Very high overvoltage threshold (20% above)
very_high_overvolt_thresh = 230 + (0.2 * 230) 

#### SPECIFY DURATIONS 

# minimum outage duration 
min_outage_duration = 5

# undervolt, very_low_undervolt
min_undervolt_duration = 5

# overvolt, very_high_overvolt
min_overvolt_duration = 2


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


## process the raw sensor into 2 minute data 

def process_data_outage_2min_version(folder_path, min_volt_thresh, freq_thresh, site_ids=None, buffer_seconds=45):
    # Read and concatenate CSVs
    print("Concatenating files in folder")
    df = read_and_concatenate_csvs(folder_path)
    
    print("Cleaning up merged folder files")
    # Clean up the data
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
                print(f"****Site {site_id} not found or no data available. Skipping...****")
                continue

            #### OUTAGE SECTION #### 
            
            print(f"Resampling for site {site_id}")
            # Perform resampling to 2-min 
            df_outage_2_min = resample_voltage_dynamic_v2(site_df, min_volt_thresh, freq_thresh, buffer_seconds)
            
            #### APPEND TO LIST #### 
            all_stats.append(df_outage_2_min)
    
    else:
        # If no specific sites are provided 
        print("****No site list provided...****")

    # Concatenate results if there are any
    if all_stats:
        print("Concatenating 2-min data for all sites")
        final_df = pd.concat(all_stats, ignore_index=True)
    else:
        final_df = pd.DataFrame()  # Return an empty DataFrame if no data is processed
    
    return final_df


## resample 2 minute data to hourly outage data 

def process_hourly_outage_data_only(df_outage_2_min, min_outage_duration):

    all_stats = []

    # run this for each unique site_id 
    print("Extracting unique site_ids from 2-min file")
    site_ids = df_outage_2_min['site_id'].unique()

    for site_id in site_ids:
        
        site_df = df_outage_2_min[df_outage_2_min['site_id'] == site_id]

        # run this for each site 
        print(f"Perfoming 1 hour resampling for site {site_id}")
        outage_hour_resampled = resample_hourly_outages_only(site_df, min_outage_duration)

        # append to list 
        all_stats.append(outage_hour_resampled)

    ## Concatenate dfs for all sites 
    print("Concatenating all 1 hourly resampled outage data")
    final_df = pd.concat(all_stats, ignore_index=True)

    return final_df


## helper function for hourly outage resampling  (embedded in the previous function)

def resample_hourly_outages_only(
    df,
    min_outage_duration=2
):
    # Ensure time is in the columns
    if 'time' not in df.columns:
        df = df.reset_index()

    # Base hourly metadata (site/location info)
    hourly_df = df.resample('1h', on='time').agg({
        'site_id': 'first',
    }).reset_index()

    # Calculate outage frequency using updated function with respondents
    outage_events_df, num_outages, total_outage_duration = calculate_outage_frequency_with_respondents(
        df, min_outage_duration=min_outage_duration, return_full_df=False
    )

    # Take first record per outage per hour
    outage_events_first_per_hour = outage_events_df.drop_duplicates(subset=['outage_group', 'time'])

    # Total outage minutes per hour
    outage_df = outage_events_first_per_hour.resample('1h', on='time')['outage_duration_mins'].sum().rename('outage_mins')

    # Number of unique outage events per hour
    num_outage_events_df = outage_events_first_per_hour.resample('1h', on='time')['outage_group'].nunique().rename('outage_events')

    # Merge into single DataFrame
    hourly_df = hourly_df.merge(num_outage_events_df, on='time', how='left')
    hourly_df = hourly_df.merge(outage_df, on='time', how='left')

    # Fill NA values with 0 (no events/respondents)
    hourly_df = hourly_df.fillna(0)

    return hourly_df


## helper function for hourly outage resampling as well  (embedded in the previous function)
# specifies minimum duration that sensors DON'T have power to be considered an outage (5 min)

# ignores momentary blips in outage restoration 
# power has to be back for at least 5 minutes before it is considered a legitimate outage restoration 


def calculate_outage_frequency_with_respondents(df, min_outage_duration=2, non_momentary_duration=5, return_full_df=False):
    df = df.copy()
    df = df.sort_values('time').reset_index(drop=True)

    # Step 1: Identify raw outage points (voltage == 0 and respondents > 1)
    df['is_outage_raw'] = (df['voltage'] == 0) & (df['num_unique_respondents'] > 1)

    # Step 2: Label contiguous segments (outage or not)
    df['group'] = (df['is_outage_raw'] != df['is_outage_raw'].shift(1)).cumsum()

    # Step 3: Calculate duration per group (in minutes, assuming 2 minutes per row)
    group_sizes = df.groupby('group').size().mul(2)  # in minutes
    group_status = df.groupby('group')['is_outage_raw'].first()

    # Step 4: Smooth short non-outage gaps (e.g., 2 or 4 mins) between real outages
    smooth_status = group_status.copy()
    for i in range(1, len(group_status) - 1):
        if (
            not group_status.iloc[i] and
            group_status.iloc[i - 1] and group_status.iloc[i + 1] and
            group_sizes.iloc[i] <= non_momentary_duration
        ):
            smooth_status.iloc[i] = True  # treat as outage too

    # Step 5: Re-map to DataFrame
    df['is_outage'] = df['group'].map(smooth_status)

    # Step 6: Detect outage starts
    df['outage_start'] = df['is_outage'] & ~df['is_outage'].shift(1, fill_value=False)

    # Step 7: Assign unique outage group IDs
    df['outage_group'] = df['outage_start'].cumsum() * df['is_outage']

    # Step 8: Compute durations of each group
    event_durations = df.groupby('outage_group')['is_outage'].sum().mul(2)  # in minutes
    valid_events = event_durations[event_durations >= min_outage_duration].index

    # Step 9: Filter and summarize outage events
    outage_events = df[df['outage_group'].isin(valid_events)]
    outage_events = outage_events.groupby('outage_group').first().reset_index()

    # Step 10: Add outage duration info
    outage_events = outage_events[['time', 'voltage', 'num_unique_respondents', 'outage_group']].merge(
        event_durations.rename('outage_duration_mins'), left_on='outage_group', right_index=True
    )

    # Step 11: Return results
    num_outage_events = len(valid_events)
    total_outage_duration = event_durations[valid_events].sum()

    if return_full_df:
        return df
    else:
        return outage_events, num_outage_events, total_outage_duration



#### Helper function used in processing of sensor data for OUTAGE metrics 
    
def process_group_dynamic(group, min_volt_thresh, freq_thresh):
    # Convert to DataFrame
    group_df = pd.DataFrame(group)

    # Extract metadata
    site_id = group_df['site_id'].iloc[0]
    district = group_df['district'].iloc[0]
    region = group_df['region'].iloc[0]
    lat = group_df['site_latitude'].iloc[0]
    lon = group_df['site_longitude'].iloc[0]
    num_unique_respondents = group_df['respondent_id'].nunique()

    # --- Outage Check ---
    # Remove the voltage check: Now only checking if `is_powered == False`
    low_voltage_unpowered = group_df[
        (group_df['is_powered'] == False)
    ]['respondent_id'].nunique()

    # If 2 or more unique sensors are unpowered
    if low_voltage_unpowered >= 2:
        return {
            'site_id': site_id,
            'district': district,
            'region': region,
            'lat': lat,
            'lon': lon,
            'time': group_df['time'].min(),
            'voltage': 0,
            'frequency': 0,
            'num_unique_respondents': low_voltage_unpowered
        }
    else:
        # --- Normal Filtering ---
        # Filter for valid voltage and frequency
        valid_data_df = group_df[
            (group_df['voltage'] > min_volt_thresh) & 
            (group_df['frequency'] > freq_thresh) &
            (group_df['is_powered'] == True)
        ]

        # If there is valid data, calculate the average voltage and frequency
        if not valid_data_df.empty:
            return {
                'site_id': site_id,
                'district': district,
                'region': region,
                'lat': lat,
                'lon': lon,
                'time': group_df['time'].min(),
                'voltage': valid_data_df['voltage'].mean(),
                'frequency': valid_data_df['frequency'].mean(),
                'num_unique_respondents': valid_data_df['respondent_id'].nunique()
            }
        else:
            # If no valid data, return 0 for voltage and frequency
            return {
                'site_id': site_id,
                'district': district,
                'region': region,
                'lat': lat,
                'lon': lon,
                'time': group_df['time'].min(),
                'voltage': 0,
                'frequency': 0,
                'num_unique_respondents': num_unique_respondents
            }

        

## Helper function used in main 2-min OUTAGE processing function 

        
def resample_voltage_dynamic_v2(sample_df, min_volt_thresh, freq_thresh, buffer_seconds=45):
 
    # Initialize variables
    result_data = []
    current_group = []

    # Iterate through the DataFrame
    for _, row in sample_df.iterrows():
        if not current_group:
            current_group.append(row)
            continue
        
        # Calculate time difference from the last row instead of the first row
        time_diff = (row['time'] - current_group[-1]['time']).total_seconds()

        if time_diff <= buffer_seconds:
            current_group.append(row)
        else:
            # Process the current group using dynamic thresholding
            result_data.append(process_group_dynamic(current_group, min_volt_thresh, freq_thresh))
            current_group = [row]

    # Process the last group
    if current_group:
        result_data.append(process_group_dynamic(current_group, min_volt_thresh, freq_thresh))

    # Convert to DataFrame & reorder the columns
    result_data_df = pd.DataFrame(result_data)
    result_data_df = result_data_df[['time', 'site_id', 'lat', 'lon', 'district', 'region', 'voltage', 'frequency', 'num_unique_respondents']]

    return result_data_df



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



### Calculating consecutive outage metrics of specified durations 

def print_outage_num_n_duration_month_hour(df, list_durations):
    results = []

    for site_id, site_df in df.groupby('site_id'):
        for dur in list_durations:
            print(f"Calculating consecutive outage metrics for site {site_id} - {dur} mins")

            # Resample outages
            df_hourly = resample_hourly_outages_only(site_df, min_outage_duration=dur)

            if df_hourly is None or df_hourly.empty:
                continue

            # Add month and hour columns
            df_hourly['month'] = df_hourly['time'].dt.month
            df_hourly['hour'] = df_hourly['time'].dt.hour

            # Aggregate total events and durations
            num_outages = df_hourly['outage_events'].sum()
            outage_mins = df_hourly['outage_mins'].sum()
            outage_hours = outage_mins / 60

            # Monthly outage counts (aggregated across all years)
            monthly_outage_counts = df_hourly.groupby('month')['outage_events'].sum()
            monthly_counts_dict = {f"Month_{month}": monthly_outage_counts.get(month, 0) for month in range(1, 13)}

            # Hourly outage counts
            hourly_outage_counts = df_hourly.groupby('hour')['outage_events'].sum()
            hourly_counts_dict = {f"Hour_{hour}": hourly_outage_counts.get(hour, 0) for hour in range(24)}

            # Build result entry
            result_entry = {
                "Site_ID": site_id,
                "Min_outage_dur_mins": dur,
                "Min_outage_dur_hours": round(dur / 60, 2),
                "Total_num_outages": num_outages,
                "Total_outage_dur_hours": round(outage_hours, 2)
            }

            result_entry.update(monthly_counts_dict)
            result_entry.update(hourly_counts_dict)

            print("Concatenating all sites")
            results.append(result_entry)

    return pd.DataFrame(results)




### Calculating consecutive undervoltage metrics of specified durations 

def print_undervoltages_num_n_duration_revised_month_hour(df, list_durations):
    results = []

    for site_id, site_df in df.groupby('site_id'):
        for dur in list_durations:
            print(f"Calculating consecutive undervoltage metrics for site {site_id} - {dur} mins")

            # Undervolt events per hour (per sensor)
            sensor_volts_per_hour = summarize_voltage_events_per_hour_v2(
                site_df, min_undervolt_duration=dur, min_overvolt_duration=2
            )

            # Average across sensors
            df_hourly = average_voltage_events_all_sensors_v2(sensor_volts_per_hour)

            if df_hourly is None or df_hourly.empty:
                continue

            # Add month and hour columns
            df_hourly['month'] = df_hourly['time'].dt.month
            df_hourly['hour'] = df_hourly['time'].dt.hour

            # Monthly undervolt count (aggregated across all years)
            monthly_undervolt_counts = df_hourly.groupby("month")['total_undervolt_events'].sum()
            monthly_counts_dict = {
                f"Month_{month}": monthly_undervolt_counts.get(month, 0) for month in range(1, 13)
            }

            # Hourly undervolt count
            hourly_undervolt_counts = df_hourly.groupby("hour")['total_undervolt_events'].sum()
            hourly_counts_dict = {
                f"Hour_{hour}": hourly_undervolt_counts.get(hour, 0) for hour in range(24)
            }

            # Aggregate total events & durations
            num_undervolts = df_hourly['total_undervolt_events'].sum()
            undervolt_mins = df_hourly['total_undervolt_duration'].sum()
            undervolt_hours = undervolt_mins / 60

            # Create result entry
            result_entry = {
                "Site_ID": site_id,
                "Min_undervolt_dur_mins": dur,
                "Min_undervolt_dur_hours": round(dur / 60, 2),
                "Total_num_undervolts": num_undervolts,
                "Total_undervolt_dur_hours": round(undervolt_hours, 2)
            }

            result_entry.update(monthly_counts_dict)
            result_entry.update(hourly_counts_dict)

            print("Concatenating all sites")
            results.append(result_entry)

    return pd.DataFrame(results)


####### 


### List of ALL sites 

full_site_df = pd.read_csv('/work/pi_jtaneja_umass_edu/kdonkor_umass_edu/common_sites_merged_df_final.csv')
list_full = list(np.sort(full_site_df['site_id'].unique()))


###### 2022 

## specify folder path with PQR sensor data 
fold_path = '/work/pi_jtaneja_umass_edu/kdonkor_umass_edu/2022_PQR_Data/2022'

######

## Process 2-min OUTAGE data 
test_2min = process_data_outage_2min_version(fold_path, min_volt_thresh, freq_thresh, site_ids=list_full, buffer_seconds=45)

## Process 2-min VOLTAGE data 
test_2min_volt = process_data_voltage_2min_version(fold_path, min_volt_thresh, under_volt_thresh, over_volt_thresh, site_ids=list_full, buffer_seconds=45)

## Outage hourly resample 
outage_hourly_22 = process_hourly_outage_data_only(test_2min, min_outage_duration)

## Voltage hourly resample 
voltage_hourly_22 = process_hourly_voltage_data_only(test_2min_volt, min_undervolt_duration, min_overvolt_duration)

## Merge Outage & Voltage 
merged_outage_n_voltage_hourly_22 = process_merge_voltage_n_outage_hourly(outage_hourly_22, voltage_hourly_22)


year = merged_outage_n_voltage_hourly_22['time'].dt.year.unique()[0]

## Save to file 
merged_outage_n_voltage_hourly_22.to_csv(f'/work/pi_jtaneja_umass_edu/kdonkor_umass_edu/Geospatial_Files/{year}/merged_outage_n_voltage_hourly_22_NEW.csv')


###### -------- ####### 


## Calculating consecutive metrics (outages & undervoltages)


# 5min, 1hr, 2hr, 4hr, 6hr, 8hr, 10hr, 12hr, 14hr, 16hr, 20hr, 24hr  
list_dur_outages = [5, 60, 120, 240, 360, 480, 600, 720, 840, 960, 1200, 1440] 

# 5min, 20min, 40min, 1hr, 1.5hr, 2hr, 4hr, 8hr 
list_dur_voltages = [5, 20, 40, 60, 90, 120, 240, 480]

## Consecutive Outage 
consecutive_outages_22 = print_outage_num_n_duration_month_hour(test_2min, list_dur_outages)

## Consecutive Undervoltages 
consecutive_undervolts_22 = print_undervoltages_num_n_duration_revised_month_hour(test_2min_volt, list_dur_voltages)

### Save to file 
consecutive_outages_22.to_csv(f'/work/pi_jtaneja_umass_edu/kdonkor_umass_edu/Geospatial_Files/{year}/consecutive_outages_22_NEW.csv')
consecutive_undervolts_22.to_csv(f'/work/pi_jtaneja_umass_edu/kdonkor_umass_edu/Geospatial_Files/{year}/consecutive_undervolts_22_NEW.csv')





###### -------- ####### 






###### 2023 

## specify folder path with PQR sensor data 
fold_path = '/work/pi_jtaneja_umass_edu/kdonkor_umass_edu/2023_PQR_Data/2023'

######

## Process 2-min OUTAGE data 
test_2min = process_data_outage_2min_version(fold_path, min_volt_thresh, freq_thresh, site_ids=list_full, buffer_seconds=45)

## Process 2-min VOLTAGE data 
test_2min_volt = process_data_voltage_2min_version(fold_path, min_volt_thresh, under_volt_thresh, over_volt_thresh, site_ids=list_full, buffer_seconds=45)

## Outage hourly resample 
outage_hourly_23 = process_hourly_outage_data_only(test_2min, min_outage_duration)

## Voltage hourly resample 
voltage_hourly_23 = process_hourly_voltage_data_only(test_2min_volt, min_undervolt_duration, min_overvolt_duration)

## Merge Outage & Voltage 
merged_outage_n_voltage_hourly_23 = process_merge_voltage_n_outage_hourly(outage_hourly_23, voltage_hourly_23)


year = merged_outage_n_voltage_hourly_23['time'].dt.year.unique()[0]

## Save to file 
merged_outage_n_voltage_hourly_23.to_csv(f'/work/pi_jtaneja_umass_edu/kdonkor_umass_edu/Geospatial_Files/{year}/merged_outage_n_voltage_hourly_23_NEW.csv')


###### -------- ####### 


## Calculating consecutive metrics (outages & undervoltages)

## Consecutive Outage 
consecutive_outages_23 = print_outage_num_n_duration_month_hour(test_2min, list_dur_outages)

## Consecutive Undervoltages 
consecutive_undervolts_23 = print_undervoltages_num_n_duration_revised_month_hour(test_2min_volt, list_dur_voltages)

### Save to file 
consecutive_outages_23.to_csv(f'/work/pi_jtaneja_umass_edu/kdonkor_umass_edu/Geospatial_Files/{year}/consecutive_outages_23_NEW.csv')
consecutive_undervolts_23.to_csv(f'/work/pi_jtaneja_umass_edu/kdonkor_umass_edu/Geospatial_Files/{year}/consecutive_undervolts_23_NEW.csv')