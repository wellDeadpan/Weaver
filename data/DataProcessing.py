import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the conditions.csv DataFrame
conditions_df = data_frames['conditions.csv']
# Get the frequency of each condition description
description_counts = conditions_df['DESCRIPTION'].value_counts().head(20)

# Display the frequency counts
print(description_counts)

# Visualize the frequency counts
plt.figure(figsize=(12, 8))
sns.barplot(x=description_counts.values, y=description_counts.index, palette='viridis')
plt.xlabel('Frequency')
plt.ylabel('Condition Description')
plt.title('Top 20 Condition Descriptions by Frequency')
plt.show()

# Add a new column 'HFYN' to indicate heart failure
conditions_df['HFYN'] = conditions_df['DESCRIPTION'].apply(lambda x: 1 if 'heart failure' in x.lower() else 0)

# Display the DataFrame with the new column
print(conditions_df.head())


# Extract the start date of the heart failure condition
heart_failure_df = conditions_df[conditions_df['HFYN'] == 1]
heart_failure_dates = heart_failure_df[['PATIENT', 'START']]

# Convert the START column to datetime using .loc to avoid SettingWithCopyWarning
heart_failure_dates = heart_failure_dates.copy()

heart_failure_dates.rename(columns={'PATIENT': 'PATIENT', 'START': 'HFSTART'}, inplace=True)

# Load other DataFrames
medications_df = data_frames['medications.csv']
observations_df = data_frames['observations.csv']
encounters_df = data_frames['encounters.csv']
procedures_df = data_frames['procedures.csv']


# Function to filter records before and after the heart failure start date
def filter_records(df, date_col, heart_failure_dates):
    filtered_records = pd.merge(df, heart_failure_dates, on='PATIENT')
    # Step 1: Convert columns to datetime if they are not already
    filtered_records[date_col] = pd.to_datetime(filtered_records[date_col])  # Ensure date_col is datetime
    filtered_records['HFSTART'] = pd.to_datetime(filtered_records['HFSTART'])  # Ensure HFSTART is datetime

    if filtered_records[date_col].dt.tz is not None or filtered_records['HFSTART'].dt.tz is not None:
        # Remove time zone information if mismatch occurs
        filtered_records[date_col] = filtered_records[date_col].dt.tz_localize(None)
        filtered_records['HFSTART'] = filtered_records['HFSTART'].dt.tz_localize(None)

    before_hf = filtered_records[filtered_records[date_col] < filtered_records['HFSTART']]
    after_hf = filtered_records[filtered_records[date_col] >= filtered_records['HFSTART']]
    print(f"Total: {len(filtered_records)}, Before HF: {len(before_hf)}, After HF: {len(after_hf)}")
    return before_hf, after_hf

# Filter records
medications_before_hf, medications_after_hf = filter_records(medications_df, 'START', heart_failure_dates)
observations_before_hf, observations_after_hf = filter_records(observations_df, 'DATE', heart_failure_dates)
encounters_before_hf, encounters_after_hf = filter_records(encounters_df, 'START', heart_failure_dates)
procedures_before_hf, procedures_after_hf = filter_records(procedures_df, 'START', heart_failure_dates)

# Display the results
print("Medications before heart failure:")
print(medications_before_hf.head())
print("Medications after heart failure:")
print(medications_after_hf.head())

print("Observations before heart failure:")
print(observations_before_hf.head())
print("Observations after heart failure:")
print(observations_after_hf.head())

print("Encounters before heart failure:")
print(encounters_before_hf.head())
print("Encounters after heart failure:")
print(encounters_after_hf.head())

print("Procedures before heart failure:")
print(procedures_before_hf.head())
print("Procedures after heart failure:")
print(procedures_after_hf.head())

encounters_df = encounters_before_hf.copy()

# Count of encounters per patient
encounter_counts = encounters_df.groupby('PATIENT').size().reset_index(name='encounter_count')

# Count by encounter type (e.g., 'ambulatory', 'inpatient')
encounter_types = pd.crosstab(encounters_df['PATIENT'], encounters_df['ENCOUNTERCLASS']).reset_index()


meds_df = medications_before_hf.copy()

# Total medications per patient
med_counts = meds_df.groupby('PATIENT').size().reset_index(name='med_count')

# Optional: cardiac meds only (filter by CODE or DESCRIPTION if possible)
cardiac_meds = meds_df[meds_df['DESCRIPTION'].str.contains('beta|ACE|diuretic', case=False, na=False)]
cardiac_med_counts = cardiac_meds.groupby('PATIENT').size().reset_index(name='cardiac_med_count')


obs_df = observations_before_hf.copy()

# Select key observations (e.g., systolic BP, glucose)
key_obs = obs_df[obs_df['DESCRIPTION'].isin(['Systolic Blood Pressure', 'Glucose'])]

# Get latest value per patient per observation type
latest_obs = key_obs.sort_values(['PATIENT', 'DATE'], ascending=[True, False]) \
                    .groupby(['PATIENT', 'DESCRIPTION'])['VALUE'].first().unstack().reset_index()
latest_obs.columns.name = None


proc_df = procedures_before_hf.copy()

# Count of procedures per patient
proc_counts = proc_df.groupby('PATIENT').size().reset_index(name='procedure_count')

# Optional: flag cardiac-related procedures
cardiac_procs = proc_df[proc_df['DESCRIPTION'].str.contains('cardiac|heart', case=False, na=False)]
cardiac_proc_flag = cardiac_procs.groupby('PATIENT').size().reset_index(name='cardiac_proc_count')


# Start with patients dataframe
features_df = patients_df[['Id']]  # assuming 'Id' is patient ID
features_df = features_df.rename(columns={'Id': 'PATIENT'})

# Merge all feature sets
features_df = features_df.merge(encounter_counts, on='PATIENT', how='left')
features_df = features_df.merge(encounter_types, on='PATIENT', how='left')
features_df = features_df.merge(med_counts, on='PATIENT', how='left')
features_df = features_df.merge(cardiac_med_counts, on='PATIENT', how='left')
features_df = features_df.merge(latest_obs, on='PATIENT', how='left')
features_df = features_df.merge(proc_counts, on='PATIENT', how='left')
features_df = features_df.merge(cardiac_proc_flag, on='PATIENT', how='left')

# Fill NA with 0 for count columns
features_df.fillna(0, inplace=True)
