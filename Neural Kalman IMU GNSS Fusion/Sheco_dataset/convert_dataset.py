import pandas as pd

# Function to convert dataset format
def convert_dataset(input_csv, output_csv):
    # Read the input dataset
    df = pd.read_csv(input_csv)
    
    # Create a new DataFrame with required columns for the new format
    # Frame: simply an index number
    df_new = pd.DataFrame()
    df_new['Frame'] = range(1, 1 + len(df))  # Start the frame from 1597

    # Time (Seconds): derived from the timestamp (difference from the first)
    first_time = df['%time'].iloc[0]
    df_new['Time (Seconds)'] = (df['%time'] - first_time) / 1e9  # Convert nanoseconds to seconds

    # X, Z: lat (Y-axis equivalent) and lon (X-axis equivalent) to be renamed as Z and X
    df_new['X'] = df['lon']
    df_new['Z'] = df['lat']

    # IMU (s): Use the Time (Seconds) values directly
    df_new['IMU (s)'] = df_new['Time (Seconds)']

    # Ax, Ay, Az: Acceleration data (already named as ax, ay, az)
    df_new['Ax'] = df['ax']
    df_new['Ay'] = df['ay']
    df_new['Az'] = df['az']

    # Gx, Gy, Gz: Gyroscope data (already named as wx, wy, wz)
    df_new['Gx'] = df['wx']
    df_new['Gy'] = df['wy']
    df_new['Gz'] = df['wz']

    # # Mx, My, Mz: Set as 0 since the provided dataset does not have magnetometer data
    # df_new['Mx'] = 0
    # df_new['My'] = 0
    # df_new['Mz'] = 0

    # Save the new dataset to CSV
    df_new.to_csv(output_csv, index=False)

# Example usage
input_csv = 'merged_output1.csv'
output_csv = 'sheco1.csv'
convert_dataset(input_csv, output_csv)
