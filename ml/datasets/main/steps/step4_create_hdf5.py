import os
import numpy as np
import pandas as pd
import h5py
import cv2
from skimage.io import imread
from astropy.io import fits
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import matplotlib.pyplot as plt
from astropy.visualization import ImageNormalize, AsinhStretch

# AIA and HMI processing parameters
AIA_VERTICAL_CROP = 20
AIA_HORIZONTAL_CROP = 15
HMI_TEXT_HEIGHT = 10
HMI_TEXT_WIDTH = 70

def read_fits(file_path):
    """Read FITS file and return data"""
    with fits.open(file_path) as hdul:
        data = hdul[1].data
    return data

def read_jpeg(file_path):
    """Read JPEG file and return data"""
    return imread(file_path, as_gray=True)

def process_aia_data(data):
    """Process AIA data"""
    # Crop the image to remove text
    h, w = data.shape
    data = data[AIA_VERTICAL_CROP:h-AIA_VERTICAL_CROP, AIA_HORIZONTAL_CROP:w-AIA_HORIZONTAL_CROP]
    
    # Resize to 256x256
    data = cv2.resize(data, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    
    # Convert to float32
    return data.astype(np.float32)

def process_hmi_data(data):
    """Process HMI data"""
    # Crop the image to remove text
    h, w = data.shape
    data = data[HMI_TEXT_HEIGHT:, HMI_TEXT_WIDTH:]
    
    # Resize to 256x256
    data = cv2.resize(data, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    
    # Convert to float32
    return data.astype(np.float32)

def is_missing_data(data, threshold=0.01):
    """Check if data is missing (all zeros or very low values)"""
    return np.mean(data) < threshold

def find_past_similar_file(timestamp, processed_files):
    """Find past similar file for missing data"""
    if not processed_files:
        return None
    
    # Sort by time difference
    sorted_files = sorted(processed_files.items(), key=lambda x: abs((x[0] - timestamp).total_seconds()))
    return processed_files[sorted_files[0][0]]

def save_sample_images(X, timestamp, output_dir):
    """Save sample images"""
    sample_dir = os.path.join(
        output_dir, "visualization", timestamp.strftime("%Y%m%d_%H%M%S")
    )
    os.makedirs(sample_dir, exist_ok=True)

    # Define AIA wavelengths and corresponding colormaps
    aia_wavelengths_vis = ["94", "131", "171", "193", "211", "304", "335", "1600", "4500"]

    # Save images for each channel
    for i in range(10):
        plt.figure(figsize=(8, 8))

        if i < 9:
            # Visualize AIA channels
            wavelength = aia_wavelengths_vis[i]
            title = f"AIA {wavelength} Å"

            if wavelength in ["1600", "4500"]:
                cmap = plt.get_cmap("sdoaia4500")
            else:
                cmap = plt.get_cmap(f"sdoaia{wavelength}")

            # Apply normalization and stretch
            norm = ImageNormalize(X[i], stretch=AsinhStretch(0.005), clip=True)

            plt.imshow(X[i], cmap=cmap, norm=norm)

        else:
            # Visualize HMI
            title = "HMI Grayscale"
            plt.imshow(X[i], cmap="gray")

        plt.title(title)
        plt.colorbar()
        plt.axis("off")

        # Save to sample folder
        save_path = os.path.join(
            sample_dir, f'channel_{i:02d}_{title.replace(" ", "_")}.png'
        )
        plt.savefig(save_path, bbox_inches="tight", dpi=100)
        plt.close()

    # Create README in sample folder
    with open(os.path.join(sample_dir, "README.txt"), "w") as f:
        f.write("Channel Information:\n")
        for i in range(9):
            f.write(f"Channel {i:02d}: AIA {aia_wavelengths_vis[i]} Å\n")
        f.write("Channel 09: HMI Grayscale\n")

def process_hour(
    year, month, day, hour, 
    aia_base_dir, hmi_base_dir, xrs_base_dir, 
    aia_wavelengths, processed_files, 
    visualize=False, vis_dir=None
):
    """Process one hour of data"""
    timestamp = pd.Timestamp(year=year, month=month, day=day, hour=hour)

    # Process XRS data - updated to use 24-hour forward window
    xrs_file_path = os.path.join(
        xrs_base_dir, f"{year}/{month:02d}/complete_{year}_{month:02d}_{day:02d}.csv"
    )
    
    try:
        # Load XRS data
        xrs_df = pd.read_csv(xrs_file_path)
        xrs_df["time"] = pd.to_datetime(xrs_df["time"])
        
        # Find the current timestamp in the data
        current_time = timestamp
        current_row = xrs_df[xrs_df["time"] == current_time]
        
        if current_row.empty:
            y_label = 0  # Missing data
        else:
            y_label = current_row["flare_class"].values[0]
            
    except Exception as e:
        logging.error(f"Error processing XRS data for {timestamp}: {e}")
        y_label = 0

    # Process AIA data
    X_data = []
    for wavelength in aia_wavelengths:
        aia_file = os.path.join(
            aia_base_dir, f"{year}/{month:02d}/{day:02d}/{wavelength}/{hour:02d}00.fits"
        )
        try:
            aia_data = read_fits(aia_file)
            aia_data = process_aia_data(aia_data)
            X_data.append(aia_data)
        except Exception as e:
            logging.error(f"Error processing AIA data for {wavelength}: {e}")
            X_data.append(np.zeros((256, 256), dtype=np.float32))

    # Process HMI data
    hmi_file = os.path.join(
        hmi_base_dir, f"{year}/{month:02d}/{day:02d}/{hour:02d}00.jpg"
    )
    try:
        hmi_data = read_jpeg(hmi_file)
        hmi_processed = process_hmi_data(hmi_data)

        # If HMI data is missing, supplement with past data
        if is_missing_data(hmi_processed):
            past_data = find_past_similar_file(timestamp, processed_files)
            if past_data is not None:
                hmi_processed = past_data[-1]  # Last channel is HMI
    except Exception as e:
        logging.error(f"Error processing HMI data: {e}")
        hmi_processed = np.zeros((256, 256), dtype=np.float32)

    # Combine all channels
    X_combined = np.stack(X_data + [hmi_processed], axis=0)

    # In visualization mode
    if visualize and vis_dir is not None:
        save_sample_images(X_combined, timestamp, vis_dir)

    return X_combined, y_label, str(timestamp)

def process_date(date, aia_base_dir, hmi_base_dir, xrs_base_dir, output_dir, aia_wavelengths, mode="create", vis_dir=None):
    """Function to process one hour of data"""
    try:
        X, y, _ = process_hour(
            date.year,
            date.month,
            date.day,
            date.hour,
            aia_base_dir, 
            hmi_base_dir, 
            xrs_base_dir,
            aia_wavelengths,
            {},  # processed_files not used in parallel processing
            visualize=(mode == "visualize"),
            vis_dir=vis_dir,
        )

        # File name format: YYYYMMDD_HHMMSS.h5
        timestamp_str = date.strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"{timestamp_str}.h5")

        with h5py.File(output_file, "w") as f:
            f.create_dataset("X", data=X, compression="gzip", compression_opts=9)
            f.create_dataset("y", data=y)
            f.create_dataset("timestamp", data=timestamp_str)

        return True
    except Exception as e:
        logging.error(f"Error processing {date}: {e}")
        return False

def create_hourly_hdf5_datasets(
    aia_base_dir, hmi_base_dir, xrs_base_dir, output_dir,
    start_date, end_date, aia_wavelengths,
    mode="create", vis_dir=None, num_workers=os.cpu_count()
):
    """Create datasets (parallel processing version)"""
    os.makedirs(output_dir, exist_ok=True)

    # Create list of dates to process
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        if mode == "visualize":
            # In visualize mode, 50 samples every 1000 hours
            date_list.append(current_date)
            current_date += pd.Timedelta(hours=1000)
            if len(date_list) >= 50:
                break
        else:
            # In create mode, every hour
            date_list.append(current_date)
            current_date += pd.Timedelta(hours=1)

    # Setup progress bar
    total = len(date_list)
    processed = 0
    pbar = tqdm(total=total, desc="Processing data")

    # Execute parallel processing
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit jobs
        futures = [
            executor.submit(
                process_date, 
                date, 
                aia_base_dir, 
                hmi_base_dir, 
                xrs_base_dir, 
                output_dir, 
                aia_wavelengths,
                mode, 
                vis_dir
            ) for date in date_list
        ]

        # Collect results
        for future in as_completed(futures):
            try:
                success = future.result()
                if success:
                    processed += 1
                    pbar.update(1)
            except Exception as e:
                logging.error(f"Error in parallel processing: {e}")

    pbar.close()
    logging.info(f"Processed {processed}/{total} files successfully")
    return processed