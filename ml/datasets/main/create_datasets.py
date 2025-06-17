import os
import sys
import argparse
import logging
import time
import pandas as pd
from datetime import datetime

# 相対インポートを使用
from steps.step1_nc_to_csv import convert_all_nc_to_csv
from steps.step2_raw_to_complete import process_csv_files
from steps.step3_flare_classification import process_all_days_for_flare_class
from steps.step4_create_hdf5 import create_hourly_hdf5_datasets

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Directory paths
# aia_base_dir = "./flarebench_dataset/solar_images/aia/"
# hmi_base_dir = "./flarebench_dataset/solar_images/hmi/"
# xrs_base_dir = "./flarebench_dataset/xrs/"

# Date ranges
# start_date = pd.Timestamp("2023-11-30")
# start_date = pd.Timestamp("2024-04-30")
# end_date = pd.Timestamp("2024-05-01")
start_date = pd.Timestamp('2011-06-01')
end_date = pd.Timestamp('2024-05-31')

# Define AIA wavelengths (excluding 1700)
aia_wavelengths = [
    "0094",
    "0131",
    "0171",
    "0193",
    "0211",
    "0304",
    "0335",
    "1600",
    "4500",
]


def run_pipeline(args):
    """Run the complete pipeline"""
    start_time = time.time()

    # ディレクトリパスを設定
    data_root = args.data_root
    aia_base_dir = os.path.join(data_root, "aia/")
    hmi_base_dir = os.path.join(data_root, "hmi/")
    xrs_base_dir = os.path.join(data_root, "xrs/")
    output_dir = args.output_dir

    # Step 1: Convert NetCDF to CSV
    if args.step <= 1:
        logging.info("Step 1: Converting NetCDF files to CSV")
        convert_all_nc_to_csv(xrs_base_dir, args.start_year, args.end_year)

    # Step 2: Process CSV files to create complete day data
    if args.step <= 2:
        logging.info("Step 2: Processing CSV files to create complete day data")
        process_csv_files(xrs_base_dir)

    # Step 3: Process days for flare classification
    if args.step <= 3:
        logging.info("Step 3: Processing days for flare classification")
        process_all_days_for_flare_class(xrs_base_dir, args.start_year, args.end_year)

    # Step 4: Create datasets
    if args.step <= 4:
        logging.info("Step 4: Creating datasets")
        create_hourly_hdf5_datasets(
            aia_base_dir=aia_base_dir,
            hmi_base_dir=hmi_base_dir,
            xrs_base_dir=xrs_base_dir,
            output_dir=output_dir,
            start_date=start_date,
            end_date=end_date,
            aia_wavelengths=aia_wavelengths,
            mode=args.mode,
            vis_dir=args.vis_dir,
            num_workers=args.workers,
        )

    end_time = time.time()
    logging.info(f"Pipeline completed in {end_time - start_time:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(
        description="Create solar observation datasets with integrated XRS processing"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./flarebench_dataset",
        help="Root directory for input data files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./flarebench_dataset/all_data_hours/",
        help="Directory for output dataset files",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["create", "visualize"],
        default="create",
        help="Mode of operation: create datasets or visualize samples",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Number of worker processes for parallel processing",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=4,
        choices=[1, 2, 3, 4],
        help="Start from step (1: NetCDF to CSV, 2: Raw CSV to Complete, 3: Flare Classification, 4: Create Datasets)",
    )
    parser.add_argument(
        "--start_year",
        type=int,
        default=2011,
        help="Start year for processing",
    )
    parser.add_argument(
        "--end_year",
        type=int,
        default=2023,
        help="End year for processing",
    )
    parser.add_argument(
        "--vis_dir",
        type=str,
        default=os.path.join("results", "dataset_samples"),
        help="Directory for visualization results",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (process only a few files)",
    )
    args = parser.parse_args()

    if args.debug:
        logging.info("Running in debug mode")
        # Limit processing in debug mode
        args.start_year = 2017
        args.end_year = 2017

    run_pipeline(args)


if __name__ == "__main__":
    main()
