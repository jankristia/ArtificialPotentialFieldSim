import csv
import os
import matplotlib.pyplot as plt


def open_csv_file():
    # Ensure 'log' directory exists
    log_dir = "log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Open CSV file for writing
    csv_path = os.path.join(log_dir, "log.csv")
    csv_file = open(csv_path, mode='w', newline='')  # Open file in write mode
    csv_writer = csv.writer(csv_file)

    # Write CSV header
    csv_writer.writerow(["Time", "Surge", "Sway", "YawRate", "LeftThrsut", "RightThrust", "DiffThrust", "CrossTrackError", "HeadingError", "Heading", "DesiredHeading"])
    simulation_time = [0]  # Time tracker

    return log_dir, csv_file, csv_writer, simulation_time

def close_and_save_csv_file(csv_file, log_dir):
    # Close the CSV file
    csv_file.close()

    # Save the trajectory plot
    plot_path = os.path.join(log_dir, "trajectory_plot.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Trajectory plot saved to {plot_path}")

    # Close the CSV file
    csv_file.close()