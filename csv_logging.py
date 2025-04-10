import csv
import os
import matplotlib.pyplot as plt


def open_csv_file():
    log_dir = "log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    csv_path = os.path.join(log_dir, "log.csv")
    csv_file = open(csv_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)

    csv_writer.writerow(["Time", "Surge", "Sway", "YawRate", "LeftPWM", "RightPWM", "DiffPWM", "CrossTrackError", "HeadingError", "Heading", "ColAvDesiredHeading", "LOSDesiredHeading", "ShortestObjectDist"])
    simulation_time = [0]  # Time tracker

    return log_dir, csv_file, csv_writer, simulation_time

def close_and_save_csv_file(csv_file, log_dir):
    csv_file.close()

    plot_path = os.path.join(log_dir, "trajectory_plot.pdf")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Trajectory plot saved to {plot_path}")

    csv_file.close()