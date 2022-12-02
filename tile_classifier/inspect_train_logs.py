import json
import numpy as np


def plot_measurements(measurements, titles):
    import matplotlib.pyplot as plt

    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    n = len(measurements)

    for i in range(len(measurements)):
        data = measurements[i]
        title = titles[i]
        x = np.arange(len(data))

        rects = ax.bar(x - (2*i*width/n), data, width, label=title)
        ax.bar_label(rects, padding=3)


    ax.set_xticks(x, x)
    ax.legend()

    fig.tight_layout()

    plt.show()

def check_statistic(times, check = "total_encode_compare_with_IO"):
    relevant_keys = [k for k in times.keys() if k.endswith(check)]
    relevant_keys.sort()

    print(relevant_keys)
    measurements = []
    for k in relevant_keys:
        measurement = times[k]
        measurements.append(measurement)
    measurements = np.asarray(measurements)
    return measurements

def inspect_logs(log_path = "../results/logs.json"):
    # Opening JSON file
    with open(log_path) as json_file:
        log_data = json.load(json_file)

    keys = log_data.keys()
    args_keys = [k for k in keys if "args_" in k]
    time_keys = [k for k in keys if "time_" in k]

    args = {k: log_data[k] for k in args_keys}
    times = {k: log_data[k] for k in time_keys}

    for k in times:
        print(k)

    print("Used args:",args)
    print("Measured times:",times)


    times_traininig_epochs = check_statistic(times, "_full")

    plot_measurements([times_traininig_epochs],
                      ["Training time per epoch"])

if __name__ == "__main__":
    log_path = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/logs_unibap/results03_withbatchsizes/tile_classifier_log_64batch.json"
    inspect_logs(log_path)

"""
Available times:

time_dataset_load
time_model_create
time_epoch_000_full
time_epoch_001_full
time_epoch_002_full
time_epoch_003_full
time_epoch_004_full
time_epoch_005_full
time_epoch_006_full
time_epoch_007_full
time_epoch_008_full
time_epoch_009_full
time_save_model
time_one_batch_prediction


"""