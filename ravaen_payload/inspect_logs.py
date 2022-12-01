import json
import numpy as np


# def plot_measurements(measurements, titles):
#     import matplotlib.pyplot as plt
#     width = 0.35  # the width of the bars
#     fig, ax = plt.subplots()
#     n = len(measurements)
#
#     for i in range(len(measurements)):
#         data = measurements[i]
#         title = titles[i]
#         x = np.arange(len(data))
#
#     N = 3
#     ind = np.arange(N)
#     width = 0.25
#
#     xvals = [8, 9, 2]
#     bar1 = plt.bar(ind, xvals, width, color = 'r')
#
#     yvals = [10, 20, 30]
#     bar2 = plt.bar(ind+width, yvals, width, color='g')
#
#     zvals = [11, 12, 13]
#     bar3 = plt.bar(ind+width*2, zvals, width, color = 'b')
#
#     plt.xlabel("Dates")
#     plt.ylabel('Scores')
#     plt.title("Players Score")
#
#     plt.xticks(ind+width,['2021Feb01', '2021Feb02', '2021Feb03'])
#     plt.legend( (bar1, bar2, bar3), ('Player1', 'Player2', 'Player3') )
#     plt.show()
#

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

    print("Used args:",args)
    print("Measured times:",times)


    times_with_io = check_statistic(times, "total_encode_compare_with_IO")
    times_without_io = check_statistic(times, "total_encode_compare")
    times_one_encode = check_statistic(times, "first_full_batch_encode")
    times_one_compare = check_statistic(times, "first_full_batch_compare")

    # plot_measurements([times_with_io, times_without_io], ["Processing time with IO", "Processing time without IO"])
    plot_measurements([times_with_io, times_without_io,times_one_encode,times_one_compare],
                      ["Processing time with IO", "Processing time without IO", "One batch encode", "One batch compare"])

if __name__ == "__main__":
    log_path = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/results_logs/results_1strun/logs.json"
    log_path = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/results_logs/results_2ndrun_both/logs.json"
    inspect_logs(log_path)

"""
Available times:

time_file_000_first_full_batch_encode
time_file_000_first_full_batch_compare
time_file_000_total_encode_compare
time_file_000_total_encode_compare_with_IO
time_file_001_first_full_batch_encode
time_file_001_first_full_batch_compare
time_file_001_total_encode_compare
time_file_001_total_encode_compare_with_IO
time_file_002_first_full_batch_encode
time_file_002_first_full_batch_compare
time_file_002_total_encode_compare
time_file_002_total_encode_compare_with_IO


"""