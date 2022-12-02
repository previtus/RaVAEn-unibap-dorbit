import json
import numpy as np

from matplotlib import pyplot as plt


def custom_bar_plot(ax, data, std_data=None, colors=None, total_width=0.8, single_width=1, legend=True):
    # source: https://stackoverflow.com/questions/14270391/python-matplotlib-multiple-bars
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    if std_data is not None:
        std_values_i_x = {}
        for i, (name, values) in enumerate(std_data.items()):
            std_values_i_x[i] = {}
            for x, y in enumerate(values):
                std_values_i_x[i][x] = y

    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            yerr = None
            if std_data is not None: yerr = std_values_i_x[i][x]
            bar = ax.bar(x + x_offset, y, yerr=yerr, width=bar_width * single_width, color=colors[i % len(colors)])

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys())



def plot_measurements(measurements, titles, plot_title=""):
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
    if len(plot_title) > 0:
        plt.title(plot_title)
    plt.show()

def check_statistic(times, check = "total_encode_compare_with_IO", ignore_file_i_above=None):
    relevant_keys = [k for k in times.keys() if k.endswith(check)]
    relevant_keys.sort()

    # print(relevant_keys)
    measurements = []
    for k in relevant_keys:
        if ignore_file_i_above is not None:
            # _file_000_
            file_i = int( k.split("_file_")[-1].split("_")[0] )
            if file_i > ignore_file_i_above:
                continue

        measurement = times[k]
        measurements.append(measurement)
    measurements = np.asarray(measurements)
    return measurements

def load_logs(log_path = "../results/logs.json"):
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
    return args, times

def plot_all_files(log_path = "../results/logs.json", ignore_file_i_above=None):
    args, times = load_logs(log_path)

    times_with_io = check_statistic(times, "total_encode_compare_with_IO", ignore_file_i_above)
    times_without_io = check_statistic(times, "total_encode_compare", ignore_file_i_above)
    times_one_encode = check_statistic(times, "first_full_batch_encode", ignore_file_i_above)
    times_one_compare = check_statistic(times, "first_full_batch_compare", ignore_file_i_above)

    # plot_measurements([times_with_io, times_without_io], ["Processing time with IO", "Processing time without IO"])
    plot_measurements([times_with_io, times_without_io,times_one_encode,times_one_compare],
                      ["Processing time with IO", "Processing time without IO", "One batch encode", "One batch compare"])

def plot_times_multiple_runs(log_paths, run_names):
    num_runs = len(log_paths)

    observed_times = [
        "total_encode_compare_with_IO",
        "total_encode_compare",
        "first_full_batch_encode",
        "first_full_batch_compare"
    ]
    times_names = ["Processing time with IO", "Processing time without IO", "One batch encode", "One batch compare"]

    means_per_runs = []
    stds_per_runs = []

    for run_i in range(num_runs):
        args, times = load_logs(log_paths[run_i])
        name = run_names[run_i]
        print(name)

        # exclude file 0 - that one doesn't have any comparison
        delete_keys = []
        for k in times.keys():
            if "_file_000_" in k:
                delete_keys.append(k)
        for k in delete_keys: del times[k]

        for_plots = []
        name_plots = []
        for i, observed in enumerate(observed_times):
            for_plots.append( check_statistic(times, observed) )
            name_plots.append( times_names[i] )

        #plot_measurements(for_plots, name_plots, plot_title = name)

        means_per_runs.append( [np.mean(vals) for vals in for_plots] )
        stds_per_runs.append( [np.std(vals) for vals in for_plots] )
    print(means_per_runs)
    print(stds_per_runs)
    print(run_names)
    print("each one has", name_plots)

    data = {}
    std_data = {}
    for i, run_name in enumerate(run_names):
        data[run_name] = means_per_runs[i]
        std_data[run_name] = stds_per_runs[i]

    fig, ax = plt.subplots(figsize=(10, 5))
    custom_bar_plot(ax, data, std_data, total_width=.8, single_width=.9)
    plt.xticks(range(len(name_plots)), name_plots)
    plt.show()


if __name__ == "__main__":
    # A
    #log_path = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/logs_unibap/results03_withbatchsizes/log_64batch.json"
    ignore_file_i_above = None
    #log_path = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/logs_unibap/results04_cdwholedataset/log_128batch.json"
    log_path = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/logs_unibap/results05_newnames_allversions/log_64batch.json"
    #ignore_file_i_above = 30 # can't even render beyond 100
    plot_all_files(log_path, ignore_file_i_above=ignore_file_i_above)

    # B
    # logs_folder = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/logs_unibap/results03_withbatchsizes/" # 3 files, variety of batchsizes
    # batchsizes = [2,4,8,16,32,64,128]
    # logs_folder = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/logs_unibap/results04_cdwholedataset/" # all 1024 files!, fewer batchsizes
    # batchsizes = [16, 32, 64, 128]
    logs_folder = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/logs_unibap/results05_newnames_allversions/"
    batchsizes = [2, 4, 8, 16, 32, 64, 128]
    logs = [ logs_folder+"log_"+str(i)+"batch.json" for i in batchsizes]
    names = [ "Batch Size "+str(i) for i in batchsizes]
    plot_times_multiple_runs(logs, names)

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

v2 shoudl also have
time_file_???_dataloader_create

"""