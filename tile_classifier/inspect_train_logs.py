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
            ax.bar_label(bar)

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

def inspect_train_logs(log_path = "../results/logs.json"):
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

def plot_train_times_multiple_runs(log_paths, run_names, plot_title=""):
    num_runs = len(log_paths)

    observed_times = [
        "_full",
    ]
    times_names = ["Average epoch training time"]

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

    fig, ax = plt.subplots(figsize=(5, 5))
    if len(plot_title) > 0:
        plt.title(plot_title)

    custom_bar_plot(ax, data, std_data, total_width=.8, single_width=.9)
    plt.xticks(range(len(name_plots)), name_plots)
    plt.show()



def plot_other_times(log_paths, run_names):
    num_runs = len(log_paths)

    observed_times = [
        "dataset_load","dataloader_load","model_create","save_model","one_batch_prediction"
    ]
    times_names = ["dataset_load","dataloader_load","model_create","save_model","one_batch_prediction"]

    means_per_runs = []
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
    print(means_per_runs)
    print(run_names)
    print("each one has", name_plots)

    data = {}
    for i, run_name in enumerate(run_names):
        data[run_name] = means_per_runs[i]

    fig, ax = plt.subplots(figsize=(10, 5))
    custom_bar_plot(ax, data, total_width=.8, single_width=.9)
    plt.xticks(range(len(name_plots)), name_plots)
    plt.show()



if __name__ == "__main__":
    logs_folder = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/_logs_unibap_step1/results15a_prefinal_again/"
    logs_folder = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/_logs_unibap_step1/results15b_testFromDockerWithLog/"

    logs_folder = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/_from_their_side_step2/results01_dec13_corrected/ravaen_result/results/"

    logs_folder = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/step3_after it was packaged as a scfw app/14-12-2022_121557__ravaen/volumes/ravaen/"

    batchsizes = [32, 64, 128, 256]
    logs = [ logs_folder+"tile_classifier_log_"+str(i)+"batch.json" for i in batchsizes]
    names = ["Batch Size " + str(i) for i in batchsizes]
    plot_train_times_multiple_runs(logs, names, "Training time, model: [128-Dense-1]")

    # logs = [logs_folder + "tile_classifier_log_" + str(i) + "batch_multiclass_4classes.json" for i in batchsizes]
    # plot_train_times_multiple_runs(logs, names, "Training time, model: [128-Dense-4], 4 classes")
    #
    # logs = [logs_folder + "tile_classifier_log_" + str(i) + "batch_multiclass_12classes.json" for i in batchsizes]
    # plot_train_times_multiple_runs(logs, names, "Training time, model: [128-Dense-12], 12 classes")
    #

    b = "256"
    # b = "128"
    logs = ["tile_classifier_log_"+b+"batch.json","tile_classifier_log_"+b+"batch_multiclass_4classes.json","tile_classifier_log_"+b+"batch_multiclass_12classes.json"]
    logs = [logs_folder+l for l in logs]
    names = ["classifier 1", "classifier 4", "classifier 12"]
    plot_train_times_multiple_runs(logs, names, "Training times (with batch "+b+")")

    assert False


    # logs from dorbit
    logs_folder = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/_from_their_side_step2/results00_dec6/"
    batchsizes = [32, 64, 128, 256]
    logs = [ logs_folder+"tile_classifier_log_"+str(i)+"batch.json" for i in batchsizes]
    names = ["Batch Size " + str(i) for i in batchsizes]
    plot_train_times_multiple_runs(logs, names, "Training time, model: [128-Dense-1]")

    logs = [logs_folder + "tile_classifier_log_" + str(i) + "batch_multiclass_4classes.json" for i in batchsizes]
    plot_train_times_multiple_runs(logs, names, "Training time, model: [128-Dense-4], 4 classes")

    assert False

    logs_folder = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/_logs_unibap/results12_trainWith12classes/"
    batchsizes = [4,8,16,32, 64, 128, 256]
    logs = [ logs_folder+"tile_classifier_log_"+str(i)+"batch_multiclass_12classes.json" for i in batchsizes]
    names = [ "Batch Size "+str(i) for i in batchsizes]
    plot_train_times_multiple_runs(logs, names)


    ### v10 inspection
    logs_folder = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/_logs_unibap/results10reducedfin/"
    logs_folder = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/_logs_unibap/results11f_finals/results11f/"
    logs_folder = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/_logs_unibap/results11f_finals/results11f_checkWithBadPaths/"

    """log types:
    highres10band_128batch.json
    log_16batch.json
    * tile_classifier_log_32batch.json
    * tile_classifier_log_32batch_multiclass_4classes.json
    """
    batchsizes = [32, 64, 128, 256]
    logs = [ logs_folder+"tile_classifier_log_"+str(i)+"batch.json" for i in batchsizes]
    names = [ "Batch Size "+str(i) for i in batchsizes]
    plot_train_times_multiple_runs(logs, names)

    # plot_other_times(logs, names)


    batchsizes = [32, 64, 128, 256]
    logs = [ logs_folder+"tile_classifier_log_"+str(i)+"batch_multiclass_4classes.json" for i in batchsizes]
    names = [ "4 classes model, Batch Size "+str(i) for i in batchsizes]
    plot_train_times_multiple_runs(logs, names)


    assert False

    log_path = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/logs_unibap/results06_withlogsv2/tile_classifier_log_64batch.json"
    #inspect_train_logs(log_path)


    logs_folder = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/logs_unibap/results06_withlogsv2/"
    batchsizes = [2, 4, 8, 16, 32, 64, 128, 256]
    logs = [ logs_folder+"tile_classifier_log_"+str(i)+"batch.json" for i in batchsizes]
    names = [ "Batch Size "+str(i) for i in batchsizes]
    plot_train_times_multiple_runs(logs, names)

    plot_other_times(logs, names)

"""
Available times:

time_dataset_load
time_dataloader_load
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