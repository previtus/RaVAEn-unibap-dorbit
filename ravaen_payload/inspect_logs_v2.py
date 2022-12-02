import json
import numpy as np

from matplotlib import pyplot as plt
from inspect_logs_v1 import custom_bar_plot, plot_measurements, check_statistic, load_logs


def plot_all_files(log_path = "../results/logs.json", ignore_file_i_above=None):
    args, times = load_logs(log_path)

    # debug:
    # for k in times.keys():
    #     print(k)

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
        "first_full_batch_compare",
        "first_full_batch_encode_and_compare", # sanitycheck - it should be the same as the last two added
        "dataloader_create",
        "save_latents_changemap",
    ]
    times_names = ["Processing time with IO", "Processing time without IO", "One batch encode", "One batch compare", "One batch enc+comp",
                   "Create dataloader", "Save"]

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
    ignore_file_i_above = 10 #None # or 30
    log_path = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/logs_unibap/results06_withlogsv2/log_64batch.json"
    plot_all_files(log_path, ignore_file_i_above=ignore_file_i_above)

    logs_folder = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/logs_unibap/results06_withlogsv2/"
    batchsizes = [2, 4, 8, 16, 32, 64, 128]
    logs = [ logs_folder+"log_"+str(i)+"batch.json" for i in batchsizes]
    names = [ "Batch Size "+str(i) for i in batchsizes]
    plot_times_multiple_runs(logs, names)

    logs_folder = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/logs_unibap/results06c_tried_just_dataloader/"
    batchsizes = [2, 4, 8, 16, 32, 64, 128]
    logs = [ logs_folder+"log_"+str(i)+"batch_BENCH_JUST_DATALOADER.json" for i in batchsizes]
    names = [ "Dataloder only, Batch Size "+str(i) for i in batchsizes]
    plot_times_multiple_runs(logs, names)



"""
[Available times]

time_files_query
time_model_load

_dataloader_create

_first_full_batch_encode
_first_full_batch_compare
_first_full_batch_encode_and_compare


_total_encode_compare
_total_encode_compare_with_IO

_save_latents_changemap

time_file_027_dataloader_create
time_file_027_first_full_batch_encode
time_file_027_first_full_batch_compare
time_file_027_first_full_batch_encode_and_compare
time_file_027_total_encode_compare
time_file_027_total_encode_compare_with_IO
time_file_027_save_latents_changemap


"""