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

def plot_times_multiple_runs(log_paths, run_names, plot_title = "", ignore_list=[], special=None, save=False):
    num_runs = len(log_paths)

    observed_times = [
        "total_encode_compare_with_IO",
        "total_encode_compare",
        "_batch_000_encode", # < first batch of each file ...
        # "first_full_batch_encode",
        # "first_full_batch_compare",
        # "first_full_batch_encode_and_compare", # sanitycheck - it should be the same as the last two added
        "dataloader_create",
        "save_latents_changemap",
    ]
    times_names = ["Encoding + IO", "Encoding only",
                   "One batch encode", #"One batch compare", "One batch enc+comp",
                   "Create dataloader", "Save"]
    if len(ignore_list) > 0:
        for ignore_item in ignore_list:
            for idx, a in enumerate(observed_times):
                if ignore_item == a:
                    del times_names[idx]
                    del observed_times[idx]
                    break



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
    if len(plot_title) > 0:
        plt.title(plot_title)
    custom_bar_plot(ax, data, std_data, total_width=.8, single_width=.9, special=special)
    plt.xticks(range(len(name_plots)), name_plots)
    ax.set_ylabel('Time in sec')

    # plt.show()
    if save:
        plt.savefig(save)

    plt.draw()


if __name__ == "__main__":
    # logs from dorbit
    logs_folder = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/_from_their_side_step2/results00_dec6/"
    batchsizes = [32, 64, 128]
    logs = [ logs_folder+"log_"+str(i)+"batch.json" for i in batchsizes]
    names = [ "Batch Size "+str(i) for i in batchsizes]
    plot_times_multiple_runs(logs, names, "VAE (RGB+NIR) encoding speed")

    batchsizes = [32, 64, 128]
    logs = [ logs_folder+"highres10band_"+str(i)+"batch.json" for i in batchsizes]
    names = [ "Batch Size "+str(i) for i in batchsizes]
    plot_times_multiple_runs(logs, names, "VAE (10 band) encoding speed")

    bands = [3,6,8]
    logs = [ logs_folder+"exp"+str(i)+"band_64batch.json" for i in bands]
    log_4bands = logs_folder+"log_64batch.json"
    log_10bands = logs_folder+"highres10band_64batch.json"
    logs = [logs[0]] + [log_4bands] + logs[1:] + [log_10bands]
    bands = [3, 4, 6, 8, 10]
    names = [ "Bands "+str(i) for i in bands]
    plot_times_multiple_runs(logs, names, "VAE encoding speed depending on the number of bands, batch size 64",
                             ignore_list=["save_latents_changemap"])

    assert False

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
    batchsizes = [32, 64, 128]
    logs = [ logs_folder+"log_"+str(i)+"batch.json" for i in batchsizes]
    names = [ "Batch Size "+str(i) for i in batchsizes]
    plot_times_multiple_runs(logs, names)

    batchsizes = [32, 64, 128]
    logs = [ logs_folder+"highres10band_"+str(i)+"batch.json" for i in batchsizes]
    names = [ "Batch Size "+str(i) for i in batchsizes]
    plot_times_multiple_runs(logs, names)

    logs_folder = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/_logs_unibap/results11f_finals/results11f_v2_extra/"
    batchsizes = [4, 8, 16]
    logs = [ logs_folder+"log_"+str(i)+"batch.json" for i in batchsizes]
    names = [ "Batch Size "+str(i) for i in batchsizes]
    plot_times_multiple_runs(logs, names)

    batchsizes = [4, 8, 16]
    logs = [ logs_folder+"highres10band_"+str(i)+"batch.json" for i in batchsizes]
    names = [ "Batch Size "+str(i) for i in batchsizes]
    plot_times_multiple_runs(logs, names)

    bands = [3,6,8]
    logs = [ logs_folder+"exp"+str(i)+"band_64batch.json" for i in bands]
    names = [ "Bands "+str(i) for i in bands]
    plot_times_multiple_runs(logs, names)



    assert False


    ignore_file_i_above = 10 #None # or 30
    log_path = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/_logs_unibap/results09all/results09a_basic_run_keep10latents/log_64batch.json"
    plot_all_files(log_path, ignore_file_i_above=ignore_file_i_above)
    log_path = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/_logs_unibap/results09all/results09_f_experimental_valid/highres10band_64batch.json"
    plot_all_files(log_path, ignore_file_i_above=ignore_file_i_above)

    logs_folder = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/_logs_unibap/results09all/results09a_basic_run_keep10latents/"
    logs_folder = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/_logs_unibap/results09all/results09b_no_weights_docker/"
    logs_folder = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/_logs_unibap/results09all/results09c3_no_weights_no_data_docker_rerun2/"
    batchsizes = [2, 4, 8, 16, 32, 64, 128]
    logs = [ logs_folder+"log_"+str(i)+"batch.json" for i in batchsizes]
    names = [ "Batch Size "+str(i) for i in batchsizes]
    plot_times_multiple_runs(logs, names)

    # logs_folder = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/logs_unibap/results06c_tried_just_dataloader/"
    # batchsizes = [2, 4, 8, 16, 32, 64, 128]
    # logs = [ logs_folder+"log_"+str(i)+"batch_BENCH_JUST_DATALOADER.json" for i in batchsizes]
    # names = [ "Dataloder only, Batch Size "+str(i) for i in batchsizes]
    # plot_times_multiple_runs(logs, names)


    logs_folder = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/_logs_unibap/results09all/results09_f_experimental_valid/"
    # /home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/_logs_unibap/results09all/results09_f_experimental_valid/highres10band_32batch.json
    # /home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/_logs_unibap/results09all/results09_f_experimental_valid/nodata_4batch.json
    # /home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/_logs_unibap/results09all/results09_f_experimental_valid/nodatanoweights_16batch.json
    # /home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/_logs_unibap/results09all/results09_f_experimental_valid/noweights_8batch.json
    batchsizes = [2, 4, 8, 16, 32, 64, 128]
    logs = [ logs_folder+"highres10band_"+str(i)+"batch.json" for i in batchsizes]
    names = [ "10 band; Batch Size "+str(i) for i in batchsizes]
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