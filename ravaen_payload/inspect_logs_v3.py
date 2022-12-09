import json
import numpy as np

from matplotlib import pyplot as plt
from inspect_logs_v1 import custom_bar_plot, plot_measurements, check_statistic, load_logs
from inspect_logs_v2 import plot_times_multiple_runs, plot_all_files

def all_keys(log_path = "../results/logs.json"):
    args, times = load_logs(log_path)
    for k in times.keys():
        print(k)

def plot_perf_over_batches(log_path = "../results/logs.json", check=["encode", "compare", "encode_and_compare"]):
    args, times = load_logs(log_path)

    batch_size = args['args_batch_size']

    count_files = []
    count_batches = []
    for k in times.keys():
        # print(k)
        if "_dataloader_create" in k:
            count_files.append(k)
        if "time_file_000_batch_" in k:
            if "_encode_and_compare" in k:
                count_batches.append(k)

    # print(len(count_files), count_files)
    # print(len(count_batches), count_batches)
    num_of_files = len(count_files)
    num_of_batches = len(count_batches)

    # time_file_<f>_batch_<b>_encode
    # time_file_<f>_batch_<b>_compare

    # Over individual files:
    for stat in check:
        # if True:
        # stat = "encode"
        plot_stackbar_components = {}
        for file_i in range(num_of_files):

            for batch_i in range(num_of_batches):

                if batch_i not in plot_stackbar_components:
                    plot_stackbar_components[ batch_i ] = []

                key = "time_file_"+str(file_i).zfill(3)+"_batch_"+str(batch_i).zfill(3)+"_"+stat
                t = times[key]
                # print(key, t)

                plot_stackbar_components[batch_i].append(t)

            #print("file",file_i, "=>", batches_per_file)


        ###

        # stacked bar data:
        labels = [str(i) for i in range(num_of_files)]
        width = 0.35  # the width of the bars: can also be len(x) sequence
        fig, ax = plt.subplots()

        cummulative_bottom = None
        for batch_i in range(num_of_batches):
            print(batch_i)
            component_data = plot_stackbar_components[batch_i]

            if cummulative_bottom is None:
                ax.bar(labels, component_data, width, label='Batch '+str(batch_i))
                cummulative_bottom = component_data
            else:
                ax.bar(labels, component_data, width, bottom=cummulative_bottom, label='Batch ' + str(batch_i))
                cummulative_bottom = [c + component_data[i] for i, c in enumerate(cummulative_bottom)]

        ax.set_ylabel('Time in sec')
        ax.set_xlabel('File i')
        ax.set_title('Timing per file and per batch ('+stat+'):')
        ax.legend()
        plt.show()
        # ###
        # plot_all = plot_all[0:8]
        # plot_all_labels = plot_all_labels[0:8]
        #
        # fig, ax = plt.subplots(figsize=(10, 5))
        # plt.bar(plot_all_labels, plot_all)
        # plt.xticks(range(len(plot_all_labels)), plot_all_labels)
        # plt.show()


if __name__ == "__main__":
    # V3 has more refined logs! Timings for each batch

    logs_folder = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/_logs_unibap_step1/results14_perBatchStats_mainRun/"
    batchsizes = [32, 64, 128]
    logs = [ logs_folder+"log_"+str(i)+"batch.json" for i in batchsizes]
    names = [ "Batch Size "+str(i) for i in batchsizes]
    #plot_times_multiple_runs(logs, names, "VAE (RGB+NIR) encoding speed")
    #plt.show()

    # all_keys(logs[0])

    # plot_perf_over_batches(logs[1]) # batch 64
    # plot_perf_over_batches(logs[0]) # batch 32
    plot_perf_over_batches(logs[2]) # batch 128

"""
[Available times]


time_file_022_dataloader_create
time_file_022_batch_000_encode
time_file_022_batch_000_compare
time_file_022_batch_000_encode_and_compare
time_file_022_batch_001_encode
time_file_022_batch_001_compare
time_file_022_batch_001_encode_and_compare
time_file_022_batch_002_encode
time_file_022_batch_002_compare
time_file_022_batch_002_encode_and_compare
time_file_022_batch_003_encode
time_file_022_batch_003_compare
time_file_022_batch_003_encode_and_compare
time_file_022_batch_004_encode
time_file_022_batch_004_compare
time_file_022_batch_004_encode_and_compare
time_file_022_batch_005_encode
time_file_022_batch_005_compare
time_file_022_batch_005_encode_and_compare
time_file_022_batch_006_encode
time_file_022_batch_006_compare
time_file_022_batch_006_encode_and_compare
time_file_022_batch_007_encode
time_file_022_batch_007_compare
time_file_022_batch_007_encode_and_compare
time_file_022_total_encode_compare
time_file_022_total_encode_compare_with_IO
time_file_022_save_latents_changemap


"""