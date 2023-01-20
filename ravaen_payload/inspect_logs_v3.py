import json
import numpy as np

from matplotlib import pyplot as plt
from inspect_logs_v1 import custom_bar_plot, plot_measurements, check_statistic, load_logs, load_logs_memory
from inspect_logs_v2 import plot_times_multiple_runs, plot_all_files

def all_keys(log_path = "../results/logs.json"):
    args, times = load_logs(log_path)
    for k in times.keys():
        print(k)

def inspect_mem_logs(log_path):
    memory = load_logs_memory(log_path)
    print(memory.keys())
    print(memory['memory_log_atFile1_mem1'])
    print(memory['memory_log_atFile1_mem2'])


def plot_perf_over_batches(log_path = "../results/logs.json", check=["encode", "compare", "encode_and_compare"], add_title="", save=False):
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
        fig, ax = plt.subplots(figsize=(10, 5))

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
        ax.set_title('Timing per file and per batch ('+stat+')'+add_title+':')
        ax.legend()

        if save:
            ending=save.split(".")[-1]
            plt.savefig(save + "_"+stat+"."+ending)

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

    logs_folder = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/_logs_unibap_step1/results15a_prefinal_again/"
    logs_folder = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/_logs_unibap_step1/results15b_testFromDockerWithLog/"
    logs_folder = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/_logs_unibap_step1/results15c_oncemore/"
    logs_folder = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/_from_their_side_step2/results01_dec13/results/"

    logs_folder = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/_from_their_side_step2/results01_dec13_corrected/ravaen_result/results/"

    # execution result of the application after it was packaged as a scfw app
    logs_folder = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/step3_after it was packaged as a scfw app/14-12-2022_121557__ravaen/volumes/ravaen/"

    batchsizes = [32, 64, 128]
    logs = [ logs_folder+"log_"+str(i)+"batch.json" for i in batchsizes]

    # inspect_mem_logs(logs[1])
    ignore_list = ["save_latents_changemap", "dataloader_create", "_batch_000_encode" ]

    names = [ "Batch Size "+str(i) for i in batchsizes]
    plot_times_multiple_runs(logs, names, "VAE (RGB+NIR) encoding speed",
                             ignore_list=ignore_list)
    plt.show()

    # all_keys(logs[0])

    plot_perf_over_batches(logs[0], add_title=", (batch 32)") # batch 32
    plot_perf_over_batches(logs[1], add_title=", (batch 64)") # batch 64
    # plot_perf_over_batches(logs[2]) # batch 128


    ### Torch CPU, Openvino CPU vs Openvino MYRIAD
    logs = ["log_64batch.json", "log_openvinooncpu_64batch.json", "log_openvino_64batch.json"]
    logs = [logs_folder+l for l in logs]
    names = ["Torch CPU", "Openvino CPU", "Openvino MYRIAD"]
    plot_times_multiple_runs(logs, names, "Different compute devices (batch 64)",
                             ignore_list=ignore_list, special="delfirstcolor")
    plt.show()

    plot_perf_over_batches(logs[2]) # 64 with myriad


    ### Model with 3 bands, 4 bands, 10 bands
    logs = ["exp3band_64batch.json", "log_64batch.json", "exp10band_64batch.json"]
    logs = [logs_folder + l for l in logs]
    names = ["3 bands", "4 bands (real)", "10 bands"]
    plot_times_multiple_runs(logs, names, "Model with different number of bands (batch 64)",
                             ignore_list=ignore_list)
    plt.show()

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