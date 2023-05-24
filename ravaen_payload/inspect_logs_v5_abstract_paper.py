import json
import numpy as np

from matplotlib import pyplot as plt
from inspect_logs_v1 import custom_bar_plot, plot_measurements, check_statistic, load_logs, load_logs_memory
from inspect_logs_v2 import plot_times_multiple_runs, plot_all_files

from inspect_logs_v3 import plot_perf_over_batches, inspect_mem_logs, all_keys

if __name__ == "__main__":
    logs_folder = "../_final_results_21_12_2022/"

    batchsizes = [32, 64, 128]
    logs = [ logs_folder+"log_"+str(i)+"batch.json" for i in batchsizes]

    # inspect_mem_logs(logs[1])
    ignore_list = ["save_latents_changemap", "dataloader_create", "_batch_000_encode" ]

    names = [ "Batch Size "+str(i) for i in batchsizes]
    # plot_times_multiple_runs(logs, names, "VAE (RGB+NIR) encoding speed",
    #                          ignore_list=ignore_list)
    # plt.show()

    # all_keys(logs[0])

    # plot_perf_over_batches(logs[0], add_title=", (batch 32)") # batch 32
    # plot_perf_over_batches(logs[1], add_title=", (batch 64)") # batch 64
    # plot_perf_over_batches(logs[2], add_title=", (batch 128)") # batch 128

    plot_perf_over_batches(logs[1], add_title=", (pytorch cpu, b64)", check=["encode"],
                           save="fig2_details_cpu_batches.pdf") # batch 64


    ### Torch CPU, Openvino CPU vs Openvino MYRIAD
    logs = ["log_64batch.json", "log_openvinooncpu_64batch.json", "log_openvino_64batch.json"]
    logs = [logs_folder+l for l in logs]
    names = ["Torch CPU", "Openvino CPU", "Openvino MYRIAD"]
    plot_times_multiple_runs(logs, names, "Different compute devices (batch 64)",
                             ignore_list=ignore_list, special="delfirstcolor",
                             save="fig1_inference_diff_devices.pdf")
    plt.show()


    # are there outliers when using Openvino?
    # plot_perf_over_batches(logs[1], add_title=", (openvino cpu, b64)", check=["encode"]) # batch 64
    plot_perf_over_batches(logs[2], add_title=", (openvino myriad, b64)", check=["encode"],
                           save="fig2_details_myriad_batches.pdf") # batch 64


    ### Model with 3 bands, 4 bands, 10 bands
    logs = ["exp3band_64batch.json", "log_64batch.json", "exp10band_64batch.json"]
    logs = [logs_folder + l for l in logs]
    names = ["3 bands", "4 bands (real)", "10 bands"]
    # plot_times_multiple_runs(logs, names, "Model with different number of bands (batch 64)",
    #                          ignore_list=ignore_list)
    # plt.show()
