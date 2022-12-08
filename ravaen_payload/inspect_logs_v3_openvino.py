import json
import numpy as np

from matplotlib import pyplot as plt
from inspect_logs_v1 import custom_bar_plot, plot_measurements, check_statistic, load_logs
from inspect_logs_v2 import plot_times_multiple_runs




if __name__ == "__main__":

    ### v13 d - smaller openvino model?

    logs_folder = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/_logs_unibap_step1/results13d_modelVersionsAndPlottable/" # from a openvino cpu and mu only model

    ignore_list = ["save_latents_changemap", "dataloader_create"]
    logs= ["log_64batch.json", "log_openvino_64batch.json", "log_openvinooncpu_64batch.json", "log_openvino_CPU_smallerModelMusOnly_64batch.json", "log_openvino_Myriad_smallerModelMusOnly_64batch.json"]
    logs = [logs_folder + l for l in logs]
    names = ["torch cpu", "openvino myriad", "openvino cpu", "openvino cpu, mu only", "openvino myriad, mu only"]

    logs= ["log_64batch.json", "log_openvino_CPU_smallerModelMusOnly_64batch.json", "log_openvino_Myriad_smallerModelMusOnly_64batch.json"]
    logs = [logs_folder + l for l in logs]
    names = ["torch cpu", "openvino cpu (mu only)", "openvino myriad (mu only)"]

    plot_times_multiple_runs(logs, names, "Different model versions, batch size fixed at 64", ignore_list=ignore_list)
    plt.show()
    assert False
    ### v13 b

    #logs_folder = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/_logs_unibap_step1/results13a_openvino_checkspeeds/"

    logs_folder = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/_logs_unibap_step1/results13b_timing_without_nptorch_convos/"
    # batchsizes = [4,8,16,32,64,128]

    ignore_list = ["save_latents_changemap","dataloader_create"]

    # or only 1 batch:
    ignore_list = ["total_encode_compare_with_IO",
        "total_encode_compare",
        "dataloader_create",
        "save_latents_changemap",
    ]

    batchsizes = [32,64,128]
    logs = [ logs_folder+"log_"+str(i)+"batch.json" for i in batchsizes]
    names = [ "Batch Size "+str(i) for i in batchsizes]
    plot_times_multiple_runs(logs, names, "Torch CPU",
                             ignore_list=ignore_list)

    logs = [ logs_folder+"log_openvino_"+str(i)+"batch.json" for i in batchsizes]
    names = [ "Batch Size "+str(i) for i in batchsizes]
    plot_times_multiple_runs(logs, names, "OPENVINO MYRIAD",
                             ignore_list=ignore_list)

    #log_openvinooncpu_4batch
    logs = [ logs_folder+"log_openvinooncpu_"+str(i)+"batch.json" for i in batchsizes]
    names = [ "Batch Size "+str(i) for i in batchsizes]
    plot_times_multiple_runs(logs, names, "OPENVINO CPU",
                             ignore_list=ignore_list)
    plt.show()

    # batchsizes = [32, 64, 128]
    # logs = [ logs_folder+"highres10band_"+str(i)+"batch.json" for i in batchsizes]
    # names = [ "Batch Size "+str(i) for i in batchsizes]
    # plot_times_multiple_runs(logs, names, "VAE (10 band) encoding speed")
    #
    # bands = [3,6,8]
    # logs = [ logs_folder+"exp"+str(i)+"band_64batch.json" for i in bands]
    # log_4bands = logs_folder+"log_64batch.json"
    # log_10bands = logs_folder+"highres10band_64batch.json"
    # logs = [logs[0]] + [log_4bands] + logs[1:] + [log_10bands]
    # bands = [3, 4, 6, 8, 10]
    # names = [ "Bands "+str(i) for i in bands]
    # plot_times_multiple_runs(logs, names, "VAE encoding speed depending on the number of bands, batch size 64",
    #                          ignore_list=["save_latents_changemap"])
    #
