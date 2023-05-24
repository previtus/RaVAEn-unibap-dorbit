from inspect_train_logs import plot_train_times_multiple_runs, plot_other_times, load_logs
from inspect_train_logs_v2_final import plot_perf_over_epochs
from matplotlib import pyplot as plt


if __name__ == "__main__":

    logs_folder = "../_final_results_21_12_2022/"

    batchsizes = [32, 64, 128, 256]
    logs = [ logs_folder+"tile_classifier_log_"+str(i)+"batch.json" for i in batchsizes]
    names = ["Batch Size " + str(i) for i in batchsizes]

    plot_train_times_multiple_runs(logs, names, "Training time, model: [128-Dense-1]",
                                   save="fig1_training_on_board_batchsizes.pdf")

    # logs = [logs_folder + "tile_classifier_log_" + str(i) + "batch_multiclass_4classes.json" for i in batchsizes]
    # plot_train_times_multiple_runs(logs, names, "Training time, model: [128-Dense-4], 4 classes")
    #
    # logs = [logs_folder + "tile_classifier_log_" + str(i) + "batch_multiclass_12classes.json" for i in batchsizes]
    # plot_train_times_multiple_runs(logs, names, "Training time, model: [128-Dense-12], 12 classes")
    #

    # b = "256"
    # b = "128"
    b = "64"
    logs = ["tile_classifier_log_"+b+"batch.json","tile_classifier_log_"+b+"batch_multiclass_4classes.json","tile_classifier_log_"+b+"batch_multiclass_12classes.json"]
    logs = [logs_folder+l for l in logs]
    names = ["classifier 1", "classifier 4", "classifier 12"]
    plot_train_times_multiple_runs(logs, names, "Training times (with batch "+b+")")


    # plot_perf_over_epochs(logs[2], add_title=", (classifier 12, batch "+b+")") # inspect per epochs
    #
    #
    # plot_other_times(logs, names)