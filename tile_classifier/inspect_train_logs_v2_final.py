from inspect_train_logs import plot_train_times_multiple_runs, plot_other_times, load_logs
from matplotlib import pyplot as plt

def plot_perf_over_epochs(log_path = "../results/logs.json", add_title=""):
    args, times = load_logs(log_path)
    batch_size = args['args_batch_size']

    #time_epoch_008_full
    print("time keys", times.keys())
    keys_per_epoch = [k for k in times.keys() if "time_epoch_" in k]
    keys_per_epoch.sort()
    print("sorted > ", keys_per_epoch)

    times_per_epochs = []
    for k in keys_per_epoch:
        time_per_epoch = times[k]
        times_per_epochs.append(time_per_epoch)

    labels = [str(i) for i in range(len(times_per_epochs))]
    fig, ax = plt.subplots()

    plt.bar(labels, times_per_epochs)

    ax.set_ylabel('Time in sec')
    ax.set_xlabel('Epoch i')
    ax.set_title('Training per epoch' + add_title + ':')
    # ax.legend()
    plt.show()


if __name__ == "__main__":

    logs_folder = "../_final_results_21_12_2022/"

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
    # b = "64"
    logs = ["tile_classifier_log_"+b+"batch.json","tile_classifier_log_"+b+"batch_multiclass_4classes.json","tile_classifier_log_"+b+"batch_multiclass_12classes.json"]
    logs = [logs_folder+l for l in logs]
    names = ["classifier 1", "classifier 4", "classifier 12"]
    plot_train_times_multiple_runs(logs, names, "Training times (with batch "+b+")")


    plot_perf_over_epochs(logs[2], add_title=", (classifier 12, batch "+b+")") # inspect per epochs


    plot_other_times(logs, names)