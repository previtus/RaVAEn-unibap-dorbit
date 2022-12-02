from dataset import generate_dataset
import torch
from model_pytorch import LilModel, LilDataset, get_n_params
from torch.utils.data import DataLoader
import numpy as np
import time
import os
import json

plot = False # if set to True, needs matplotlib
if plot:
    try:
        import matplotlib as plt
    except:
        print("Failed to import matplotlib, setting plot to False")
        plot = False

def main(settings):
    main_start_time = time.time()

    logged = {}
    for key in settings.keys():
        logged["args_" + key] = settings[key]

    dataset_load_path = settings["dataset_as_np"]
    trained_model_save_path = settings["trained_model_path"]
    BATCH_SIZE = int(settings["batch_size"])
    EPOCHS = int(settings["train_epochs"])


    print("Will load data from", dataset_load_path, ", train and save the model to, ",trained_model_save_path)

    ### DATASET:
    time_before_dataset = time.time()
    try:
        X_latents, _, Y = generate_dataset()
    except:
        loaded = np.load(dataset_load_path)
        X_latents = loaded["X_latents"]
        Y = loaded["Y"]

    print("Dataset:")
    print("X latents:", X_latents.shape)
    # print("X tiles:", X_tiles.shape)
    print("Y labels:", Y.shape)
    dataset_load_time = time.time() - time_before_dataset
    logged["time_dataset_load"] = dataset_load_time

    ### MODEL and training:
    time_before_dataloader = time.time()
    X_latents = torch.from_numpy(X_latents).float()
    Y = torch.from_numpy(Y).float().unsqueeze(1)
    print("X latents:", X_latents.shape, X_latents.dtype)
    print("Y labels:", Y.shape, Y.dtype)

    train_loader = DataLoader(LilDataset(X_latents,Y), batch_size=BATCH_SIZE)

    time_now = time.time()
    dataloader_load_time = time_now - time_before_dataloader
    logged["time_dataloader_load"] = dataloader_load_time
    time_before_model = time_now

    model = LilModel()
    print(model)
    print(get_n_params(model))

    criterion = model.criterion
    optimizer = model.configure_optimizers()
    model_create_time = time.time() - time_before_model
    logged["time_model_create"] = model_create_time

    example_inputs = None

    N_samples = len(Y)
    PRINT_EVERY = int(N_samples/2)
    training_losses = []
    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        time_start_of_epoch = time.time()

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            if i == 0 and epoch == 0:
                example_inputs = inputs.clone()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % PRINT_EVERY == PRINT_EVERY-1:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / PRINT_EVERY:.3f}')
                running_loss = 0.0
        loss_this_epoch = running_loss / N_samples
        training_losses.append(loss_this_epoch)

        one_epoch_time = time.time() - time_start_of_epoch
        logged["time_epoch_" + str(epoch).zfill(3) + "_full"] = one_epoch_time
        logged["loss_epoch_" + str(epoch).zfill(3) + "_loss"] = loss_this_epoch

    print('Finished Training')
    print("Losses:", training_losses)

    start_time = time.time()
    torch.save(model.state_dict(), trained_model_save_path)
    save_model_time = time.time() - start_time
    logged["time_save_model"] = save_model_time

    # Demo using the model:
    start_time = time.time()
    print("Using the model with example input:", example_inputs.shape)
    demo_predictions = model(X_latents)
    print("Y predictions:", demo_predictions.shape)
    one_batch_prediction_time = time.time() - start_time
    logged["time_one_batch_prediction"] = one_batch_prediction_time

    if plot:
        plt.plot(training_losses)
        plt.show()

    # LOG
    main_end_time = time.time()
    main_run_time = (main_end_time - main_start_time)
    print("TOTAL RUN TIME = " + str(main_run_time) + "s (" + str(main_run_time / 60.0) + "min)")

    logged["time_main"] = main_run_time


    print(logged)
    with open(os.path.join(settings["results_dir"], settings["log_name"]+"_"+str(BATCH_SIZE)+"batch.json"), "w") as fh:
        json.dump(logged, fh)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser('Run training tile classifier')
    parser.add_argument('--dataset_as_np', default="weights/train_dataset_as_np.npz",
                        help="Path to the prepared latents dataset.")
    parser.add_argument('--trained_model_path', default='results/tile_model.pt',
                        help="Where to save the model weights")
    parser.add_argument('--results_dir', default='results/',
                        help="Path where to save the results")
    parser.add_argument('--log_name', default='tile_classifier_log',
                        help="Name of the log (batch size will be appended in any case).")

    parser.add_argument('--batch_size', default=8,
                        help="Batch size for the dataloader and training")


    parser.add_argument('--train_epochs', default=10,
                        help="How many epochs to train for")

    # parser.add_argument('--time-limit', type=int, default=300,
    #                     help="time limit for running inference [300]")

    args = vars(parser.parse_args())

    start_time = time.time()
    main(args)
    end_time = time.time()

    time = (end_time - start_time)
    print("TOTAL TRAIN TIME = " + str(time) + "s (" + str(time / 60.0) + "min)")
