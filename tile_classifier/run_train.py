from dataset import generate_dataset
import torch
from model_pytorch import LilModel, LilDataset, get_n_params
from torch.utils.data import DataLoader
import numpy as np

plot = False # if set to True, needs matplotlib
if plot:
    try:
        import matplotlib as plt
    except:
        print("Failed to import matplotlib, setting plot to False")
        plot = False

def main(args):
    dataset_load_path = args["dataset_as_np"]
    trained_model_save_path = args["trained_model_path"]
    print("Will load data from", dataset_load_path, ", train and save the model to, ",trained_model_save_path)

    ### DATASET:
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

    ### MODEL and training:

    X_latents = torch.from_numpy(X_latents).float()
    Y = torch.from_numpy(Y).float().unsqueeze(1)
    print("X latents:", X_latents.shape, X_latents.dtype)
    print("Y labels:", Y.shape, Y.dtype)

    train_loader = DataLoader(LilDataset(X_latents,Y), batch_size=8)
    model = LilModel()
    print(model)
    print(get_n_params(model))


    criterion = model.criterion
    optimizer = model.configure_optimizers()

    EPOCHS = 10
    N_samples = len(Y)
    PRINT_EVERY = int(N_samples/2)
    training_losses = []
    for epoch in range(EPOCHS):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

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
        training_losses.append(running_loss / N_samples)
    print('Finished Training')
    print("Losses:", training_losses)

    torch.save(model.state_dict(), trained_model_save_path)

    if plot:
        plt.plot(training_losses)
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser('Run training tile classifier')
    parser.add_argument('--dataset_as_np', default="dataset_as_np.npz",
                        help="Path to the prepared latents dataset.")
    parser.add_argument('--trained_model_path', default='../results/tile_model.pt',
                        help="Where to save the model weights")
    # parser.add_argument('--time-limit', type=int, default=300,
    #                     help="time limit for running inference [300]")

    args = vars(parser.parse_args())

    main(args)

