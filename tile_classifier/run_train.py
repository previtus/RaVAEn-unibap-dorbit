from dataset import generate_dataset
### DATASET:
X_latents, X_tiles, Y = generate_dataset()
print("Dataset:")
print("X latents:", X_latents.shape)
print("X tiles:", X_tiles.shape)
print("Y labels:", Y.shape)


### MODEL and training:

import torch
from model_pytorch import LilModel, LilDataset, get_n_params
from torch.utils.data import DataLoader
# import pytorch_lightning as pl

X_latents = torch.from_numpy(X_latents).float()
Y = torch.from_numpy(Y).float().unsqueeze(1)
print("X latents:", X_latents.shape, X_latents.dtype)
print("Y labels:", Y.shape, Y.dtype)

train_loader = DataLoader(LilDataset(X_latents,Y), batch_size=8)
# trainer = pl.Trainer(max_epochs=1)
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

torch.save(model.state_dict(), "tile_model.pt")

import pylab as plt
plt.plot(training_losses)
plt.show()

