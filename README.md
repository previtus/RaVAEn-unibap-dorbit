# RaVAEn-unibap-dorbit
Running the RaVAEn model on board of D-Orbit satellites

## Intro

We have deployed a variational auto-encoder (VAE) model called **RaVAEn** (https://github.com/spaceml-org/RaVAEn) on-board of a satellite to measure its real-world inference times. This model has been pre-trained in an unsupervised manner on the task of data reconstruction, while encoding the data through a smaller bottleneck of latent vectors. We have presented its application for unsupervised change detection in the context of disaster detection, we furthermore consider it as a foundational model in the sense that it was pre-trained in a task agnostic manner. 

We show that this model can be reliably used with the compute available directly on-board of the D-Orbit’s ION-SCV 004 satellite. We report an encoding time of 0.110s for encoding tiles of a 4.8x4.8 km square area, using the RGB+NIR bands of the Sentinel-2 data (10m spatial resolution).

In addition, we also demonstrate to the best of our knowledge the world’s first fast and efficient few-shot training on-board of a satellite using the latent representation of the data. To this end, we use the learned encoder of the VAE model to represent tiles of 32x32 pixels with 4 bands as a 128-dimensional latent vectors. We then train a lightweight classification model using these latent vectors as inputs in a few-shot learning manner. Good representation of the Sentinel-2 data is required for training with only limited number of samples.

## Publication

If you this work useful in your research, please consider citing our paper _"Fast model inference and training on-board of Satellites"_ presented at the IEEE IGARSS 2023: https://2023.ieeeigarss.org/view_paper.php?PaperNum=5969

We also have the paper pre-print available at https://arxiv.org/abs/2307.08700:

> Růžička, V., Mateo-García, G., Bridges, C., Brunskill, C., Purcell, C., Longépé, N., & Markham, A. (2023). Fast model inference and training on-board of Satellites. arXiv preprint arXiv:2307.08700.

