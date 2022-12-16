from vis_functions import vis_image_with_predicted_labels
from dataset import available_files, file_to_tiles_data

from model_pytorch import LilModel
from dataset import tiles2latents
import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


from sklearn.metrics import precision_recall_curve, auc

def precision_recall(true_changes, pred_change_scores, mask=True):
    # convert to numpy arrays and mask invalid
    # After these lines the changes and scores are 1D
    if mask:
      invalid_masks = [c==2 for c in true_changes]
      true_changes = np.concatenate(
          [c[~m] for m,c in zip(invalid_masks, true_changes)],
          axis=0
      )
      pred_change_scores = np.concatenate(
          [c[~m] for m,c in zip(invalid_masks, pred_change_scores)],
          axis=0
      )
    else:
      # else just flatten
      true_changes = true_changes.flatten()
      pred_change_scores = pred_change_scores.flatten()

    precision, recall, thresholds = precision_recall_curve(
        true_changes,
        pred_change_scores
    )
    return precision, recall, thresholds

if __name__ == "__main__":
    dataset_dir = "../unibap_dataset"

    model = LilModel()
    # tile_model_path = "../results/tile_model.pt"
    RESULTS_DIR = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/results/step1_logs_unibap/results15c_oncemore/"
    tile_model_path = RESULTS_DIR+"tile_model_256batch.pt"

    model.load_state_dict(torch.load(tile_model_path))
    model.eval()

    all_images = available_files(dataset_dir)
    image_ids = [path.split("/")[-1].split("_")[2] for path in all_images]

    exclude_train_set_tiles = [104, 743, 448, 127, 358, 642] + [114] # train set used
    import sys
    sys.path.append('../tile_labeller')
    from read_labels import dataset_from_manual_annotation
    PATH = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/codes/RaVAEn-unibap-dorbit/tile_annotation/l1/"
    X_latents_test, X_tiles_test, Y_test = dataset_from_manual_annotation(PATH, exclude_train_set_tiles)

    print("Test dataset:")
    print("X latents (test):", X_latents_test.shape)
    print("X tiles (test):", X_tiles_test.shape)
    print("Y labels (test):", Y_test.shape)

    # Predict:
    X_latents = torch.from_numpy(X_latents_test).float()
    print("X latents:", X_latents.shape)

    Y_predictions = model(X_latents)
    print("Y predictions:", Y_predictions.shape)

    Y_predictions = Y_predictions[:,0].detach().numpy()
    print("P",Y_predictions[0:5])
    print("GT",Y_test[0:5])

    precision, recall, thresholds = precision_recall(Y_test, Y_predictions, mask=False)
    area_under_precision_curve = auc(recall, precision)
    print("AUPRC = ", area_under_precision_curve)

    THR = 0.5
    Y_predictions[Y_predictions >= THR] = 1
    Y_predictions[Y_predictions < THR] = 0

    # print("Y predictions:", Y_predictions.shape)


    report = classification_report(Y_test, Y_predictions)
    print(report)

    classes = ["NonCloud","Cloud"]
    cm = confusion_matrix(Y_test, Y_predictions) # , labels=classes
    tn, fp, fn, tp = cm.ravel()
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * (recall * precision) / (recall + precision)
    print("Recall", recall, "Precision", precision, "F1", f1)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = classes)
    disp.plot()
    plt.show()
