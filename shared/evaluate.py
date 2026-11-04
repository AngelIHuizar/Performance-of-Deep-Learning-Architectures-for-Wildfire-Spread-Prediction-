"""
shared/evaluate.py

UNA SOLA funcion de evaluacion, usada por los 4 modelos -- incluyendo xLSTM
en PyTorch, gracias a que recibe una funcion "predict_fn" generica en vez
de asumir un modelo de Keras.

Por que esto importa: la auditoria encontro que U-Net se evaluo sobre
validation_files (mal etiquetado como "test"), mientras que Autoencoder y
ConvLSTM se evaluaron sobre test_files. Con una sola funcion, ese tipo de
error deja de ser posible: los 4 modelos llaman exactamente al mismo codigo
sobre exactamente el mismo dataset de test.

NOTA IMPORTANTE SIN RESOLVER (pendiente de verificar con la documentacion
del dataset NDWS): no encontramos un "fire_id" explicito en los TFRecords.
Esta funcion agrega por "sample_id" (indice de muestra). Si cada muestra
del dataset corresponde a un incendio unico, sample_id == fire_id y no hay
problema. Si un mismo incendio puede aparecer en varias muestras (distintos
dias), hay que resolver primero un mapeo real fire_id -> sample antes de
usar esto para el bootstrap de la Semana 2. NO asumir, verificar.
"""

from typing import Callable, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from shared.config import THRESHOLD, ONLY_FIRE_PIXELS


def evaluate_model_unified(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    eval_batches: Iterable[Tuple[np.ndarray, np.ndarray]],
    sample_ids: Optional[Iterable] = None,
    threshold: float = THRESHOLD,
    only_fire_pixels: bool = ONLY_FIRE_PIXELS,
) -> pd.DataFrame:
    """
    Args:
        predict_fn: funcion que recibe un batch de inputs (np.ndarray) y
            devuelve probabilidades (np.ndarray), sin importar si el modelo
            es de Keras o PyTorch. Ejemplos de como envolver cada framework
            estan al final de este archivo.
        eval_batches: iterable de tuplas (inputs, labels) como numpy arrays,
            YA extraidas del dataset de TEST (nunca de validation).
        sample_ids: identificadores opcionales, uno por muestra, en el mismo
            orden que eval_batches produce las muestras. Si es None, se usa
            un indice secuencial (0, 1, 2, ...) -- documentar esta decision
            explicitamente en el articulo si no hay fire_id real disponible.
        threshold: umbral de binarizacion, fijo y compartido (ver config.py)
        only_fire_pixels: si True, descarta muestras sin pixeles de
            incendio en la mascara real. DEBE ser el mismo valor para los
            4 modelos -- no se decide por script.

    Returns:
        pd.DataFrame con una fila por muestra: sample_id, tp, fp, tn, fn,
        target_pixels, accuracy, precision, recall, iou, f1
    """
    eps = 1e-7
    rows = []
    running_idx = 0

    for inputs, labels in eval_batches:
        predictions = predict_fn(inputs)
        predictions_binary = (predictions > threshold).astype("int32")
        labels_int = labels.astype("int32")

        batch_size = labels_int.shape[0]
        for i in range(batch_size):
            y_true = labels_int[i].flatten()
            y_pred = predictions_binary[i].flatten()

            if only_fire_pixels and y_true.sum() == 0:
                running_idx += 1
                continue

            tp = int(np.sum((y_true == 1) & (y_pred == 1)))
            fp = int(np.sum((y_true == 0) & (y_pred == 1)))
            tn = int(np.sum((y_true == 0) & (y_pred == 0)))
            fn = int(np.sum((y_true == 1) & (y_pred == 0)))
            total = tp + fp + tn + fn

            sid = sample_ids[running_idx] if sample_ids is not None else running_idx

            rows.append({
                "sample_id": sid,
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "target_pixels": tp + fn,
                "n_pixels": total,
                "accuracy": (tp + tn) / (total + eps),
                "precision": tp / (tp + fp + eps),
                "recall": tp / (tp + fn + eps),
                "iou": tp / (tp + fp + fn + eps),
                "f1": 2 * (tp / (tp + fp + eps)) * (tp / (tp + fn + eps))
                      / ((tp / (tp + fp + eps)) + (tp / (tp + fn + eps)) + eps),
            })
            running_idx += 1

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Wrappers de ejemplo para cada framework -- usar en cada train_*.py
# ---------------------------------------------------------------------

def make_keras_predict_fn(keras_model):
    """Para U-Net, Autoencoder, ConvLSTM (TensorFlow/Keras)."""
    def predict_fn(inputs_np: np.ndarray) -> np.ndarray:
        return keras_model.predict(inputs_np, verbose=0)
    return predict_fn


def make_pytorch_predict_fn(torch_model, device):
    """Para xLSTM (PyTorch). Requiere torch importado en el script que
    llama a esto (no se importa aqui para no forzar la dependencia)."""
    import torch

    def predict_fn(inputs_np: np.ndarray) -> np.ndarray:
        torch_model.eval()
        with torch.no_grad():
            x = torch.from_numpy(inputs_np).float().to(device)
            out = torch_model(x)
            probs = torch.sigmoid(out) if out.min() < 0 or out.max() > 1 else out
            return probs.cpu().numpy()
    return predict_fn


def keras_dataset_to_batches(tf_dataset) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """Convierte un tf.data.Dataset en el formato (inputs, labels) numpy
    que espera evaluate_model_unified, para U-Net/Autoencoder/ConvLSTM."""
    for inputs, labels in tf_dataset.as_numpy_iterator():
        yield inputs, labels
