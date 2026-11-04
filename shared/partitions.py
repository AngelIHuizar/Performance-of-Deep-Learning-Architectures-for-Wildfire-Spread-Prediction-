"""
shared/partitions.py

UNA SOLA funcion de particion de datos, usada por los 4 scripts.

Por que esto importa: la auditoria encontro que xLSTM usaba una heuristica
de nombres distinta ("train"/"val"/"test") a la de los otros 3 scripts
("train"/"eval"/"test"). Funciono por coincidencia (porque "eval" contiene
"val" como substring), pero fue suerte, no diseno. Esta version es EXPLICITA
y falla ruidosamente en vez de caer en un fallback silencioso.
"""

import glob
import os
from typing import List, Tuple

from shared.config import DATASET_PATH, TRAIN_SUBSTR, VAL_SUBSTR, TEST_SUBSTR


def get_train_val_test_files(
    dataset_path: str = DATASET_PATH,
    verbose: bool = True,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Devuelve (train_files, val_files, test_files) usando SIEMPRE la misma
    heuristica de nombres, para que los 4 modelos evaluen sobre exactamente
    los mismos archivos.

    NO tiene fallback silencioso. Si algo no cuadra, lanza un error para que
    lo veas de inmediato, en vez de que el script siga corriendo con una
    particion distinta sin que te des cuenta (esto fue justo el riesgo que
    encontramos en el script de xLSTM).
    """
    all_files = sorted(glob.glob(os.path.join(dataset_path, "*.tfrecord")))

    if len(all_files) == 0:
        raise FileNotFoundError(
            f"No se encontraron archivos .tfrecord en: {dataset_path}\n"
            f"Verifica DATASET_PATH en shared/config.py"
        )

    train_files = sorted(
        f for f in all_files if TRAIN_SUBSTR in os.path.basename(f)
    )
    val_files = sorted(
        f for f in all_files if VAL_SUBSTR in os.path.basename(f)
    )
    test_files = sorted(
        f for f in all_files if TEST_SUBSTR in os.path.basename(f)
    )

    # ------------------------------------------------------------------
    # Verificaciones explicitas -- fallan ruidosamente si algo no cuadra
    # ------------------------------------------------------------------
    total_clasificado = len(train_files) + len(val_files) + len(test_files)

    if total_clasificado != len(all_files):
        raise ValueError(
            f"INCONSISTENCIA: {len(all_files)} archivos totales, pero solo "
            f"{total_clasificado} quedaron clasificados en train/val/test.\n"
            f"Revisa si hay archivos con nombres que no matchean ninguna "
            f"categoria, o que matchean mas de una (superposicion)."
        )

    overlap_train_val = set(train_files) & set(val_files)
    overlap_train_test = set(train_files) & set(test_files)
    overlap_val_test = set(val_files) & set(test_files)

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise ValueError(
            "INCONSISTENCIA: un mismo archivo quedo clasificado en mas de "
            "una particion. Esto es fuga de datos garantizada. Revisa los "
            "nombres de archivo que se solapan."
        )

    if len(train_files) == 0 or len(val_files) == 0 or len(test_files) == 0:
        raise ValueError(
            f"Alguna particion quedo vacia (train={len(train_files)}, "
            f"val={len(val_files)}, test={len(test_files)}). "
            f"NO se activa ningun fallback automatico -- decide manualmente "
            f"que hacer en este caso, no dejes que el script decida por ti."
        )

    if verbose:
        print(f"Total archivos: {len(all_files)}")
        print(f"  train: {len(train_files)}")
        print(f"  val:   {len(val_files)}")
        print(f"  test:  {len(test_files)}")

    return train_files, val_files, test_files


if __name__ == "__main__":
    # Ejecutar este archivo solo (python -m shared.partitions) para
    # verificar la particion antes de correr cualquier entrenamiento.
    get_train_val_test_files()
