"""
check_gpu.py

Correr esto DESDE PYCHARM, con el mismo boton Run/interprete que usas
para train_unet.py -- para diagnosticar si el problema es el entorno
de Python de este proyecto especifico, no la GPU ni los drivers.
"""

import sys
import tensorflow as tf

print("=" * 60)
print("Python ejecutable usado:", sys.executable)
print("Version de Python:", sys.version)
print("=" * 60)
print("Version de TensorFlow:", tf.__version__)
print("TensorFlow compilado con soporte CUDA:", tf.test.is_built_with_cuda())

try:
    build_info = tf.sysconfig.get_build_info()
    print("CUDA version (build):", build_info.get("cuda_version", "N/A"))
    print("cuDNN version (build):", build_info.get("cudnn_version", "N/A"))
except Exception as e:
    print("No se pudo leer build_info:", e)

print("=" * 60)
gpus = tf.config.list_physical_devices('GPU')
print("GPUs detectadas por TensorFlow:", gpus)

if len(gpus) == 0:
    print("""
*** DIAGNOSTICO ***
TensorFlow NO detecta ninguna GPU en este entorno.

Posibles causas mas comunes:
1. Este proyecto de PyCharm usa un interprete/venv DISTINTO al que
   tenia TensorFlow con soporte GPU configurado. Compara la ruta de
   "Python ejecutable usado" (arriba) contra el interprete que usabas
   cuando SI te funciono la GPU antes.
2. Si tu TensorFlow es version 2.11 o mas reciente Y esto corre nativo
   en Windows (no en WSL2): desde TF 2.11, Windows nativo NO tiene
   soporte de GPU. Revisa si "CUDA version (build)" salio como N/A o
   vacio -- eso lo confirmaria.
3. Instalaste "tensorflow" (CPU-only en Windows nativo desde 2.11) en
   vez de configurar WSL2 o tensorflow-directml-plugin.
""")
else:
    print("\nOK: TensorFlow SI detecta GPU en este entorno.")
    print("Si el entrenamiento sigue sin usarla, revisa que no haya una")
    print("linea 'CUDA_VISIBLE_DEVICES=-1' o similar en el codigo o en")
    print("las variables de entorno del Run Configuration de PyCharm.")
