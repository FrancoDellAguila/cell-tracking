# Contexto del Proyecto
Este repositorio se basa en el código para un método de segmentación y seguimiento de células desarrollado en el Karlsruhe Institute of Technology (KIT), específicamente por el equipo KIT-Sch-GE. El enfoque se basa en predicciones de distancia mediante Redes Neuronales Convolucionales (CNN) y una estrategia de emparejamiento basada en grafos para el seguimiento. Este trabajo fue la base de una participación en la 5ª edición del ISBI Cell Tracking Challenge en 2020 y ha sido publicado.

# Segmentación y Seguimiento Celular usando Predicciones de Distancia Basadas en CNN y una Estrategia de Emparejamiento Basada en Grafos

Método de segmentación y seguimiento utilizado para nuestra [publicación](#publicación). Nuestra participación en la 5ª edición del [ISBI Cell Tracking Challenge](http://celltrackingchallenge.net/) 2020 se basa en este código (Equipo KIT-Sch-GE).

Una versión mejorada de la segmentación (escalado y cierre ligeramente ajustados para las distancias a vecinos, proceso de entrenamiento, tamaño de lote > 1 y soporte multi-GPU para inferencia) se puede encontrar aquí: [https://git.scc.kit.edu/KIT-Sch-GE](https://git.scc.kit.edu/KIT-Sch-GE).

## Prerrequisitos
* [Distribución Anaconda](https://www.anaconda.com/products/individual)
* Una GPU compatible con CUDA
* RAM mínima / recomendada: 16 GiB / 32 GiB
* VRAM mínima / recomendada: 12 GiB / 24 GiB

## Instalación
Clona el repositorio Cell Tracking:
```
git clone https://bitbucket.org/t_scherr/cell-segmentation-and-tracking
```
Abre Anaconda Prompt (Windows) o la Terminal (Linux), ve al repositorio Cell Segmentation and Tracking y crea un nuevo entorno virtual:
```
cd ruta_al_repositorio_clonado
conda env create -f requirements.yml
```
Activa el entorno virtual cell_segmentation_and_tracking_ve:
```
conda activate cell_segmentation_and_tracking_ve
```

## Tracking
Esta sección describe cómo reproducir los resultados de seguimiento de nuestra [publicación](#publicación).

Se asume que el conjunto de datos sigue la misma estructura de carpetas y nomenclatura de archivos que los conjuntos de datos CTC:
```
dataset
└───01
└───01_GT
└───01_RES
└───02
└───02_GT
└───02_RES
```
Primero, ejecuta un enfoque de segmentación para derivar máscaras de segmentación para dataset/0x, donde x es 1 o 2.
El seguimiento con la misma parametrización que en el artículo se puede derivar ejecutando:
```
python run_tracking.py ruta_img ruta_segm ruta_res
```
donde `ruta_img` es la ruta a la carpeta dataset/0x y `ruta_segm` la ruta a la carpeta que contiene las máscaras de segmentación. Las máscaras de seguimiento resultantes y el archivo de linaje se almacenarán en `ruta_res`.
Si `ruta_segm` y `ruta_res` son la misma ruta, las máscaras de segmentación serán reemplazadas por las máscaras de seguimiento.

## Visualización de resultados
Esta sección describe cómo visualizar los resultados del seguimiento y las trayectorias.

### Visualizar Máscaras de Seguimiento
Para visualizar las máscaras de seguimiento generadas (archivos `maskXXX.tif` en la carpeta de resultados):
```
python visualize_tracking.py ruta_a_tus_resultados_RES [--save ruta_directorio_salida]
```
*   `ruta_a_tus_resultados_RES`: Ruta a la carpeta que contiene los archivos `maskXXX.tif` (por ejemplo, `datasets/BF-C2DL-HSC/01_RES`).
*   `--save ruta_directorio_salida` (opcional): Directorio donde se guardarán las imágenes de visualización. Si no se especifica, se creará un directorio llamado `visualized_tracking_NOMBRE-DATASET_NOMBRE-SECUENCIA` en la ubicación actual.

Ejemplo:
```
python visualize_tracking.py datasets/BF-C2DL-HSC/01_RES --save visualizaciones/BF-C2DL-HSC_01_tracking
```

### Visualizar Trayectorias
Para visualizar las trayectorias de las células superpuestas en las imágenes originales:
```
python visualize_trajectories.py ruta_imagenes_originales ruta_a_tus_resultados_RES [--save ruta_directorio_salida] [--length N]
```
*   `ruta_imagenes_originales`: Ruta a la carpeta que contiene la secuencia de imágenes originales (por ejemplo, `datasets/BF-C2DL-HSC/01`).
*   `ruta_a_tus_resultados_RES`: Ruta a la carpeta que contiene los archivos `maskXXX.tif` (por ejemplo, `datasets/BF-C2DL-HSC/01_RES`).
*   `--save ruta_directorio_salida` (opcional): Directorio donde se guardarán las imágenes de visualización. Si no se especifica, se creará un directorio llamado `visualized_trajectories_NOMBRE-DATASET_NOMBRE-SECUENCIA` en la ubicación actual.
*   `--length N` (opcional): Número de fotogramas pasados para dibujar en la trayectoria. Por defecto es 15.

Ejemplo:
```
python visualize_trajectories.py datasets/BF-C2DL-HSC/01 datasets/BF-C2DL-HSC/01_RES --save visualizaciones/BF-C2DL-HSC_01_trajectories --length 10
```

## Publicación
T. Scherr, K. Löffler, M. Böhland, and R. Mikut (2020). Cell Segmentation and Tracking using CNN-Based Distance Predictions and a Graph-Based Matching Strategy. PLoS ONE 15(12). DOI: [10.1371/journal.pone.0243219](https://doi.org/10.1371/journal.pone.0243219).

## Licencia
Este proyecto está licenciado bajo la Licencia MIT - consulta el archivo [LICENSE.md](LICENSE.md) para más detalles.