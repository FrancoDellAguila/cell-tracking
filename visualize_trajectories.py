import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
from tifffile import imread
from pathlib import Path
import os
import sys
from scipy.ndimage import center_of_mass
import matplotlib.colors as mcolors

def visualize_trajectories(img_path, results_path, save_dir=None, trajectory_length=15):
    """
    Visualiza las trayectorias celulares sobre las imágenes originales.
    
    Args:
        img_path: Ruta a las imágenes originales.
        results_path: Ruta a los resultados del seguimiento (contiene archivos de máscara).
        save_dir: Opcional. Ruta para guardar los fotogramas como imágenes.
        trajectory_length: Número de fotogramas pasados para dibujar en la trayectoria.
    """
    # Convertir rutas de entrada a objetos Path para un manejo más fácil
    img_path_obj = Path(img_path) 
    results_path_obj = Path(results_path) 
    print(f"Buscando imágenes originales en: {img_path_obj}")
    print(f"Buscando archivos de máscara en: {results_path_obj}")
    
    # Verificar si las rutas de entrada existen
    if not img_path_obj.exists() or not results_path_obj.exists():
        print(f"ERROR: La ruta de imágenes o de resultados no existe.")
        return
        
    # Encontrar archivos de imágenes originales (intentar con patrón 't*' y luego cualquier nombre)
    img_files = []
    for ext in ['tif', 'tiff', 'png', 'jpg']:
        files = sorted(list(img_path_obj.glob(f't*.{ext}')))
        if files:
            img_files = files
            print(f"Se encontraron {len(files)} archivos de imagen con extensión .{ext} (patrón t***)")
            break
    if not img_files: # Si no se encuentran con 't*', buscar cualquier nombre
        for ext in ['tif', 'tiff', 'png', 'jpg']:
            files = sorted(list(img_path_obj.glob(f'*.{ext}')))
            if files:
                img_files = files
                print(f"Se encontraron {len(files)} archivos de imagen con extensión .{ext} (cualquier nombre)")
                break
    
    # Encontrar archivos de máscara de segmentación
    mask_files = []
    for ext in ['tif', 'tiff', 'png']:
        files = sorted(list(results_path_obj.glob(f'mask*.{ext}')))
        if files:
            mask_files = files
            print(f"Se encontraron {len(files)} archivos de máscara con extensión .{ext}")
            break
    
    # Validar que se encontraron ambos tipos de archivos
    if not img_files or not mask_files:
        print("No se pudieron encontrar los archivos de imagen o de máscara requeridos.")
        return
        
    # Usar el número mínimo de fotogramas si las secuencias tienen longitudes diferentes
    min_frames = min(len(img_files), len(mask_files))
    if min_frames == 0:
        print("ERROR: No se encontraron fotogramas coincidentes entre imágenes y máscaras.")
        return
        
    # Recortar las listas de archivos al mínimo de fotogramas
    img_files = img_files[:min_frames]
    mask_files = mask_files[:min_frames]
        
    print(f"Procesando {min_frames} fotogramas coincidentes.")
    
    # Encontrar el ID máximo de célula en todas las máscaras para definir el mapa de colores
    max_id = 0
    print("Analizando todos los fotogramas para encontrar el ID máximo de célula...")
    for mask_file in mask_files:
        try:
            mask = imread(str(mask_file))
            max_id = max(max_id, np.max(mask))
        except Exception as e:
            print(f"Advertencia: Error al leer {mask_file}: {e}")
            
    print(f"ID máximo de célula encontrado: {max_id}")
    
    # Crear un mapa de colores: un color único para cada ID de célula
    colors = plt.cm.jet(np.linspace(0, 1, int(max_id) + 1))
    
    # Calcular los centroides de todas las células en todos los fotogramas
    print("Calculando centroides de células en todos los fotogramas...")
    centroids = {cell_id: [] for cell_id in range(1, int(max_id) + 1)} # Diccionario para almacenar centroides por ID
    
    for frame_idx, mask_file in enumerate(mask_files):
        try:
            mask = imread(str(mask_file))
            for cell_id in np.unique(mask): # Iterar sobre IDs únicos en la máscara
                if cell_id == 0:  # Ignorar el fondo
                    continue
                cell_mask = (mask == cell_id) # Máscara booleana para la célula actual
                if np.any(cell_mask): # Asegurarse de que la célula tiene píxeles
                    y, x = center_of_mass(cell_mask) # Calcular centro de masa
                    centroids[cell_id].append((frame_idx, int(x), int(y))) # Guardar (fotograma, x, y)
        except Exception as e:
            print(f"Advertencia: No se pudieron calcular centroides para {mask_file.name}: {e}")

    # Determinar nombres para el directorio de salida basados en las rutas de entrada
    dataset_folder_name = img_path_obj.parent.name  # Nombre de la carpeta del dataset (ej: BF-C2DL-HSC)
    sequence_folder_name = img_path_obj.name       # Nombre de la secuencia (ej: 01)
    descriptive_name_part = f"{dataset_folder_name}_{sequence_folder_name}" # Parte descriptiva para nombres de archivo/directorio

    final_save_path = None
    filename_prefix_str = "" 

    # Configurar la ruta de guardado y el prefijo del nombre de archivo
    if save_dir: # Si el usuario especificó un directorio de guardado
        final_save_path = Path(save_dir)
        filename_prefix_str = f"{descriptive_name_part}_" # Incluye el guion bajo y la parte descriptiva
    else: # Si no se especificó, crear un directorio por defecto
        default_output_dir_name = f"visualized_trajectories_{descriptive_name_part}" 
        final_save_path = Path(default_output_dir_name)
        # filename_prefix_str permanece vacío, los nombres de archivo serán más simples

    os.makedirs(final_save_path, exist_ok=True) # Crear el directorio de salida si no existe
    print(f"Guardando imágenes de trayectoria en: {final_save_path}")
    
    # Bucle principal: procesar cada fotograma para generar la visualización
    print("Generando visualizaciones de trayectoria...")
    for frame_idx, (img_file, mask_file) in enumerate(zip(img_files, mask_files)):
        try:
            img = imread(str(img_file)) # Cargar imagen original
            mask = imread(str(mask_file)) # Cargar máscara de segmentación
            
            # Normalizar la imagen original para visualización si no es uint8
            if img.dtype != np.uint8:
                p_low, p_high = np.percentile(img, (1, 99)) # Usar percentiles para robustez a outliers
                img_norm = (img - p_low) / (p_high - p_low + 1e-6) # Evitar división por cero
                img_display = np.clip(img_norm * 255, 0, 255).astype(np.uint8) # Convertir a uint8
            else:
                img_display = img
            
            # Convertir imagen a RGB si es necesario (para superponer colores)
            if len(img_display.shape) == 2: # Si es escala de grises
                img_rgb = np.stack([img_display, img_display, img_display], axis=2)
            elif img_display.shape[2] == 1: # Si es escala de grises con un canal singleton
                 img_rgb = np.concatenate([img_display] * 3, axis=-1)
            else: # Si ya es RGB (o tiene más de 3 canales, tomar los primeros 3)
                img_rgb = img_display[:,:,:3] 

            # Crear la figura y los ejes para el gráfico
            fig, ax = plt.subplots(figsize=(11, 9)) 
            ax.imshow(img_rgb) # Mostrar la imagen original
            
            # Obtener IDs de células presentes en el fotograma actual
            unique_ids = np.unique(mask)
            unique_ids = unique_ids[unique_ids > 0]  # Excluir el fondo (ID 0)
            
            # Dibujar las trayectorias para cada célula
            for cell_id in unique_ids:
                if cell_id not in centroids: continue # Saltar si no hay centroides para este ID (raro)

                cell_centroids = centroids[cell_id] # Obtener todos los centroides para esta célula
                # Filtrar centroides hasta el fotograma actual
                relevant_centroids = [c for c in cell_centroids if c[0] <= frame_idx]
                # Tomar los últimos 'trajectory_length' puntos para la estela
                trajectory_points = relevant_centroids[-trajectory_length:]
                
                # Dibujar la línea de trayectoria si hay al menos 2 puntos
                if len(trajectory_points) > 1:
                    frames = [c[0] for c in trajectory_points]
                    x_coords = [c[1] for c in trajectory_points]
                    y_coords = [c[2] for c in trajectory_points]
                    
                    num_segments = len(trajectory_points) - 1
                    # Dibujar segmentos de línea con alfa creciente para efecto de "estela"
                    for i_segment in range(num_segments): 
                        alpha = 0.2 + 0.8 * ((i_segment + 1) / num_segments) # Alfa aumenta hacia el punto actual
                        ax.plot(x_coords[i_segment:i_segment+2], y_coords[i_segment:i_segment+2], 
                                 color=colors[int(cell_id)], linewidth=1.8, 
                                 alpha=alpha,
                                 solid_capstyle='round') # Estilo de línea
                
                # Dibujar un círculo en la posición actual de la célula
                if trajectory_points:
                    current_x, current_y = trajectory_points[-1][1], trajectory_points[-1][2]
                    ax.plot(current_x, current_y, 'o', # Marcador circular
                             color=colors[int(cell_id)], 
                             markersize=6,  
                             markeredgecolor='black', # Borde para mejor visibilidad
                             markeredgewidth=0.6,
                             alpha=0.8)  
            
            ax.set_title(f'Fotograma {frame_idx+1} / {min_frames}') # Título del gráfico
            ax.axis('off') # Ocultar ejes
            
            # Guardar la figura
            output_filename = f'{filename_prefix_str}trayectoria_{frame_idx+1:04d}.png'
            output_file_path = final_save_path / output_filename
            plt.savefig(output_file_path, bbox_inches='tight', dpi=120) # Guardar con buen DPI

            plt.close(fig) # Cerrar la figura para liberar memoria
        except Exception as e:
            print(f"ERROR procesando fotograma {frame_idx}: {e}")
            if 'fig' in locals() and plt.fignum_exists(fig.number): # Asegurarse de cerrar la figura en caso de error
                 plt.close(fig) 

    print("Visualización de trayectorias finalizada.")

# Bloque principal para ejecutar el script desde la línea de comandos
if __name__ == "__main__":
    import argparse
    # Configurar el analizador de argumentos
    parser = argparse.ArgumentParser(description="Visualiza trayectorias de seguimiento celular sobre las imágenes originales.")
    parser.add_argument("img_path", help="Ruta a la carpeta con la secuencia de imágenes originales (ej: t001.tif)")
    parser.add_argument("results_path", help="Ruta a la carpeta con los resultados del seguimiento (archivos mask*.tif)")
    parser.add_argument("--save", dest="output_dir", help="Directorio opcional para guardar los fotogramas de salida.")
    parser.add_argument("--length", type=int, default=15, help="Longitud de la estela de la trayectoria (fotogramas). Predeterminado: 15")
    
    args = parser.parse_args() # Analizar los argumentos
    
    # Llamar a la función principal con los argumentos proporcionados
    visualize_trajectories(args.img_path, args.results_path, args.output_dir, args.length)