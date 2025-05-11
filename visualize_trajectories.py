import matplotlib.pyplot as plt
import numpy as np
from tifffile import imread
from pathlib import Path
import os
import sys
from scipy.ndimage import center_of_mass
import matplotlib.colors as mcolors
from datetime import datetime # Asegúrate de que esta importación esté presente

def visualize_trajectories(img_path, results_path, save_dir=None, trajectory_length=15):
    """
    Visualiza las trayectorias celulares sobre las imágenes originales.
    
    Args:
        img_path: Ruta a las imágenes originales.
        results_path: Ruta a los resultados del seguimiento (contiene archivos de máscara).
        save_dir: Opcional. Ruta para guardar los fotogramas como imágenes.
        trajectory_length: Número de fotogramas pasados para dibujar en la trayectoria.
    """
    img_path_obj = Path(img_path) # Renombrado para claridad, o puedes seguir usando img_path
    results_path_obj = Path(results_path) # Renombrado para claridad
    print(f"Buscando imágenes originales en: {img_path_obj}")
    print(f"Buscando archivos de máscara en: {results_path_obj}")
    
    # Verificar si las rutas existen
    if not img_path_obj.exists() or not results_path_obj.exists():
        print(f"ERROR: La ruta de imágenes o de resultados no existe.")
        return
        
    # Encontrar imágenes originales
    img_files = []
    for ext in ['tif', 'tiff', 'png', 'jpg']:
        # Buscar primero el patrón t***.ext
        files = sorted(list(img_path_obj.glob(f't*.{ext}')))
        if files:
            img_files = files
            print(f"Se encontraron {len(files)} archivos de imagen con extensión .{ext} (patrón t***)")
            break
            
    if not img_files:
        # Si no se encuentra t***, buscar cualquier nombre *.ext
        for ext in ['tif', 'tiff', 'png', 'jpg']:
            files = sorted(list(img_path_obj.glob(f'*.{ext}')))
            if files:
                img_files = files
                print(f"Se encontraron {len(files)} archivos de imagen con extensión .{ext} (cualquier nombre)")
                break
    
    # Encontrar archivos de máscara
    mask_files = []
    for ext in ['tif', 'tiff', 'png']:
        files = sorted(list(results_path_obj.glob(f'mask*.{ext}')))
        if files:
            mask_files = files
            print(f"Se encontraron {len(files)} archivos de máscara con extensión .{ext}")
            break
    
    if not img_files or not mask_files:
        print("No se pudieron encontrar los archivos de imagen o de máscara requeridos.")
        return
        
    min_frames = min(len(img_files), len(mask_files))
    if min_frames == 0:
        print("ERROR: No se encontraron fotogramas coincidentes entre imágenes y máscaras.")
        return
        
    img_files = img_files[:min_frames]
    mask_files = mask_files[:min_frames]
        
    print(f"Procesando {min_frames} fotogramas coincidentes.")
    
    max_id = 0
    print("Analizando todos los fotogramas para encontrar el ID máximo de célula...")
    for mask_file in mask_files:
        try:
            mask = imread(str(mask_file))
            max_id = max(max_id, np.max(mask))
        except Exception as e:
            print(f"Advertencia: Error al leer {mask_file}: {e}")
            
    print(f"ID máximo de célula encontrado: {max_id}")
    
    colors = plt.cm.jet(np.linspace(0, 1, int(max_id) + 1))
    
    print("Calculando centroides de células en todos los fotogramas...")
    centroids = {cell_id: [] for cell_id in range(1, int(max_id) + 1)}
    
    for frame_idx, mask_file in enumerate(mask_files):
        try:
            mask = imread(str(mask_file))
            for cell_id in np.unique(mask):
                if cell_id == 0: 
                    continue
                cell_mask = (mask == cell_id)
                if np.any(cell_mask):
                    y, x = center_of_mass(cell_mask)
                    centroids[cell_id].append((frame_idx, int(x), int(y)))
        except Exception as e:
            print(f"Advertencia: No se pudieron calcular centroides para {mask_file.name}: {e}")

    # Determinar identificadores de dataset y secuencia para nombres
    # Si img_path_obj es "D:\cell-tracking\datasets\BF-C2DL-HSC\01"
    # dataset_folder_name será "BF-C2DL-HSC"
    # sequence_folder_name será "01"
    dataset_folder_name = img_path_obj.parent.name  
    sequence_folder_name = img_path_obj.name       
    descriptive_name_part = f"{dataset_folder_name}_{sequence_folder_name}"

    final_save_path = None
    output_image_prefix = ""

    if save_dir: # Si el usuario proporcionó un directorio
        final_save_path = Path(save_dir)
        # Prefijar imágenes con el nombre descriptivo para distinguir si múltiples datasets/secuencias
        # se guardan en el mismo directorio personalizado.
        output_image_prefix = f"{descriptive_name_part}_" 
    else: # Si el usuario no proporcionó un directorio, crear uno por defecto
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # El nombre del directorio por defecto incluye la parte descriptiva y la marca de tiempo para unicidad.
        default_output_dir_name = f"visualized_trajectories_{descriptive_name_part}_{timestamp}"
        final_save_path = Path(default_output_dir_name)
        # No se necesita prefijo para las imágenes si el nombre del directorio ya es único y por defecto.
        # output_image_prefix permanece ""

    os.makedirs(final_save_path, exist_ok=True)
    print(f"Guardando imágenes de trayectoria en: {final_save_path}")
    
    print("Generando visualizaciones de trayectoria...")
    for frame_idx, (img_file, mask_file) in enumerate(zip(img_files, mask_files)):
        try:
            img = imread(str(img_file))
            mask = imread(str(mask_file)) 
            
            if img.dtype != np.uint8:
                p_low, p_high = np.percentile(img, (1, 99))
                img_norm = (img - p_low) / (p_high - p_low + 1e-6) 
                img_display = np.clip(img_norm * 255, 0, 255).astype(np.uint8)
            else:
                img_display = img
            
            if len(img_display.shape) == 2:
                img_rgb = np.stack([img_display, img_display, img_display], axis=2)
            elif img_display.shape[2] == 1: 
                 img_rgb = np.concatenate([img_display] * 3, axis=-1)
            else: 
                img_rgb = img_display[:,:,:3] 

            fig, ax = plt.subplots(figsize=(11, 9)) 
            ax.imshow(img_rgb)
            
            unique_ids = np.unique(mask)
            unique_ids = unique_ids[unique_ids > 0]  
            
            for cell_id in unique_ids:
                if cell_id not in centroids: continue 

                cell_centroids = centroids[cell_id]
                relevant_centroids = [c for c in cell_centroids if c[0] <= frame_idx]
                trajectory_points = relevant_centroids[-trajectory_length:]
                
                if len(trajectory_points) > 1:
                    frames = [c[0] for c in trajectory_points]
                    x_coords = [c[1] for c in trajectory_points]
                    y_coords = [c[2] for c in trajectory_points]
                    
                    num_segments = len(trajectory_points) - 1
                    for i in range(num_segments):
                        alpha = 0.2 + 0.8 * ((i + 1) / num_segments)
                        ax.plot(x_coords[i:i+2], y_coords[i:i+2], 
                                 color=colors[int(cell_id)], linewidth=1.8, 
                                 alpha=alpha,
                                 solid_capstyle='round') 
                
                if trajectory_points:
                    current_x, current_y = trajectory_points[-1][1], trajectory_points[-1][2]
                    ax.plot(current_x, current_y, 'o', 
                             color=colors[int(cell_id)], 
                             markersize=6,  
                             markeredgecolor='black', 
                             markeredgewidth=0.6,
                             alpha=0.8)  
            
            ax.set_title(f'Fotograma {frame_idx+1} / {min_frames}')
            ax.axis('off')
            
            output_file = final_save_path / f'{output_image_prefix}trayectoria_{frame_idx+1:04d}.png'
            plt.savefig(output_file, bbox_inches='tight', dpi=120)
            plt.close(fig) 
        except Exception as e:
            print(f"ERROR procesando fotograma {frame_idx}: {e}")
            if 'fig' in locals() and plt.fignum_exists(fig.number): 
                 plt.close(fig) 

    print("Visualización de trayectorias finalizada.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualiza trayectorias de seguimiento celular sobre las imágenes originales.")
    parser.add_argument("img_path", help="Ruta a la carpeta con la secuencia de imágenes originales (ej: t001.tif)")
    parser.add_argument("results_path", help="Ruta a la carpeta con los resultados del seguimiento (archivos mask*.tif)")
    parser.add_argument("--save", dest="output_dir", help="Directorio opcional para guardar los fotogramas de salida.")
    parser.add_argument("--length", type=int, default=15, help="Longitud de la estela de la trayectoria (fotogramas). Predeterminado: 15")
    
    args = parser.parse_args()
    
    # Usar img_path_obj y results_path_obj si cambiaste los nombres de las variables arriba
    # o simplemente pasar args.img_path y args.results_path si no los cambiaste.
    visualize_trajectories(args.img_path, args.results_path, args.output_dir, args.length)