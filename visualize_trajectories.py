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
    img_path = Path(img_path)
    results_path = Path(results_path)
    print(f"Buscando imágenes originales en: {img_path}")
    print(f"Buscando archivos de máscara en: {results_path}")
    
    # Verificar si las rutas existen
    if not img_path.exists() or not results_path.exists():
        print(f"ERROR: La ruta de imágenes o de resultados no existe.")
        return
        
    # Encontrar imágenes originales
    img_files = []
    for ext in ['tif', 'tiff', 'png', 'jpg']:
        # Buscar primero el patrón t***.ext
        files = sorted(list(img_path.glob(f't*.{ext}')))
        if files:
            img_files = files
            print(f"Se encontraron {len(files)} archivos de imagen con extensión .{ext} (patrón t***)")
            break
            
    if not img_files:
        # Si no se encuentra t***, buscar cualquier nombre *.ext
        for ext in ['tif', 'tiff', 'png', 'jpg']:
            files = sorted(list(img_path.glob(f'*.{ext}')))
            if files:
                img_files = files
                print(f"Se encontraron {len(files)} archivos de imagen con extensión .{ext} (cualquier nombre)")
                break
    
    # Encontrar archivos de máscara
    mask_files = []
    for ext in ['tif', 'tiff', 'png']:
        files = sorted(list(results_path.glob(f'mask*.{ext}')))
        if files:
            mask_files = files
            print(f"Se encontraron {len(files)} archivos de máscara con extensión .{ext}")
            break
    
    if not img_files or not mask_files:
        print("No se pudieron encontrar los archivos de imagen o de máscara requeridos.")
        return
        
    # Asegurarse de tener el mismo número de archivos o usar el mínimo
    min_frames = min(len(img_files), len(mask_files))
    if min_frames == 0:
        print("ERROR: No se encontraron fotogramas coincidentes entre imágenes y máscaras.")
        return
        
    img_files = img_files[:min_frames]
    mask_files = mask_files[:min_frames]
        
    print(f"Procesando {min_frames} fotogramas coincidentes.")
    
    # Paso 1: Encontrar el valor máximo de ID en todos los fotogramas
    max_id = 0
    print("Analizando todos los fotogramas para encontrar el ID máximo de célula...")
    for mask_file in mask_files:
        try:
            mask = imread(str(mask_file))
            max_id = max(max_id, np.max(mask))
        except Exception as e:
            print(f"Advertencia: Error al leer {mask_file}: {e}")
            
    print(f"ID máximo de célula encontrado: {max_id}")
    
    # Paso 2: Crear un color para cada ID de célula
    # Se suma 1 porque el ID 0 es el fondo
    colors = plt.cm.jet(np.linspace(0, 1, int(max_id) + 1))
    
    # Paso 3: Rastrear centroides para todas las células a través de los fotogramas
    print("Calculando centroides de células en todos los fotogramas...")
    # Almacenar como {id_celula: [(fotograma, x, y), ...]}
    centroids = {cell_id: [] for cell_id in range(1, int(max_id) + 1)}
    
    for frame_idx, mask_file in enumerate(mask_files):
        try:
            mask = imread(str(mask_file))
            for cell_id in np.unique(mask):
                if cell_id == 0:  # Omitir fondo
                    continue
                    
                # Crear máscara binaria para esta célula
                cell_mask = (mask == cell_id)
                if np.any(cell_mask):
                    # Calcular centro de masa (formato y, x)
                    y, x = center_of_mass(cell_mask)
                    # Guardar como (fotograma, x, y)
                    centroids[cell_id].append((frame_idx, int(x), int(y)))
        except Exception as e:
            print(f"Advertencia: No se pudieron calcular centroides para {mask_file.name}: {e}")

    # Crear directorio de guardado si es necesario
    if save_dir:
        save_dir_path = Path(save_dir)
        os.makedirs(save_dir_path, exist_ok=True)
        print(f"Guardando imágenes de trayectoria en: {save_dir_path}")
    
    # Procesar cada fotograma
    print("Generando visualizaciones de trayectoria...")
    for frame_idx, (img_file, mask_file) in enumerate(zip(img_files, mask_files)):
        try:
            # Cargar imagen original y máscara
            img = imread(str(img_file))
            mask = imread(str(mask_file)) # Se usa para saber qué IDs están presentes
            
            # Normalizar imagen para visualización si no es de 8 bits
            if img.dtype != np.uint8:
                # Usar percentiles para robustez contra outliers
                p_low, p_high = np.percentile(img, (1, 99))
                img_norm = (img - p_low) / (p_high - p_low + 1e-6) # Evitar división por cero
                img_display = np.clip(img_norm * 255, 0, 255).astype(np.uint8)
            else:
                img_display = img
            
            # Si la imagen es escala de grises, convertir a RGB
            if len(img_display.shape) == 2:
                img_rgb = np.stack([img_display, img_display, img_display], axis=2)
            elif img_display.shape[2] == 1: # Manejar imágenes monocanal
                 img_rgb = np.concatenate([img_display] * 3, axis=-1)
            else: # Asumir que ya es RGB o similar
                img_rgb = img_display[:,:,:3] # Tomar solo los primeros 3 canales si hay más

            # Crear figura
            fig, ax = plt.subplots(figsize=(11, 9)) # Tamaño ajustado
            ax.imshow(img_rgb)
            
            # Obtener IDs únicos de células en el fotograma actual
            unique_ids = np.unique(mask)
            unique_ids = unique_ids[unique_ids > 0]  # Excluir fondo
            
            # Dibujar trayectorias
            for cell_id in unique_ids:
                if cell_id not in centroids: continue # Chequeo de seguridad

                # Obtener todos los centroides para esta célula
                cell_centroids = centroids[cell_id]
                
                # Filtrar centroides hasta el fotograma actual
                relevant_centroids = [c for c in cell_centroids if c[0] <= frame_idx]
                
                # Mantener solo los últimos 'trajectory_length'
                trajectory_points = relevant_centroids[-trajectory_length:]
                
                if len(trajectory_points) > 1:
                    # Extraer coordenadas x e y
                    frames = [c[0] for c in trajectory_points]
                    x_coords = [c[1] for c in trajectory_points]
                    y_coords = [c[2] for c in trajectory_points]
                    
                    # Dibujar línea con grosor/alfa creciente para posiciones más recientes
                    num_segments = len(trajectory_points) - 1
                    for i in range(num_segments):
                        # Líneas más gruesas/opacas para posiciones más recientes
                        # El alfa aumenta hacia el punto actual
                        alpha = 0.2 + 0.8 * ((i + 1) / num_segments)
                        ax.plot(x_coords[i:i+2], y_coords[i:i+2], # Dibujar segmento
                                 color=colors[int(cell_id)], linewidth=1.8, # Línea ligeramente más fina
                                 alpha=alpha,
                                 solid_capstyle='round') # Extremos de línea más suaves
                
                # Marcar posición actual con círculo - más pequeño y translúcido
                if trajectory_points:
                    current_x, current_y = trajectory_points[-1][1], trajectory_points[-1][2]
                    ax.plot(current_x, current_y, 'o', 
                             color=colors[int(cell_id)], 
                             markersize=6,  # Marcador más pequeño
                             markeredgecolor='black', # Borde negro
                             markeredgewidth=0.6,
                             alpha=0.8)  # Marcador ligeramente más opaco
                    
                    # Se eliminó el código de la etiqueta de ID para mantener la visualización más limpia
            
            ax.set_title(f'Fotograma {frame_idx+1} / {min_frames}')
            ax.axis('off')  # Ocultar ejes
            
            if save_dir:
                output_file = save_dir_path / f'trayectoria_{frame_idx+1:04d}.png' # Usar 4 dígitos
                plt.savefig(output_file, bbox_inches='tight', dpi=120) # DPI más bajo para guardado rápido
                # print(f"  Guardado en {output_file.name}") # Mensaje de guardado menos verboso
                plt.close(fig) # Cerrar figura para liberar memoria
            else:
                plt.tight_layout() # Ajustar diseño
                plt.show(block=False) # Mostrar sin bloquear
                plt.pause(0.1) # Pausa corta para permitir actualización
                user_input = input("Presiona Enter para el siguiente fotograma (o 'q' para salir): ")
                plt.close(fig) # Cerrar figura
                if user_input.lower() == 'q':
                    print("Saliendo de la visualización.")
                    break # Salir del bucle de fotogramas
        except Exception as e:
            print(f"ERROR procesando fotograma {frame_idx}: {e}")
            if 'fig' in locals() and plt.fignum_exists(fig.number): # Asegurarse de que fig existe
                 plt.close(fig) # Cerrar figura en caso de error

    print("Visualización de trayectorias finalizada.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualiza trayectorias de seguimiento celular sobre las imágenes originales.")
    parser.add_argument("img_path", help="Ruta a la carpeta con la secuencia de imágenes originales (ej: t001.tif)")
    parser.add_argument("results_path", help="Ruta a la carpeta con los resultados del seguimiento (archivos mask*.tif)")
    parser.add_argument("--save", dest="output_dir", help="Directorio opcional para guardar los fotogramas de salida.")
    parser.add_argument("--length", type=int, default=15, help="Longitud de la estela de la trayectoria (fotogramas). Predeterminado: 15")
    
    args = parser.parse_args()
    
    visualize_trajectories(args.img_path, args.results_path, args.output_dir, args.length)