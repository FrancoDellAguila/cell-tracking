import matplotlib.pyplot as plt
import numpy as np
from tifffile import imread
from pathlib import Path
import os
import sys
from matplotlib.colors import ListedColormap

def view_tracking(results_path, save_dir=None):
    """
    Visualiza los resultados del seguimiento usando matplotlib con colores consistentes.
    
    Args:
        results_path: Ruta al directorio de resultados del seguimiento (contiene archivos de máscara).
        save_dir: Opcional. Ruta para guardar los fotogramas como imágenes.
    """
    results_path = Path(results_path)
    print(f"Buscando archivos de máscara en: {results_path}")
    
    # Verificar si la ruta existe
    if not results_path.exists():
        print(f"ERROR: La ruta no existe: {results_path}")
        return
        
    # Intentar diferentes extensiones para los archivos de máscara
    mask_files = []
    for ext in ['tif', 'tiff', 'png']:
        files = sorted(list(results_path.glob(f'mask*.{ext}')))
        if files:
            mask_files = files
            print(f"Se encontraron {len(files)} archivos de máscara con extensión .{ext}")
            break # Usar la primera extensión encontrada
    
    if not mask_files:
        print("No se encontraron archivos de máscara. Archivos disponibles en el directorio:")
        for file in results_path.iterdir():
            print(f"  {file.name}")
        return
    
    # Imprimir los primeros archivos de máscara encontrados
    print("Primeros archivos de máscara encontrados:")
    for file in mask_files[:3]:
        print(f"  {file.name}")
        
    # Paso 1: Encontrar el valor máximo de ID en todos los fotogramas
    max_id = 0
    print("Analizando todos los fotogramas para encontrar el ID máximo de célula...")
    for mask_file in mask_files:
        try:
            mask = imread(str(mask_file))
            max_id = max(max_id, np.max(mask))
        except Exception as e:
            print(f"Error al leer {mask_file}: {e}")
            
    print(f"ID máximo de célula encontrado: {max_id}")
    
    # Paso 2: Crear un mapa de colores fijo lo suficientemente grande para todos los IDs posibles
    # Se suma 1 porque el ID 0 es el fondo
    colors = plt.cm.jet(np.linspace(0, 1, int(max_id) + 1))
    
    # Establecer el fondo (ID 0) a negro
    colors[0] = [0, 0, 0, 1]
    
    # Crear mapa de colores personalizado con los colores fijos
    custom_cmap = ListedColormap(colors)
    
    # Crear directorio de guardado si es necesario
    if save_dir:
        save_dir_path = Path(save_dir)
        os.makedirs(save_dir_path, exist_ok=True)
        print(f"Guardando imágenes en: {save_dir_path}")
    
    # Procesar cada fotograma
    for i, mask_file in enumerate(mask_files):
        try:
            mask = imread(str(mask_file))
            # Obtener IDs únicos excluyendo el fondo (0)
            unique_ids = np.sort(np.unique(mask))
            unique_ids_no_bg = unique_ids[unique_ids > 0]
            
            print(f"Procesando fotograma {i+1}, forma: {mask.shape}, IDs de célula: {unique_ids_no_bg[:5]}...")
            
            fig, ax = plt.subplots(figsize=(12, 10)) # Crear figura y ejes
            
            # Mostrar la máscara con colores consistentes
            # vmax asegura que la escala de colores sea consistente
            im = ax.imshow(mask, cmap=custom_cmap, vmax=max_id)
            
            # Crear una leyenda separada en lugar de una barra de color
            # Mostrar solo hasta 20 IDs de célula para evitar saturación
            if len(unique_ids_no_bg) > 0:
                handles = []
                # Elegir un subconjunto de IDs si hay demasiados
                display_ids = unique_ids_no_bg
                if len(display_ids) > 20:
                    # Elegir 20 IDs espaciados uniformemente
                    indices = np.linspace(0, len(display_ids)-1, 20, dtype=int)
                    display_ids = display_ids[indices]
                
                # Crear entradas para la leyenda
                for cell_id in display_ids:
                    # Asegurarse de que cell_id sea un índice válido para colors
                    if cell_id < len(colors):
                        handle = plt.Rectangle((0,0), 1, 1, color=colors[int(cell_id)])
                        handles.append(handle)
                    else:
                         print(f"Advertencia: ID de célula {cell_id} fuera del rango del mapa de colores.")

                # Añadir la leyenda al lado derecho
                if handles: # Solo añadir leyenda si hay elementos válidos
                    ax.legend(handles, [f"Célula {int(id)}" for id in display_ids], 
                              loc='center left', bbox_to_anchor=(1, 0.5),
                              title="IDs de Célula")
            
            ax.set_title(f'Fotograma {i+1}')
            ax.axis('off') # Ocultar ejes
            
            if save_dir:
                output_file = os.path.join(save_dir, f'tracking_frame_{i+1:03d}.png')
                plt.savefig(output_file, bbox_inches='tight') # bbox_inches='tight' ajusta el guardado
                print(f"  Guardado en {output_file}")
                plt.close(fig) # Cerrar la figura para liberar memoria
            else:
                plt.tight_layout() # Ajustar diseño antes de mostrar
                plt.show(block=True) # Mostrar la figura y esperar
                input("Presiona Enter para el siguiente fotograma...") # Pausar hasta que el usuario presione Enter
                plt.close(fig) # Cerrar la figura
        except Exception as e:
            print(f"Error procesando {mask_file}: {e}")
            if 'fig' in locals() and plt.fignum_exists(fig.number): # Asegurarse de que fig existe antes de cerrar
                 plt.close(fig)

    print("Visualización completada.")

# Mantener el bloque principal para ejecución desde línea de comandos
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualizar resultados del seguimiento celular")
    parser.add_argument("results_path", help="Ruta al directorio de resultados del seguimiento")
    parser.add_argument("--save", help="Ruta opcional para guardar los fotogramas visualizados")
    args = parser.parse_args()
    
    view_tracking(args.results_path, args.save)