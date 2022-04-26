#!/bin/bash
#SBATCH --job-name=job-array   # nom du job
#SBATCH --array=0-300%100
#SBATCH --partition=cbio-cpu
#SBATCH --ntasks=1             # Nombre total de processus MPI
#SBATCH --ntasks-per-node=1    # Nombre de processus MPI par noeud
# Dans le vocabulaire Slurm "multithread" fait référence à l"hyperthreading.
#SBATCH --hint=nomultithread   # 1 processus MPI par coeur physique (pas d"hyperthreading)
#SBATCH --time=03:00:00        # Temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=logs/%x_%A_%a.out  # Nom du fichier de sortie contenant l"ID et l"indice
#SBATCH --error=logs/%x_%A_%a.out   # Nom du fichier d"erreur (ici commun avec la sortie)
 
glob_wsi="/path/where/the/raw/WSI/are"
model_path="/path/of/the/model/if/using/ssl/for/encoding.pth" 
level=1
mask_level=-1
size=224
auto_mask=1
tiler="simple" 
normalizer="macenko"
path_outputs="/path/where/to/store/the/output"
python main_tiling_array.py --path_wsi "$glob_wsi" --level "$level" --auto_mask "$auto_mask" --tiler "$tiler" --size "$size" --mask_level "$mask_level" --model_path "$model_path" --path_outputs "$path_outputs" --normalizer "$normalizer"

