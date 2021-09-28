#!/usr/bin/env nextflow
glob_wsi = '/linkhome/rech/gendqp01/uub32zv/data/tcga/tcga_breast/*.svs' // insert glob pattern at the end (*.tiff for instance)
root_out = '/gpfsscratch/rech/gdg/uub32zv/working_data/tcga_breast_all_try'
level = 1 
max_nb_tiles = 5
size = 256
tiler = 'simple' // dispo : simple | imagenet | imagenet_v2
dataset = Channel.fromPath(glob_wsi)
				 .map { file -> tuple(file.baseName, file) } 
				 .into { dataset_1; dataset_2}
root_outputs = file("${root_out}/${tiler}/size_${size}/res_${level}/")

process Tiling_folder {
	publishDir "${output_folder}", overwrite: true, pattern: "*.png", mode: 'copy'
	publishDir "$root_outputs/visu/", overwrite: true, pattern: "*visu.png", mode: 'copy'

    clusterOptions "--account=gdg@cpu"
	memory "30GB"
    time { 1.h * task.attempt }
    errorStrategy 'retry'
	maxRetries 3
	maxForks 50

	input:
	set val(slideID), file(slidePath) from dataset_1

	output:
	val slideID into out
	file('*.png')

	script:
	slideID = slidePath.baseName
	output_folder = file("$root_outputs/${slideID}")
	python_script = file("./main_tiling.py")
	"""
	python ${python_script} --path_wsi ${slidePath} \
							--level $level \
							--tiler ${tiler} \
							--size $size \
                            --max_nb_tiles $max_nb_tiles \
                            --nf
	"""
}
