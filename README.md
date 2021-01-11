# On-grammar-improvements-of-GPT-2-generation

This repository includes the complete code for the paper XXXX. The purpose of this repository is to make experiments reproducable and give advanced insights into the experiments that were conducted. 	

<img src="architecture.png" alt="drawing" width="200"/>

## Getting Started

This repository includes the following things: 

  - Documentation of the Dataset building process
  - Finetuning, Grammar Correction and Generation scripts that were used during this research project
  - Documentation of the complete evaluation process
  - A mountain of generated samples that was used during evaluation
  - Documentation of the model combination evaluation
  - Documentation of generating samples referenced in our paper


## Build on

* [HuggingFace -Transformers](https://github.com/huggingface/transformers)

### Installation

To install dependencies simply run

```
pip install -r requirements.txt
```
You should be good to go. 


  
  
## The following external resources should be added in order to retrace all steps: 

- LAMBADA data files should be extracted to the LAMBADA folder (downloadable from https://wiki.cimec.unitn.it/tiki-index.php?page=CLIC)
- The GPT-2 generation datasets should be extracted to the original_data folder (downloadable from https://github.com/openai/gpt-2-output-dataset). For the purpose of this paper, only the small-117M datasets are needed.
- The model checkpoints are needed to retrace everything. If needed they are available by contacting Gideon-Stein. 


## Generation script usage:
 ```
 python transgenerator_translation.py --model_path=../trained_models/the_dream_final_3/checkpoint-257616/pytorch_model.bin --text_path ../build_data/EOS_new_filter_700.txt --n_data 1000 --save_path the_dream_filter_700_3_1.p
 python run_generation_edited.py  --model_name_or_path=model_save/only_correctedAll/pytorch_model.bin --save_name oootf 
```
  Parameters can be added and changed accordingly to the script.

## Finetuning script usage:
 ```
 python run_lm_finetuning_frozen_3.py --output_dir=model_save/the_dream_finetune_3 --model_type=gpt2 --model_name_or_path=gpt2 --do_train --train_data_file=classic_finetune_train.txt  --per_gpu_train_batch_size 1  --gradient_accumulation_steps 4 --save_steps 41599 --save_total_limit 20  --num_train_epochs 20 
 python run_generation_edited.py  --model_name_or_path=model_save/the_dream_classic_finetune_2/first/checkpoint-41599/pytorch_model.bin --save_name generate_cf_1  --max_length 1024 --n_sentences 100
 ```


 

 ## Authors

* **Gideon Stein** - *Initial work* - [Github](https://github.com/Gideon-Stein)



