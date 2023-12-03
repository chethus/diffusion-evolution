#!/usr/bin/env bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="/home/chet/diffusers/examples/pokemon_blip_evolution"

accelerate launch --mixed_precision="fp16" train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME --caption_column="evolved_text" --image_column="evolution_image" \
  --resolution=512 --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=100 --checkpointing_steps=5000 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="sd-pokemon-model-lora" \
  --validation_prompt="majestic, colorful scales, larger size, longer tail, fierce expression, sharper teeth, wings" \
  --report_to="wandb" \
  --prefix lora \
  --run_name blip_evolution_test