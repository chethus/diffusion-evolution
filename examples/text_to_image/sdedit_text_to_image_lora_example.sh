#!/usr/bin/env bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATASET_NAME="/iliad/u/chethanb/diffusion-evolution/examples/pokemon_blip_evolution"

accelerate launch text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME --caption_column="evolved_text" --image_column="evolution_image" \
  --enable_xformers_memory_efficient_attention \
  --train_batch_size=4 --gradient_accumulation_steps=4 --gradient_checkpointing \
  --use_8bit_adam \
  --max_train_steps=15000 \
  --resolution=512 --random_flip \
  --num_train_epochs=100 --checkpointing_steps=5000 \
  --learning_rate=5e-05 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="sd-edit-pokemon-model-lora-lr-5e-5" \
  --val_image_url="https://datasets-server.huggingface.co/cached-assets/lambdalabs/pokemon-blip-captions/--/8b762e1dac1b31d60e01ee8f08a9d8a232b59e17/--/default/train/65/image/image.jpg" \
  --validation_prompt="a drawing of a larger pink cat with a longer tail, blue eyes, larger ears, sharper claws, more defined muscles, a shiny coat of fur, a confident expression, and a graceful posture" \
  --report_to="wandb" \
  --prefix sdedit \
  --run_name blip_evolution_lr_5e-5
