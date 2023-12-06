export WANDB_MODE=online
#!/usr/bin/env bash
accelerate launch --mixed_precision="fp16" instruct_pix2pix/train_instruct_pix2pix.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --train_data_dir="/iliad/u/chethanb/diffusion-evolution/examples/pokemon_blip_evolution" \
    --enable_xformers_memory_efficient_attention \
    --use_8bit_adam \
    --resolution=256 --random_flip \
    --train_batch_size=4 --gradient_accumulation_steps=4 --gradient_checkpointing \
    --max_train_steps=15000 \
    --checkpointing_steps=5000 --checkpoints_total_limit=1 \
    --learning_rate=5e-05 --max_grad_norm=1 --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --mixed_precision=fp16 \
    --original_image_column image \
    --edited_image_column evolution_image \
    --edit_prompt_column evolved_text \
    --val_image_url="https://datasets-server.huggingface.co/cached-assets/lambdalabs/pokemon-blip-captions/--/8b762e1dac1b31d60e01ee8f08a9d8a232b59e17/--/default/train/65/image/image.jpg" \
    --validation_prompt="a drawing of a larger pink cat with a longer tail, blue eyes, larger ears, sharper claws, more defined muscles, a shiny coat of fur, a confident expression, and a graceful posture" \
    --num_validation_images 4 \
    --seed=42 \
    --report_to=wandb \
    --prefix instruct_pix2pix \
    --run_name blip_evolution_no_guidance \
    --push_to_hub
