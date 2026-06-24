#!/usr/bin/env bash
set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-12345}

INPUT_JSON=${INPUT_JSON:-test_case/double_human.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-output/domainshuttle_output_batch_double_human_shift_5}
DOMAIN_MODEL_NAME=${DOMAIN_MODEL_NAME:-models/Diffusion_Transformer/Wan2.2-DomainShuttle-A14B}

if [[ ! -f "${DOMAIN_MODEL_NAME}/high_noise_model/diffusion_pytorch_model.safetensors" ]] || \
   [[ ! -f "${DOMAIN_MODEL_NAME}/low_noise_model/diffusion_pytorch_model.safetensors" ]]; then
  python scripts/wan2.2_domainshuttle/convert_mindspeed_domainshuttle.py \
    --output "${DOMAIN_MODEL_NAME}"
fi


#Unofficial inference code, the official code is currently under institutional review.
#shift 5 (480P) or 12 (720P)
torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" \
  examples/wan2.2_domainshuttle/predict_r2v_batch.py \
  --input_json "${INPUT_JSON}" \
  --output_dir "${OUTPUT_DIR}" \
  --domain_model_name "${DOMAIN_MODEL_NAME}" \
  --height 480 \
  --width 832 \
  --video_length 81 \
  --fps 24 \
  --num_inference_steps 40 \
  --guidance_scale 4.0 3.0 \
  --shift 5 \
  --ulysses_degree "${NPROC_PER_NODE}" \
  --ring_degree 1 \
  --seed 42
