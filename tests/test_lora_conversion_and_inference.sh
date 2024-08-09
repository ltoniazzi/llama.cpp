#!/bin/bash
set -e

# Step 1: Clone the Hugging Face repository if the directory does not exist
if [ ! -d "reduce-llms-for-testing" ]; then
    echo "Cloning the Hugging Face repository..."
    git clone https://huggingface.co/ltoniazzi/reduce-llms-for-testing
else
    echo "Repository already exists. Skipping clone."
fi


run_llama_cli() {
    local model_name=$1
    local size=$2
    local model_size=$3

    echo "Running convert_hf_to_gguf.py for $model_name with size $size..."
    python convert_hf_to_gguf.py reduce-llms-for-testing/$model_name/size=$size/base --outtype f32

    echo "Running convert_lora_to_gguf.py for $model_name with size $size..."
    python3 convert_lora_to_gguf.py reduce-llms-for-testing/$model_name/size=$size/lora --base reduce-llms-for-testing/$model_name/size=$size/base --outtype f32

    echo "Running llama-cli without lora for $model_name with size $size and model size $model_size..."
    llama-cli -m reduce-llms-for-testing/$model_name/size=$size/base/Base-$model_size-F32.gguf -p "<bos>When forty winters shall besiege" -n 50

    echo "Running llama-cli with lora for $model_name with size $size and model size $model_size..."
    llama-cli -m reduce-llms-for-testing/$model_name/size=$size/base/Base-$model_size-F32.gguf --lora reduce-llms-for-testing/$model_name/size=$size/lora/Lora-F32-LoRA.gguf -p "<bos>I see a " -n 50

    echo "All steps completed for $model_name with size $size and model size $model_size!"
}

# Example usage:
run_llama_cli "Gemma2ForCausalLM" "64" "19M"

