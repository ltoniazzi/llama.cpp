# Main plan

1. Design inference pattern to pick up LoRA adapters for matmul (factor it for phi3 only). 
    - How to check performance in these cases?
2. Ask advice on compilation of adapters in the model, basically where to gguf write them, and should it be done with the base model from the safetensors format or should we do it from two compiled versions, should ask somebody about this design.




## TODOs

1. ~~How to debug mat_mul (run tests in cpp?)~~
2. How to wrap the suggestion from slaren on matmul ~~(need to see how to find the llora info to pick up)~~. Something about lora being loaded in the context? How to pick a specifi LoRA
3. check the PR "It was removed in [#7204](https://github.com/ggerganov/llama.cpp/pull/7204). `convert-lora-to-ggml.py` seems to write  loras to gguf witouth the model? Should check the train script and see how they match lora with base layers
4. https://github.com/ggerganov/llama.cpp/discussions/3489
5. check lora example in examples `examples/export-lora/export-lora.cpp`, ask gpt if can be used to extend applying multiple Loras, then ask back to slaren
6. try debug cpp at runtime to understand the context variable and where the adapter lives there


1. Add lora to context and then to mat_mul for ollama
2. Add anothe lora and swap
3. Understand runtime swapping logic (like the server is up, run the init only when passing different lora param)
4. If design from finetuning no need to design from peft.


## Comments/Tips

- Compile with (check faster compilation with debug mode)
    ```
    make clean && make -j 8 LLAMA_DEBUG=1
    ```

- The debug compiled main is much slower at inference, to rebuild
:
    ```bash
    make clean && make -j 8
    ```

- Disable `ccache` with `LLAMA_NO_CCACHE`

- Run with
    ```bash
    ./main -m ./models/phi-3-mini/ggml-model-Q4_K_M.gguf -n 128
    ```


On the compilation side of this probelm:
1. Should I write the adapter weights in the same gguf file with the base weight? 
2. If yes to (1), are there other scripts in the repo to get inspiration from for how to compile base weights along with lora weights in the same gguf file?
3. If no to (1), what is a good strategy to compile the lora weights?


## Debug

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python Debugger: Current File",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": true
        },
        {
            "name": "(gdb) Launch",
            "type": "cppdbg",
            "request": "launch",
            "program": "${workspaceFolder}/main",
            "args": ["-m", "./models/phi-3-mini/ggml-model-Q4_K_M.gguf", "-n", "128"],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "setupCommands": [
                {
                    "description": "Enable pretty-printing for gdb",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                }
            ]
        }
    ]
}
```


download ollama
quantize
finetune llora

## Open-LLama

load loading model? check export lora

1. [Open-llama download safe tensors](https://huggingface.co/openlm-research/open_llama_3b_v2/tree/main)

2. Data `data/shakespeare_short.txt` is just two words.

3. Run:
    ```bash
    python3 convert-hf-to-gguf.py --model models/open-llama/

    ./quantize ./models/open-llama/ggml-model-f16.gguf ./models/open-llama/ggml-model-q8_0.gguf Q8_0
    
    ./finetune         --model-base models/open-llama/open-llama-3b-v2-q8_0.gguf         --checkpoint-in  models/open-llama/chk-lora-open-llama-3b-v2-q8_0-shakespeare-LATEST.gguf         --checkpoint-out models/open-llama/chk-lora-open-llama-3b-v2-q8_0-shakespeare-ITERATION.gguf         --lora-out models/open-llama/lora-open-llama-3b-v2-q8_0-shakespeare-ITERATION.bin         --train-data "data/shakespeare_short.txt"         --save-every 1         --threads 1 --adam-iter 1 --batch 1 --ctx 16         --use-checkpointing
    
    ./export-lora     -m models/open-llama/open-llama-3b-v2-q8_0.gguf     -o models/open-llama/open-llama-3b-v2-q8_0-english2tokipona-chat.gguf     -l models/open-llama/lora-open-llama-3b-v2-q8_0-shakespeare-LATEST.bin

    ```