#!/bin/bash
apt-get update && apt-get upgrade
sudo apt-get install build-essential
pip install -r requirements.txt
python -m spacy download en_core_web_sm

echo -e "   *\n  ***\n *****\n*******"
echo 'if you want to use other LLM please follow instructions on https://www.llama.com/llama-downloads/'
echo -e "   *\n  ***\n *****\n*******"
echo 'or use this link https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_K_M.gguf'
echo -e "   *\n  ***\n *****\n*******"
echo 'remember to put the model into ./models!!!!'
