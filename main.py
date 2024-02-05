from whisper_mic.whisper_mic import WhisperMic
from transformers import BloomTokenizerFast, BloomForCausalLM
import torch

def whisper_listen():
    result = mic.listen_loop()
    print(result)
    return result

print(f'CUDA Detected: {torch.cuda.is_available()}')
mic = WhisperMic()

tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom")
model = BloomForCausalLM.from_pretrained("bigscience/bloom-3b")
output_length = 50

while True:
    print('hamood-3 is ready for input!')
    current_input = whisper_listen()
    hamood_tokens = tokenizer(current_input, return_tensors="pt")
    hamood_output = tokenizer.decode(model.generate(hamood_tokens["input_ids"], max_length=output_length)[0])
    print(hamood_output)

    #TODO Make this bad boy faster
