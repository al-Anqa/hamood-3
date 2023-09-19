from whisper_mic.whisper_mic import WhisperMic
from transformers import BloomTokenizerFast, BloomForCausalLM
import torch

def whisper_listen():
    mic = WhisperMic()
    result = mic.listen()
    print(result)
    return result

print(torch.cuda.is_available())

tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom")
model = BloomForCausalLM.from_pretrained("bigscience/bloom-3b")
output_length = 50

while True:
    print('hamood-3 is ready for input!')
    current_input = whisper_listen()
    hamood_tokens = tokenizer(current_input, return_tensors="pt")
    hamood_output = tokenizer.decode(model.generate(hamood_tokens["input_ids"], max_length=output_length)[0])
    print(hamood_output)

    
