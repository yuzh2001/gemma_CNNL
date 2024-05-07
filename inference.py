import sys 

sys.path.append("./gemma/")
# sys.path.append("/kaggle/working/gemma_pytorch/") 
from gemma.config import get_config_for_2b
from gemma.model import GemmaForCausalLM
#from gemma.tokenizer import Tokenizer
import contextlib
import os
import torch
# add a line
import torch_mlu

# from IPython.display import Markdown as md

# Load the model
VARIANT = "2b"
# MACHINE_TYPE = "cuda" 
MACHINE_TYPE = "mlu"
weights_dir = f'/workspace/dataset/private/data/models/'

@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
  """Sets the default torch dtype to the given dtype."""
  torch.set_default_dtype(dtype)
  yield
  torch.set_default_dtype(torch.float)
  
# Model Config.
model_config = get_config_for_2b()
model_config.tokenizer = os.path.join(weights_dir, "tokenizer.model")
model_config.quant = "quant" in VARIANT
model_config.dtype = "float"

# Model.
device = torch.device(MACHINE_TYPE)
with _set_default_tensor_type(model_config.get_dtype()):
  model = GemmaForCausalLM(model_config)
  ckpt_path = os.path.join(weights_dir, f'gemma-{VARIANT}.ckpt')
  model.load_weights(ckpt_path)
  model = model.to(device).eval()
  
# Use the model

USER_CHAT_TEMPLATE = "<start_of_turn>user\n{prompt}<end_of_turn>\n"
MODEL_CHAT_TEMPLATE = "<start_of_turn>model\n{prompt}<end_of_turn>\n"

USER_PROMPT = "How to join a competition on the Kaggle platform?"

prompt = (
    USER_CHAT_TEMPLATE.format(prompt=USER_PROMPT)
    + MODEL_CHAT_TEMPLATE.format(prompt="Kaggle Platform.")
    + "<start_of_turn>model\n"
)

result = model.generate(
    USER_CHAT_TEMPLATE.format(prompt=prompt),
    device=device,
    output_len=1000,
)

# print the output nicely using markdown
print(f'{result}')

print(f"{model.generate(prompts='what is the name of the largest continent on earth', device=device, output_len=1000)}")

print(f"{model.generate(prompts='what is the difference between gemma 2b and gemma 2b-it model released by google', device=device, output_len=1000)}")
