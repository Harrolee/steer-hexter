# general imports
import os
import torch
from tqdm import tqdm
import plotly.express as px

torch.set_grad_enabled(False);


# package import
from torch import Tensor
from transformer_lens import utils
from functools import partial
from jaxtyping import Int, Float

# device setup
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")





from transformer_lens import HookedTransformer
from sae_lens import SAE

# Choose a layer you want to focus on
# For this tutorial, we're going to use layer ????
layer = 0

# get model
model = HookedTransformer.from_pretrained("tiny-stories-1L-21M", device = device)

# get the SAE for this layer
sae = SAE.load_from_pretrained("sae_tiny-stories-1L-21M_blocks.0.hook_mlp_out_16384", device = device)

# get hook point
hook_point = sae.cfg.hook_name
print(hook_point)



sv_prompt = " Lily"
sv_logits, activationCache = model.run_with_cache(sv_prompt, prepend_bos=True)
sv_feature_acts = sae.encode(activationCache[hook_point])
print(torch.topk(sv_feature_acts, 3).indices.tolist())

# Generate


sv_prompt = " Lily"
sv_logits, activationCache = model.run_with_cache(sv_prompt, prepend_bos=True)
tokens = model.to_tokens(sv_prompt)
print(tokens)

# get the feature activations from our SAE
sv_feature_acts = sae.encode(activationCache[hook_point])

# get sae_out
sae_out = sae.decode(sv_feature_acts)

# print out the top activations, focus on the indices
print(torch.topk(sv_feature_acts, 3))



# get the neurons to use;
print(torch.topk(sv_feature_acts, 3).indices.tolist())


# choose the vector -- find this from the above section
# 
steering_vector = sae.W_dec[10284]

example_prompt = "Once upon a time"
coeff = 1000
sampling_kwargs = dict(temperature=1.0, top_p=0.1, freq_penalty=1.0)



# apply steering vector when the model generates

def steering_hook(resid_pre, hook):
    if resid_pre.shape[1] == 1:
        return

    position = sae_out.shape[1]
    if steering_on:
      # using our steering vector and applying the coefficient
      resid_pre[:, :position - 1, :] += coeff * steering_vector


def hooked_generate(prompt_batch, fwd_hooks=[], seed=None, **kwargs):
    if seed is not None:
        torch.manual_seed(seed)

    with model.hooks(fwd_hooks=fwd_hooks):
        tokenized = model.to_tokens(prompt_batch)
        result = model.generate(
            stop_at_eos=False,  # avoids a bug on MPS
            input=tokenized,
            max_new_tokens=50,
            do_sample=True,
            **kwargs)
    return result


def run_generate(example_prompt):
  model.reset_hooks()
  editing_hooks = [(f"blocks.{layer}.hook_resid_post", steering_hook)]
  res = hooked_generate([example_prompt] * 3, editing_hooks, seed=None, **sampling_kwargs)

  # Print results, removing the ugly beginning of sequence token
  res_str = model.to_string(res[:, 1:])
  print(("\n\n" + "-" * 80 + "\n\n").join(res_str))


steering_on = False
run_generate(example_prompt)



# evaluate features

import pandas as pd

# Let's start by getting the top 10 logits for each feature
projection_onto_unembed = sae.W_dec @ model.W_U


# get the top 10 logits.
vals, inds = torch.topk(projection_onto_unembed, 10, dim=1)

# get 10 random features
random_indices = torch.randint(0, projection_onto_unembed.shape[0], (10,))

# Show the top 10 logits promoted by those features
top_10_logits_df = pd.DataFrame(
    [model.to_str_tokens(i) for i in inds[random_indices]],
    index=random_indices.tolist(),
).T
top_10_logits_df
# [7195, 5910, 2041]
top_10_associated_words_logits_df = model.to_str_tokens(inds[5910])
# See the words associated with feature 7195 (Should be "Golden")
