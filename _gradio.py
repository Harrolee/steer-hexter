# setup
# general imports
import os
import torch
from tqdm import tqdm
import plotly.express as px

from torch import Tensor
from transformer_lens import utils
from functools import partial
from jaxtyping import Int, Float
from transformer_lens import HookedTransformer
from sae_lens import SAE


if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL = "gemma-2b"
PRETRAINED_SAE = "gemma-2b-res-jb"
LAYER = 6
# get model
model = HookedTransformer.from_pretrained(MODEL, device = device)

# get the SAE for this layer
sae, cfg_dict, _ = SAE.from_pretrained(
    release = PRETRAINED_SAE,
    sae_id = f"blocks.{LAYER}.hook_resid_post",
    device = device
)

hook_point = sae.cfg.hook_name


class Inference:

    def __init__(self, model, pretrained_sae, layer):
        self.setModel(model)
        self.setSAE(pretrained_sae)
        self.layer = layer
        self.sae_id = f"blocks.{LAYER}.hook_resid_post"
        self.sampling_kwargs = dict(temperature=1.0, top_p=0.1, freq_penalty=1.0)
 
    
    def set_model(self, model):
        self.model = HookedTransformer.from_pretrained(model, device = device)

    def set_coeff(self, coeff):
        self.coeff = coeff
    
    def set_temperature(self, temperature):
        self.sampling_kwargs['temperature'] = temperature

    def set_steering_vector_prompt(self, prompt: str):
        self.steering_vector_prompt = prompt

    def setSAE(self, sae_name):
        sae, cfg_dict, _ = SAE.from_pretrained(
            release = sae_name,
            sae_id = self.sae_id,
            device = device
        )
        self.sae = sae
        self.cfg_dict = cfg_dict

    def _getFeatureActivations(self, steering_vector_prompt):
        sv_logits, activationCache = model.run_with_cache(steering_vector_prompt, prepend_bos=True)
        sv_feature_acts = sae.encode(activationCache[hook_point])
        # self.sae_out = sae.decode(sv_feature_acts) 
        return self.sae.decode(sv_feature_acts) 


    def _hooked_generate(self, feature, steering_on, prompt_batch, seed=None, **kwargs):
        coeff = self.coeff
        sae_out = self._getFeatureActivations()

        steering_vector = sae.W_dec[feature]

        def steering_hook(self, resid_pre, hook):
            if resid_pre.shape[1] == 1:
                return

            position = sae_out.shape[1]
            if steering_on:
            # using our steering vector and applying the coefficient
                resid_pre[:, :position - 1, :] += coeff * steering_vector

        if seed is not None:
            torch.manual_seed(seed)


        fwd_hooks = [(self.sae_id, steering_hook)]


        with self.model.hooks(fwd_hooks=fwd_hooks):
            tokenized = self.model.to_tokens(prompt_batch)
            result = self.model.generate(
                stop_at_eos=False,  # avoids a bug on MPS
                input=tokenized,
                max_new_tokens=50,
                do_sample=True,
                **kwargs)
        return result

    def _get_feature(self, sv_feature_activations):
        # return torch.topk(sv_feature_acts, 1).indices.tolist()
        print(f'is this a single index? {torch.topk(sv_feature_acts, 1).indices}')
        return torch.topk(sv_feature_acts, 1).indices

    def _run_generate(self, example_prompt, steering_on: bool):
        self.model.reset_hooks()
        feature = self._get_feature()
        res = self._hooked_generate([example_prompt] * 3, feature, steering_on, seed=None, **self.sampling_kwargs)
        # Print results, removing the ugly beginning of sequence token
        res_str = self.model.to_string(res[:, 1:])
        print(("\n\n" + "-" * 80 + "\n\n").join(res_str))

    # def generate(self, coeff, temp, steering_prompt, steering: bool, )
    #     _run_generate()



def findFeature():
    ...

def applyFeature(feature: int, sae: SAE):
    steering_vector = sae.W_dec[feature]

    ...

def generate():
    ...

# 
sv_prompt = " Lily"
sv_logits, activationCache = model.run_with_cache(sv_prompt, prepend_bos=True)
sv_feature_acts = sae.encode(activationCache[hook_point])
print(torch.topk(sv_feature_acts, 3).indices.tolist())




# def _construct_fwd_hooks(features: []):
#     def steering_hook(self, resid_pre, hook):
#         if resid_pre.shape[1] == 1:
#             return

#         position = sae_out.shape[1]
#         if steering_on:
#         # using our steering vector and applying the coefficient
#             resid_pre[:, :position - 1, :] += coeff * steering_vector

#     steering_vectors = [sae.W_dec[f] for f in features]
    
#     fwd_hooks = [(self.sae_id, steering_hook)]

#     return fwd_hooks