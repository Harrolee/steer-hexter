from transformer_lens import HookedTransformer
from sae_lens import SAE
import torch

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

class Inference:

    def __init__(self, model, pretrained_sae, layer):
        self.layer = layer
        if model == "gemma-2b":
            self.sae_id = f"blocks.{layer}.hook_resid_post"
        elif model == "gpt2-small":
            print(f"using {model}")
            self.sae_id = f"blocks.{0}.hook_resid_pre"
        self.sampling_kwargs = dict(temperature=1.0, top_p=0.1, freq_penalty=1.0)
        self.set_coeff(1)
        self.set_model(model)
        self.set_SAE(pretrained_sae)
 
    
    def set_model(self, model):
        self.model = HookedTransformer.from_pretrained(model, device = device)

    def set_coeff(self, coeff):
        self.coeff = coeff
    
    def set_temperature(self, temperature):
        self.sampling_kwargs['temperature'] = temperature

    def set_steering_vector_prompt(self, prompt: str):
        self.steering_vector_prompt = prompt

    def set_SAE(self, sae_name):
        sae, cfg_dict, _ = SAE.from_pretrained(
            release = sae_name,
            sae_id = self.sae_id,
            device = device
        )
        self.sae = sae
        self.cfg_dict = cfg_dict

    def _get_sae_out_and_feature_activations(self):
        sv_logits, activationCache = self.model.run_with_cache(self.steering_vector_prompt, prepend_bos=True)
        sv_feature_acts = self.sae.encode(activationCache[self.sae.cfg.hook_name])
        # get top_k of 1
        # self.sae_out = sae.decode(sv_feature_acts) 
        return self.sae.decode(sv_feature_acts), sv_feature_acts

    def _hooked_generate(self, prompt_batch, seed=None, **kwargs):
        coeff = self.coeff
        sae_out, sv_feature_acts = self._get_sae_out_and_feature_activations()
        feature = self._get_feature(sv_feature_acts)
        steering_vector = self.sae.W_dec[feature]

        def steering_hook(self, resid_pre, hook):
            if resid_pre.shape[1] == 1:
                return

            position = sae_out.shape[1]
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
        print(f'is this a single index? {torch.topk(sv_feature_activations, 1).indices}')
        return torch.topk(sv_feature_activations, 1).indices

    def _run_generate(self, example_prompt, steering_on: bool):
        self.model.reset_hooks()

        if steering_on: 
            res = self._hooked_generate([example_prompt] * 3, seed=None, **self.sampling_kwargs)
        else:
            tokenized = self.model.to_tokens([example_prompt])
            res = self.model.generate(
                stop_at_eos=False,  # avoids a bug on MPS
                input=tokenized,
                max_new_tokens=50,
                do_sample=True,
                **self.sampling_kwargs)
            
        # Print results, removing the ugly beginning of sequence token
        res_str = self.model.to_string(res[:, 1:])
        response = ("\n\n" + "-" * 80 + "\n\n").join(res_str)
        print(response)
        return response


    def generate(self, message: str, steering_on: bool):
        return self._run_generate(message, steering_on)



MODEL = "gemma-2b"
PRETRAINED_SAE = "gemma-2b-res-jb"
MODEL = "gpt2-small"
PRETRAINED_SAE = "gpt2-small-res-jb"
LAYER = 10
chatbot_model = Inference(MODEL,PRETRAINED_SAE, LAYER)


import time
import gradio as gr

default_image = "Hexter-Hackathon.png"

def slow_echo(message, history):
    result = chatbot_model.generate(message, False)
    breakpoint()
    for i in range(len(result)):
        time.sleep(0.08)
        yield result[: i + 1]
def slow_echo_steering(message, history):
    result = chatbot_model.generate(message, True)
    breakpoint()
    for i in range(len(result)):
        time.sleep(0.08)
        yield result[: i + 1]

def onSubmit(temp, coeff, steer):
    print('steer ' + str(steer))
    print('temp ' + str(temp))
    print('coeff ' + str(coeff))

with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("*STANDARD HEXTER BOT*")
    with gr.Row():
        chatbot = gr.ChatInterface(
            slow_echo,
            chatbot=gr.Chatbot(min_width=1000),
            textbox=gr.Textbox(placeholder="Ask Hexter anything!", min_width=1000),
            theme="soft",
            cache_examples=False,
            retry_btn=None,
            clear_btn=None,
            undo_btn=None,
        )
    with gr.Row():
        gr.Markdown("*STEERED HEXTER BOT*")
    with gr.Row():
        chatbot_steered = gr.ChatInterface(
            slow_echo_steering,
            chatbot=gr.Chatbot(min_width=1000),
            textbox=gr.Textbox(placeholder="Ask Hexter anything!", min_width=1000),
            theme="soft",
            cache_examples=False,
            retry_btn=None,
            clear_btn=None,
            undo_btn=None,
        )
    with gr.Row():
        steering_prompt = gr.Textbox(label="Steering prompt", value="Golden Gate Bridge")
    with gr.Row():
        coeff = gr.Slider(1, 30, 25, label="Coefficient", info="Coefficient is..", interactive=True)
    with gr.Row():
        temp = gr.Slider(0, 90, 45, label="Temperature", info="Temperature is..", interactive=True)
    with gr.Row():
        steer = gr.Checkbox(label="Steer")

    # Set up an action when the sliders change
    temp.change(chatbot_model.set_temperature, inputs=[temp], outputs=[])
    coeff.change(chatbot_model.set_coeff, inputs=[coeff], outputs=[])
    chatbot_model.set_steering_vector_prompt(steering_prompt)
    steering_prompt.change(chatbot_model.set_steering_vector_prompt, inputs=[steering_prompt], outputs=[])

demo.queue()
demo.launch(debug=True)

if __name__ == "__main__":
    demo.launch(allowed_paths=["/"])
