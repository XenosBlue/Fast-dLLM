
Folder highlights
Code and logs detail evaluation runs for the Fast-dLLM v2 model across various tasks like GSM8K and IFEval, referencing layer skipping strategies.

from typing import Callable, Optional, Union
import torch
import torch.nn.functional as F
import types
import weakref
from transformers.utils import auto_docstring, logging

FAST_DLLM_MASK_ID = 151665
FAST_DLLM_STOP_TOKEN = 151645
MASK_COLOR = 0.5  
TOKEN_COLOR = -0.5  

@auto_docstring
class Fast_dLLM_QwenForCausalLM:

    @torch.no_grad()
    def batch_sample(self, input_ids, tokenizer, block_size, max_new_tokens, small_block_size, min_len, seq_len, mask_id=151665, threshold=0.95, stop_token=151645, use_block_cache=False, top_p=0.95, temperature=0.0):
        self._skip_return_tuple = True
        self._skip_layer_enabled = False
        
        if hasattr(self, 'layer_skip_state'):
            self.layer_skip_state['prev_input'] = None
            self.layer_skip_state['total_flops'] = 0.0
        else:
            self.layer_skip_state = {'prev_input': None, 'total_flops': 0.0}

        
        num_blocks = max_new_tokens // block_size + seq_len.max().item() // block_size
        batch_size = input_ids.shape[0]


        if min_len > block_size:
            self._skip_return_tuple = True
            self._skip_layer_enabled = False 
            output = self.forward(input_ids=input_ids[:, :(min_len // block_size * block_size)], use_cache=True, update_past_key_values=True, block_size=block_size)
            logits, past_key_values = output.logits, output.past_key_values
            if min_len % block_size == 0:
                predict_sample_idx = (seq_len == min_len)
                predict_logits = logits[predict_sample_idx, -1:, :]
                next_token = predict_logits.argmax(dim=-1)
                if input_ids.shape[1] <= min_len:
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                else:
                    input_ids[predict_sample_idx, min_len] = next_token.squeeze(dim=-1)
        else:
            past_key_values = None

        seq_block_idx = seq_len // block_size
        finished_flag = torch.zeros((batch_size), device=self.device, dtype=torch.bool)
        start_block_idx = min_len // block_size
        num_small_blocks = block_size // small_block_size
        
        if hasattr(self, 'layer_skip_state'):
             self.layer_skip_state['prev_input'] = None

        sample_indices = torch.arange(batch_size, device=self.device)
        finished_samples = {}
        
        for block_idx in range(start_block_idx, num_blocks):
            if finished_flag.all(): break
            
            if (seq_block_idx == block_idx).all():
                x_init = mask_id * torch.ones((input_ids.shape[0], block_size-input_ids.shape[1]%block_size), device=self.device, dtype=torch.long)
                x_init = torch.cat([input_ids, x_init], dim=1)
                input_ids = x_init
            else:
                x_init = input_ids[:, :(block_idx + 1)*block_size]
            x_init[finished_flag, -block_size:] = tokenizer.pad_token_id
            x_t = x_init.clone()
            step = 0
            block_past_key_values = None
            
            while True:
                mask_idx = (x_t[:, -block_size:] == mask_id)
                
                if mask_idx.sum() == 0:
                    for sample_idx in range(x_t.shape[0]):
                        if finished_flag[sample_idx] and seq_len[sample_idx] < (block_idx + 1) * block_size:
                            stop_token_idx = (x_t[sample_idx, seq_len[sample_idx]:] == stop_token).nonzero()[0][0]
                            x_t[sample_idx, seq_len[sample_idx]+stop_token_idx+1:] = tokenizer.pad_token_id
                    if finished_flag.all(): break
                    
                    self._skip_return_tuple = True 
                    self._skip_layer_enabled = False
                    output = self.forward(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, update_past_key_values=True, block_size=block_size)
                    logits, past_key_values = output.logits, output.past_key_values
                    next_token = logits[:, -1:, :].argmax(dim=-1)
                    next_token[finished_flag] = tokenizer.pad_token_id
                    x_t = torch.cat([x_t, next_token], dim=1)
                    step += 1

                    break
                
                for small_block_idx in range(num_small_blocks):
                    small_block_start_idx = small_block_idx * small_block_size
                    small_block_end_idx = small_block_start_idx + small_block_size
                    start = -block_size + small_block_start_idx
                    end = None if block_size == small_block_end_idx else -block_size + small_block_end_idx
                    
                    while True:
                        mask_idx = (x_t[:, -block_size:] == mask_id)
                        mask_idx_slice = mask_idx[:, start:end]
                        if mask_idx_slice.sum() == 0:
                            break
                        
                        if use_block_cache:
                            if block_past_key_values is None or (x_t[:, -block_size + small_block_start_idx] == mask_id).any():
                                self._skip_return_tuple = True
                                self._skip_layer_enabled = False
                                output = self.forward(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, update_past_key_values=False, block_size=block_size, use_block_cache=True)
                                logits, block_past_key_values = output.logits, output.block_past_key_values
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                                logits = logits[:, start:end]
                            else:
                                self._skip_return_tuple = True
                                self._skip_layer_enabled = False
                                output = self.forward(input_ids=x_t[:, start:end], use_cache=True, past_key_values=past_key_values, update_past_key_values=False, block_size=block_size, use_block_cache=True, block_past_key_values=block_past_key_values, replace_position=small_block_start_idx)
                                logits = output.logits
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                        else:
                            self._skip_return_tuple = False
                            self._skip_layer_enabled = True
                            output = self.forward(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, update_past_key_values=False, block_size=block_size)
                            logits = output.logits
                            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                            logits = logits[:, start:end]
                        p_1t = torch.softmax(logits, dim=-1)
                        x_1 = torch.argmax(p_1t, dim=-1)
                        x1_p = torch.squeeze(torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1)
                        x1_p = torch.where(mask_idx[:, start:end], x1_p, -torch.inf)
                        unmask_idx = (x1_p > threshold)
                        max_prob_idx = x1_p.argmax(dim=-1)
                        unmask_idx[torch.arange(x_1.shape[0]), max_prob_idx] = True
                        unmask_idx = unmask_idx & mask_idx[:, start:end]
                        x_t[:, start:end][unmask_idx] = x_1[unmask_idx]
                        finished_row_flags = ((x_1 == stop_token) & unmask_idx).any(dim=1)
                        finished_flag = finished_flag | finished_row_flags
                        step += 1

            if input_ids.shape[1] ==  x_t.shape[1]: input_ids = x_t
            else:
                input_ids[:, :(block_idx + 1)*block_size] = x_t[:, :-1]
                if (seq_block_idx == block_idx).all():
                    input_ids = torch.cat([input_ids, x_t[:, -1:]], dim=1)
                else:
                    if input_ids.shape[1] <= (block_idx + 1)*block_size: input_ids = x_t
                    else: input_ids[seq_block_idx == block_idx, (block_idx + 1)*block_size] = x_t[seq_block_idx == block_idx, (block_idx + 1)*block_size]
            seq_block_idx[seq_block_idx == block_idx] = block_idx + 1
            if finished_flag.any():
                for sample_idx in range(x_t.shape[0]):
                    if finished_flag[sample_idx]:
                        original_idx = sample_indices[sample_idx].item()
                        finished_samples[original_idx] = x_t[sample_idx:sample_idx+1].clone().squeeze(dim=0)
                sample_indices = sample_indices[~finished_flag]
                input_ids = input_ids[~finished_flag]
                seq_block_idx = seq_block_idx[~finished_flag]
                seq_len = seq_len[~finished_flag]
                x_t = x_t[~finished_flag]
                for layer_id in range(len(past_key_values)):
                    past_key_values.key_cache[layer_id] = past_key_values.key_cache[layer_id][~finished_flag]
                    past_key_values.value_cache[layer_id] = past_key_values.value_cache[layer_id][~finished_flag]
                finished_flag = finished_flag[~finished_flag]

        if len(finished_samples) < batch_size:
            for sample_idx in range(x_t.shape[0]):
                original_idx = sample_indices[sample_idx].item()
                finished_samples[original_idx] = x_t[sample_idx:sample_idx+1].clone().squeeze(dim=0)
        assert len(finished_samples) == batch_size
        
        if hasattr(self, 'layer_skip_state'):
            actual_flops = self.layer_skip_state['total_flops']
            print(f"[BENCHMARK] Layer-Skip Strategy | Total TFLOPs: {actual_flops / 1e12:.4f}")
        return finished_samples

    @torch.no_grad()
    def mdm_sample_with_visualization(self, *args, **kwargs): pass

class SkippingLayerWrapper(torch.nn.Module):
    def __init__(self, original_layer, layer_idx, model_ref, threshold):
        super().__init__()
        self.original_layer = original_layer
        self.layer_idx = layer_idx
        self.model_ref = weakref.ref(model_ref) 
        self.threshold = threshold
        layer_params = sum(p.numel() for p in original_layer.parameters())
        self.layer_flops_per_token = 2 * layer_params

    def forward(self, hidden_states, *args, **kwargs):
        model = self.model_ref()
        if model is None: return self.original_layer(hidden_states, *args, **kwargs)

        if isinstance(hidden_states, tuple): hidden_states = hidden_states[0]

        if not hasattr(model, 'layer_skip_state'):
            model.layer_skip_state = {'prev_input': None, 'total_flops': 0.0}
        
        should_return_tuple = getattr(model, '_skip_return_tuple', True)
        skipping_enabled = getattr(model, '_skip_layer_enabled', False)

        state = model.layer_skip_state
        
        
        if self.layer_idx == 0:
            state['prev_input'] = hidden_states.detach()
            batch_tokens = hidden_states.numel() / hidden_states.shape[-1]
            state['total_flops'] += batch_tokens * self.layer_flops_per_token
            
            outputs = self.original_layer(hidden_states, *args, **kwargs)
            if not should_return_tuple and isinstance(outputs, tuple): return outputs[0]
            return outputs

        should_skip = False
        if skipping_enabled and state['prev_input'] is not None:
            cos_sim = F.cosine_similarity(hidden_states.float(), state['prev_input'].float(), dim=-1).mean()
            if cos_sim > self.threshold:
                should_skip = True
        state['prev_input'] = hidden_states.detach()

        if should_skip:
            if should_return_tuple:
                return (hidden_states, None)
            return hidden_states
        
        batch_tokens = hidden_states.numel() / hidden_states.shape[-1]
        state['total_flops'] += batch_tokens * self.layer_flops_per_token
        
        outputs = self.original_layer(hidden_states, *args, **kwargs)
        if not should_return_tuple and isinstance(outputs, tuple): return outputs[0]
        return outputs

def apply_layer_skipping(model, cosine_threshold=0.98):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        raise ValueError("Could not find transformer layers at model.model.layers")

    if not hasattr(model, "layer_skip_state"):
        model.layer_skip_state = {"prev_input": None, "total_flops": 0.0}
    else:
        model.layer_skip_state["prev_input"] = None
        model.layer_skip_state["total_flops"] = 0.0

    for idx, layer in enumerate(layers):
        if isinstance(layer, SkippingLayerWrapper):
            layers[idx] = SkippingLayerWrapper(layer.original_layer, idx, model, cosine_threshold)
        else:
            layers[idx] = SkippingLayerWrapper(layer, idx, model, cosine_threshold)

def setup_model_with_custom_generation(model, custom_generation_fn: Optional[Callable]=None):
    if custom_generation_fn is not None:
        model.batch_sample = types.MethodType(custom_generation_fn, model)
    return model
