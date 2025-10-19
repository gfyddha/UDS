import torch
import torch.nn.functional as F
from transformers import Trainer
from typing import Dict, Any, Union
import torch.nn as nn


import torch
import torch.nn as nn
from typing import Dict, Union, Any
from transformers import Trainer

    
class FastJLProjector:
    def __init__(self, T, V, r1, r2, device="cuda"):
        self.T, self.V, self.r1, self.r2 = T, V, r1, r2
        self.device = device

        # ±1 masks stored as float32 on device
        self.d2 = (torch.randint(0, 2, (V,), device=device, dtype=torch.float32) * 2 - 1)
        self.idx2 = torch.randint(0, V, (r2,), device=device)

        self.d1 = (torch.randint(0, 2, (T,), device=device, dtype=torch.float32) * 2 - 1)
        self.idx1 = torch.randint(0, T, (r1,), device=device)

        # precompute scaling factors
        self.scale_r = (self.V ** 0.5) / (self.r2 ** 0.5)
        self.scale_l = (self.T ** 0.5) / (self.r1 ** 0.5)

    def project(self, X):
        """
        Maintain single-sample API for compatibility: X shape (T, V) -> returns (r1, r2)
        Internally delegates to project_batch for efficiency.
        """
        if X.dim() == 2:
            out = self.project_batch(X.unsqueeze(0))  # (1, r1, r2)
            return out[0]
        else:
            raise ValueError("project expects 2D tensor (T, V)")

    def project_batch(self, X):
        """
        Batch version: X shape (B, T, V) -> returns (B, r1, r2)
        Vectorized: apply right transform, FFT along last dim, select idx2, apply left transform, FFT along time dim, select idx1.
        """
        # Expect X on same device; but ensure
        X = X.to(self.device)

        # X: (B, T, V)
        # Right transform: multiply by d2 (V) -> FFT along last dim -> select idx2
        # result: (B, T, r2)
        X = X * self.d2  # broadcasting on last dim
        X = torch.fft.fft(X, dim=-1).real
        X = X[..., self.idx2]  # (B, T, r2)
        X = X * self.scale_r

        # Left transform: multiply rows by d1, FFT across time dim (dim=1), then select idx1
        X = X * self.d1.view(1, self.T, 1)  # broadcast multiply on time dim
        X = torch.fft.fft(X, dim=1).real  # keep real part
        X = X[:, self.idx1, :]  # (B, r1, r2)
        X = X * self.scale_l

        return X.contiguous()
    

class UDSamplingTrainer(Trainer):
    def __init__(
        self, 
        model,
        args = None,
        data_collator = None,
        train_dataset = None,
        eval_dataset = None,
        tokenizer = None,
        model_init = None,
        compute_metrics = None,
        callbacks = None,
        optimizers = (None, None),
        preprocess_logits_for_metrics = None,
        **kwargs,
    ):
        start_sampling_step = kwargs.pop('start_sampling_step', 0)
        queue_size = kwargs.pop('queue_size', 1024)
        projection_dim1 = kwargs.pop('projection_dim1', 128)
        projection_dim2 = kwargs.pop('projection_dim2', 8)

        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)
        
        self.start_sampling_step = start_sampling_step
        self.queue_size = queue_size
        self.projection_dim1 = projection_dim1
        self.projection_dim2 = projection_dim2
        self.alpha = kwargs.pop('alpha', 2.5e-3)
        self.K_base = kwargs.pop('k_base', 4)
        self.random = kwargs.pop('random', False)
        self.output_file = kwargs.pop('output_file', 'out.txt')
        
        self.history_count = 0
        self.history_sum = 0.0
        self.history_sum_sq = 0.0

        self.repr_queue = None
        self.queue_ptr = 0
        self.queue_count = 0
        
        self.vocab_size = model.config.vocab_size
        self.max_seq_len = args.max_seq_length if hasattr(args, 'max_seq_length') else 512
        self.projector = FastJLProjector(self.max_seq_len, self.vocab_size, 
                self.projection_dim2, self.projection_dim1, device="cuda")
        print(self.vocab_size)
        
        self.scoring_time = 0.0
        self.forward_time = 0.0
        self.backward_time = 0.0

    @torch.no_grad()
    def compute_diversity_scores(self, logits):
        """
        logits: (B, L, V)
        returns: (diversity_scores: (B,), projected_repr: (B, projection_dim1*projection_dim2))
        Vectorized implementation using batch projector.
        """
        batch_size, seq_len, vocab_size = logits.shape

        # pad only if necessary; attempt to avoid allocating if seq_len == max_seq_len
        if seq_len < self.max_seq_len:
            padded = logits.new_zeros((batch_size, self.max_seq_len, vocab_size))
            padded[:, :seq_len, :] = logits
            logits = padded
            seq_len = self.max_seq_len

        projected = torch.zeros(batch_size, self.projector.r1 * self.projector.r2, device=logits.device)
        for i in range(batch_size):
            projected[i] = self.projector.project(logits[i]).view(1, -1)  # (B, r1 * r2)

        # if queue empty -> zero diversity scores
        if self.repr_queue is None or self.queue_count == 0:
            zeros = torch.zeros(batch_size, device=logits.device, dtype=logits.dtype)
            return zeros, projected

        # valid queue slice on same device as projected (move queue if needed)
        # ensure queue is on device
        if self.repr_queue.device != projected.device:
            self.repr_queue = self.repr_queue.to(projected.device)

        valid_queue = self.repr_queue[:self.queue_count]  # (queue_count, D)
        # distances: (B, queue_count)
        distances = torch.cdist(projected, valid_queue, p=2)
        diversity_scores = distances.mean(dim=1)
        return diversity_scores, projected

    def update_repr_queue(self, projected_repr):
        """
        Vectorized FIFO update of repr queue.
        projected_repr: (B, D)
        """
        batch_size, D = projected_repr.shape
        if self.repr_queue is None:
            # allocate queue on same device as projected_repr
            self.repr_queue = torch.zeros(self.queue_size, D, device=projected_repr.device, dtype=projected_repr.dtype)
            self.queue_ptr = 0
            self.queue_count = 0

        # Write batch into queue with wrap-around
        # compute indices to write to
        idx = (torch.arange(self.queue_ptr, self.queue_ptr + batch_size, device=projected_repr.device) % self.queue_size)
        self.repr_queue[idx] = projected_repr
        # update pointer and count
        self.queue_ptr = int((self.queue_ptr + batch_size) % self.queue_size)
        self.queue_count = min(self.queue_size, self.queue_count + batch_size)

    @torch.no_grad()
    def compute_top_index(self, model, inputs, debug: bool = True):
        """
        Non-mutating: do NOT pop/modify inputs dict.
        Returns topk indices (on same device as inputs tensors).
        """
        # extract labels non-destructively
        labels = inputs.pop("labels", None)
        # call model
        model_to_call = self.accelerator.unwrap_model(model)
        with torch.no_grad():
            outputs = model_to_call(**inputs)
        if labels is not None:
            inputs["labels"] = labels
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        logits = outputs.logits  # (B,L,V)
        B, L, V = logits.shape

        # utility: batch SVD singular values sum
        # use svdvals which supports batch; result shape (B, min(L,V))
        svals = torch.linalg.svdvals(logits)  # (B, K)
        utility_scores = svals.sum(dim=1)

        diversity_scores, projected_repr = self.compute_diversity_scores(logits)

        if debug:
            print("utility score: ", utility_scores)
            print("diversity score: ", diversity_scores)

        combined_scores = utility_scores + self.alpha * diversity_scores
        if debug:
            print("combined score: ", combined_scores)

        _, topk_idx = torch.topk(combined_scores, self.K_base, largest=True, sorted=False)
        if debug:
            print(f"K_t = {len(topk_idx)}")

        # update queue with selected projected repr (move to same device)
        if topk_idx.numel() > 0:
            self.update_repr_queue(projected_repr[topk_idx].to(self.repr_queue.device if self.repr_queue is not None else projected_repr.device))

        self.history_count += int(topk_idx.numel())
        # minimally touching file I/O like before
        with open(self.output_file, "w") as f:
            f.write(f"history_count = {self.history_count}\n")

        return topk_idx.to(logits.device)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.state.global_step >= self.start_sampling_step:
            if self.random == False:
                topk_idx = self.compute_top_index(model, inputs)
            else:
                batch_size = inputs['input_ids'].size(0) if 'input_ids' in inputs else next(iter(inputs.values())).size(0)
                print(batch_size)
                K = min(self.K_base, batch_size)
                topk_idx = torch.randperm(batch_size, device="cuda")[:K]

            small_inputs = {
                k: torch.index_select(v, dim=0, index=topk_idx) if torch.is_tensor(v) else v
                for k, v in inputs.items()
            }
        else:
            small_inputs = inputs

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, small_inputs)
        
        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            torch.cuda.empty_cache()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        self.accelerator.backward(loss)

        return (loss / self.args.gradient_accumulation_steps).detach()
