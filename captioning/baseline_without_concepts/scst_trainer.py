from transformers import Seq2SeqTrainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from pycocoevalcap.bleu.bleu import Bleu
import numpy as np
from transformers import (
    LogitsProcessorList,
    MaxLengthCriteria,
    MinLengthLogitsProcessor,
    StoppingCriteriaList,
    TopKLogitsWarper,
)
import sys


class RewardCriterion(nn.Module):

    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, scores, seq, reward, pad_token_id, reduction='mean'):
        # convert from list of tensors to single tensor
        scores = torch.stack(list(scores), dim=1)
        N, L = scores.shape[:2]

        scores = F.log_softmax(scores, dim=2)
        scores = scores.gather(2, seq.unsqueeze(2)).squeeze(
            2)  # gathers the dim 2 of scores with indexes seq
        scores[seq == pad_token_id] = 0.
        scores = scores.reshape(-1)

        reward = reward.reshape(-1)

        seq = torch.where(seq == pad_token_id, -100,
                          seq)  # to ignore pad tokens
        mask = (seq > 0).to(scores)
        mask = mask.reshape(-1)

        output = -scores * reward * mask
        if reduction == 'none':
            output = output.view(N, L).sum(1) / mask.view(N, L).sum(1)
        elif reduction == 'mean':
            output = torch.sum(output) / torch.sum(mask)
        return output


class CustomTrainer(Seq2SeqTrainer):

    def init_loss(self, model):
        self.scorer = Bleu(4)
        self.criterion = RewardCriterion()
        self.logits_processor = LogitsProcessorList([
            MinLengthLogitsProcessor(model.config.min_length,
                                     eos_token_id=model.config.eos_token_id),
        ])
        self.logits_warper = LogitsProcessorList([
            TopKLogitsWarper(model.config.decoder.top_k),
        ])
        self.stopping_criteria = StoppingCriteriaList(
            [MaxLengthCriteria(max_length=model.config.max_length)])
        print("MAX LEGTH " + str(model.config.max_length))

    def _array_to_str(self, arr):
        out = ''
        for i in range(len(arr)):
            out += str(arr[i]) + ' '
            if arr[i] == 0:
                break
        return out.strip()

    def _get_self_critical_reward(self, baseline_res, sample_res, references):
        device = sample_res.device
        batch_size = len(references)
        sample_res_size = sample_res.shape[0]
        seq_per_img = sample_res_size // len(
            references)  # sample_res_size  = batch_size * seq_per_img
        assert baseline_res.shape[0] == batch_size

        # sample_res = sample_res.data.cpu().numpy()
        # baseline_res = baseline_res.data.cpu().numpy()
        # references = references.data.cpu().numpy()

        sample_dict = {}
        for i, s in enumerate(sample_res):
            sample_dict[i] = [self._array_to_str(s)]

        baseline_dict = {}
        for i, b in enumerate(baseline_res):
            baseline_dict[i] = [self._array_to_str(b)]

        refs_dict = {}
        for i, r in enumerate(references):
            refs_dict[i] = [self._array_to_str(r)]

        _, baseline_bleu_scores = self.scorer.compute_score(refs_dict,
                                                            baseline_dict,
                                                            verbose=0)
        baseline_scores = np.array(baseline_bleu_scores[3]) * 100

        refs_dict_extended = {
            i: refs_dict[i // seq_per_img]
            for i in range(sample_res_size)
        }

        _, sample_bleu_scores = self.scorer.compute_score(refs_dict_extended,
                                                          sample_dict,
                                                          verbose=0)
        sample_scores = np.array(sample_bleu_scores[3]) * 100
        sample_scores = sample_scores.reshape(batch_size, seq_per_img)

        rewards = sample_scores - baseline_scores[:, np.newaxis]

        rewards = rewards.reshape(sample_res_size)

        # extend rewards to have 1 reward per token per generated sentence per image
        rewards = np.repeat(rewards[:, np.newaxis], sample_res.shape[1], 1)
        rewards = torch.Tensor(rewards).to(device)

        return rewards

    def compute_loss(self, model, input, return_outputs=False):
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)

        # we do not want to do teacher forcing, so we remove labels in order for generate to do free generation without conditioning
        if 'labels' in list(input.keys()):
            gts = input.pop('labels')

        pixel_values = input.pop('pixel_values')
        bs = pixel_values.shape[0]
        input["input_ids"] = torch.ones(
            (bs, 1), dtype=torch.long,
            device=model.device) * model.config.decoder_start_token_id

        # start.record()
        # we obtain the outputs of the encoder to avoid doing it twice (when calling the generate and sample methods)
        input["encoder_outputs"] = model.encoder(pixel_values)
        # print(input["encoder_outputs"])
        # end.record()
        # torch.cuda.synchronize()
        # print("Encoder " + str(start.elapsed_time(end)))

        # baseline
        # start.record()
        with torch.no_grad():
            baseline = model.greedy_search(
                **input, stopping_criteria=self.stopping_criteria
            )  #,logits_processor=self.logits_processor, logits_warper=self.logits_warper)
            baseline = baseline[:,
                                1:]  # ignore BOS token (since labels does not include it)
            baseline[baseline == model.config.pad_token_id] = -100
            # end.record()
            # torch.cuda.synchronize()
            # print("Baseline generate " + str(start.elapsed_time(end)))

        ############## multinomial sampling ##############
        # cannot use generate fn because it uses torch.no_grad() and we need gradients on the scores for backprop

        # build the input_ids for the decoder with the BOS token (as is done in the generate method)
        # start.record()
        sample = model.sample(
            **input,
            stopping_criteria=self.stopping_criteria,
            output_scores=True,
            return_dict_in_generate=True
        )  #, logits_processor=self.logits_processor, logits_warper=self.logits_warper,
        # sample["sequences"].detach()
        sample["sequences"] = sample["sequences"][:, 1:]  # ignore BOS token
        # end.record()
        # torch.cuda.synchronize()
        # print("Sample generate " + str(start.elapsed_time(end)))

        # start.record()
        with torch.no_grad():
            rewards = self._get_self_critical_reward(baseline,
                                                     sample['sequences'], gts)
        # end.record()
        # torch.cuda.synchronize()
        # print("Get critical reward " + str(start.elapsed_time(end)))

        # start.record()
        loss = self.criterion(sample["scores"], sample["sequences"], rewards,
                              model.config.pad_token_id)
        # end.record()
        # torch.cuda.synchronize()
        # print("Criterion " + str(start.elapsed_time(end)))

        # from torchviz import make_dot
        # dot = make_dot(loss, params=dict(model.named_parameters()))
        # # dot.format = 'svg'
        # dot.render()
        # quit()

        return loss if return_outputs is False else (loss, rewards)
