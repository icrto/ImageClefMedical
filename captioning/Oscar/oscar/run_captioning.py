# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.

import argparse
import os
import os.path as op
import random
import time
import json
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from torchvision import transforms as T
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import datetime

from oscar.utils.logger import setup_logger
from oscar.utils.tsv_file_ops import (tsv_writer, concat_tsv_files,
                                      delete_tsv_files, reorder_tsv_keys)
from oscar.utils.misc import (mkdir, set_seed)
from oscar.utils.caption_evaluate import evaluate_on_coco_caption
from oscar.modeling.modeling_bert import BertForImageCaptioning
from transformers import BertTokenizer, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from pycocotools.coco import COCO
from PIL import Image
import pandas as pd


class CaptionDataset(Dataset):

    def __init__(self,
                 base_path,
                 captions_file,
                 tokenizer=None,
                 add_concepts=True,
                 concept_map_file=None,
                 concept_imgs_file=None,
                 img_seq_length=50,
                 max_seq_length=70,
                 max_seq_a_length=40,
                 is_train=True,
                 mask_prob=0.15,
                 max_masked_tokens=3,
                 transform=None,
                 **kwargs):

        self.base_path = base_path
        self.captions_file = captions_file

        self.concept_map_file = concept_map_file
        self.concept_data_file = concept_imgs_file

        if add_concepts:
            assert op.isfile(self.concept_map_file)
            assert op.isfile(self.concept_data_file)

        if is_train:
            assert op.isfile(self.captions_file) and tokenizer is not None

        self.captioning_data = COCO(self.captions_file)
        self.image_keys = self.captioning_data.getImgIds()[:50]

        self.tokenizer = tokenizer
        self.tensorizer = CaptionTensorizer(self.tokenizer,
                                            img_seq_length,
                                            max_seq_length,
                                            max_seq_a_length,
                                            mask_prob,
                                            max_masked_tokens,
                                            is_train=is_train)
        self.add_concepts = add_concepts
        self.is_train = is_train
        self.transform = transform
        self.kwargs = kwargs

        self.captions = self.prepare_captions()
        self.imgkey2caption = self.prepare_image_key_to_captions()

        if self.add_concepts:
            self.cuis2conceptnames = self.prepare_concept_mapping()
            self.imgkey2cuis = self.prepare_image_key_to_cuis()
            self.imgkey2conceptnames = self.prepare_image_key_to_concept_names(
            )

    def prepare_captions(self):
        ids_cap = self.captioning_data.getAnnIds(imgIds=self.image_keys)
        return [d for d in self.captioning_data.loadAnns(ids=ids_cap)]

    def prepare_image_key_to_captions(self):
        if self.captions:
            key2captions = {key: None for key in self.image_keys}
            for cap in self.captions:
                key2captions[cap['image_id']] = cap['caption']
            return key2captions

    def prepare_concept_mapping(self):

        to_remove = [" (qualifier value)", " (severity modifier)"]
        df = pd.read_csv(self.concept_map_file, sep="\t")

        concept_dict = {}
        for _, r in df.iterrows():
            raw_concept = r['concept_name'].lower().replace(",", "")
            for remov in to_remove:
                raw_concept = raw_concept.replace(remov, "")

            concept_dict[r['concept']] = raw_concept

        return concept_dict

    def prepare_image_key_to_cuis(self):
        df = pd.read_csv(self.concept_data_file, header=0, sep="\t")
        data = {}
        for _, r in df.iterrows():
            data[r['ID']] = r['cuis'].split(";")
        return data

    def prepare_image_key_to_concept_names(self):
        data = {}
        for key, values in self.imgkey2cuis.items():
            data[key] = [self.cuis2conceptnames[cui] for cui in values]
        return data

    def get_image_key(self, img_idx):
        return self.image_keys[img_idx]

    def get_caption(self, img_key):
        return self.imgkey2caption[img_key]

    def get_concepts(self, img_key):
        concepts = " ".join([l for l in self.imgkey2conceptnames[img_key]])
        return concepts

    def get_captions_by_key(self, key):
        return self.key2captions[key]

    def __getitem__(self, img_idx):
        # image
        img_key = self.get_image_key(img_idx)

        if 'train' in img_key:
            subset_folder = 'train'
        elif 'valid' in img_key:
            subset_folder = 'valid'
        else:
            subset_folder = 'test'

        img_path = os.path.join(
            self.base_path,
            subset_folder,
            img_key + '.jpg',
        )

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        # text
        if self.is_train:
            caption = self.get_caption(img_key)
        else:
            caption = None

        # concepts
        if self.add_concepts:
            concepts = self.get_concepts(img_key)
        else:
            concepts = None

        example = self.tensorizer.tensorize_example(caption,
                                                    img,
                                                    text_b=concepts)
        return img_key, example

    def __len__(self):

        return len(self.image_keys)


class CaptionTensorizer(object):

    def __init__(self,
                 tokenizer,
                 img_seq_length=50,
                 max_seq_length=70,
                 max_seq_a_length=40,
                 mask_prob=0.15,
                 max_masked_tokens=3,
                 is_train=True):
        """Constructor.
        Args:
            tokenizer: tokenizer for text processing.
            img_seq_length: image sequence length.
            max_seq_length: max text sequence length.
            max_seq_a_length: max caption sequence length.
            is_train: train or test mode.
            mask_prob: probability to mask a input token.
            max_masked_tokens: maximum number of tokens to be masked in one sentence.
        """
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.img_seq_len = img_seq_length
        self.max_seq_len = max_seq_length
        self.max_seq_a_len = max_seq_a_length
        self.mask_prob = mask_prob
        self.max_masked_tokens = max_masked_tokens
        self._triangle_mask = torch.tril(
            torch.ones((self.max_seq_len, self.max_seq_len), dtype=torch.long))

    def tensorize_example(self,
                          text_a,
                          img,
                          text_b=None,
                          cls_token_segment_id=0,
                          pad_token_segment_id=0,
                          sequence_a_segment_id=0,
                          sequence_b_segment_id=1):
        if self.is_train:
            tokens_a = self.tokenizer.tokenize(text_a)
        else:
            # fake tokens to generate masks
            tokens_a = [self.tokenizer.mask_token] * (self.max_seq_a_len - 2)
        if len(tokens_a) > self.max_seq_a_len - 2:
            tokens_a = tokens_a[:(self.max_seq_a_len - 2)]

        tokens = [self.tokenizer.cls_token
                  ] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id
                       ] + [sequence_a_segment_id] * (len(tokens) - 1)
        seq_a_len = len(tokens)
        if text_b:
            # pad text_a to keep it in fixed length for better inference.
            padding_a_len = self.max_seq_a_len - seq_a_len
            tokens += [self.tokenizer.pad_token] * padding_a_len
            segment_ids += ([pad_token_segment_id] * padding_a_len)

            tokens_b = self.tokenizer.tokenize(text_b)
            if len(tokens_b) > self.max_seq_len - len(tokens) - 1:
                tokens_b = tokens_b[:(self.max_seq_len - len(tokens) - 1)]
            tokens += tokens_b + [self.tokenizer.sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        seq_len = len(tokens)
        if self.is_train:
            masked_pos = torch.zeros(self.max_seq_len, dtype=torch.int)
            # randomly mask words for prediction, ignore [CLS]
            candidate_masked_idx = list(range(1,
                                              seq_a_len))  # only mask text_a
            random.shuffle(candidate_masked_idx)
            num_masked = min(round(self.mask_prob * seq_a_len, 1),
                             self.max_masked_tokens)
            num_masked = int(num_masked)
            masked_idx = candidate_masked_idx[:num_masked]
            masked_idx = sorted(masked_idx)
            masked_token = [tokens[i] for i in masked_idx]
            for pos in masked_idx:
                if random.random() <= 0.8:
                    # 80% chance to be a ['MASK'] token
                    tokens[pos] = self.tokenizer.mask_token
                elif random.random() <= 0.5:
                    # 10% chance to be a random word ((1-0.8)*0.5)
                    from random import randint
                    i = randint(0, len(self.tokenizer.vocab))
                    self.tokenizer._convert_id_to_token(i)
                    tokens[pos] = self.tokenizer._convert_id_to_token(i)
                else:
                    # 10% chance to remain the same (1-0.8-0.1)
                    pass

            masked_pos[masked_idx] = 1
            # pad masked tokens to the same length
            if num_masked < self.max_masked_tokens:
                masked_token = masked_token + (
                    [self.tokenizer.pad_token] *
                    (self.max_masked_tokens - num_masked))
            masked_ids = self.tokenizer.convert_tokens_to_ids(masked_token)
        else:
            masked_pos = torch.ones(self.max_seq_len, dtype=torch.int)

        # pad on the right for image captioning
        padding_len = self.max_seq_len - seq_len
        tokens = tokens + ([self.tokenizer.pad_token] * padding_len)
        segment_ids += ([pad_token_segment_id] * padding_len)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # prepare attention mask:
        # note that there is no attention from caption to image
        # because otherwise it will violate the triangle attention
        # for caption as caption will have full attention on image.
        max_len = self.max_seq_len + self.img_seq_len
        attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
        # C: caption, L: label, R: image region
        c_start, c_end = 0, seq_a_len
        l_start, l_end = self.max_seq_a_len, seq_len
        r_start, r_end = self.max_seq_len, self.max_seq_len + self.img_seq_len
        # triangle mask for caption to caption
        attention_mask[c_start:c_end,
                       c_start:c_end].copy_(self._triangle_mask[0:seq_a_len,
                                                                0:seq_a_len])
        # full attention for L-L, R-R
        attention_mask[l_start:l_end, l_start:l_end] = 1
        attention_mask[r_start:r_end, r_start:r_end] = 1
        # full attention for C-L, C-R
        attention_mask[c_start:c_end, l_start:l_end] = 1
        attention_mask[c_start:c_end, r_start:r_end] = 1
        # full attention for L-R:
        attention_mask[l_start:l_end, r_start:r_end] = 1
        attention_mask[r_start:r_end, l_start:l_end] = 1

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)

        if self.is_train:
            masked_ids = torch.tensor(masked_ids, dtype=torch.long)
            return (input_ids, attention_mask, segment_ids, img, masked_pos,
                    masked_ids)
        return (input_ids, attention_mask, segment_ids, img, masked_pos)


def build_dataset(captions_file, concept_map_file, concept_imgs_file, tokenizer, args, transform=None, is_train=True):
    if is_train:
        return CaptionDataset(base_path=os.path.dirname(captions_file),
                              captions_file=captions_file,
                              concept_map_file=concept_map_file,
                              concept_imgs_file=concept_imgs_file,
                              tokenizer=tokenizer,
                              add_concepts=args.add_concepts,
                              img_seq_length=args.img_seq_length,
                              max_seq_length=args.max_seq_length,
                              max_seq_a_length=args.max_seq_a_length,
                              transform=transform,
                              is_train=True,
                              mask_prob=args.mask_prob,
                              max_masked_tokens=args.max_masked_tokens,
                              )

    return CaptionDataset(base_path=os.path.dirname(captions_file),
                          captions_file=captions_file,
                          concept_map_file=concept_map_file,
                          concept_imgs_file=concept_imgs_file,
                          tokenizer=tokenizer,
                          add_concepts=args.add_concepts,
                          img_seq_length=args.img_seq_length,
                          max_seq_length=args.max_seq_length,
                          max_seq_a_length=args.max_seq_a_length,
                          transform=transform,
                          is_train=False
                          )


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset,
                                                               shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_data_loader(args,
                     captions_file,
                     concept_map_file,
                     concept_imgs_file,
                     tokenizer,
                     transform=None,
                     is_distributed=True,
                     is_train=True):

    dataset = build_dataset(captions_file,
                            concept_map_file,
                            concept_imgs_file,
                            tokenizer,
                            args,
                            transform,
                            is_train=is_train)
    if is_train:
        shuffle = True
        images_per_gpu = args.per_gpu_train_batch_size
        images_per_batch = images_per_gpu * get_world_size()
        iters_per_batch = len(dataset) // images_per_batch
        num_iters = iters_per_batch * args.num_train_epochs
        logger.info("Train with {} images per GPU.".format(images_per_gpu))
        logger.info("Total batch size {}".format(images_per_batch))
        logger.info("Total training steps {}".format(num_iters))
    else:
        shuffle = False
        images_per_gpu = args.per_gpu_eval_batch_size

    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        sampler=sampler,
        batch_size=images_per_gpu,
        pin_memory=True,
    )
    return data_loader


def save_checkpoint(model, tokenizer, args, epoch, iteration, num_trial=10):
    checkpoint_dir = op.join(args.output_dir,
                             'checkpoint-{}-{}'.format(epoch, iteration))
    if not is_main_process():
        return checkpoint_dir
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    for _ in range(num_trial):
        try:
            model_to_save.save_pretrained(checkpoint_dir)
            torch.save(args, op.join(checkpoint_dir, 'training_args.bin'))
            tokenizer.save_pretrained(checkpoint_dir)
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            pass
    else:
        logger.info(
            "Failed to save checkpoint after {} trials.".format(num_trial))
    return checkpoint_dir


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, -1)[1].data  # argmax
    scores = logits == labels
    return scores


def train(args, train_dataloader, val_dataloader, model, tokenizer):
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) //
                                                   args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps \
            * args.num_train_epochs

    # Prepare optimizer and scheduler
    no_decay = ['bias', 'LayerNorm.weight']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not
                    any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if
                    any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(grouped_parameters,
                      lr=args.learning_rate,
                      eps=args.adam_epsilon)
    if args.scheduler == "constant":
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps)
    elif args.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    else:
        raise ValueError("Unknown scheduler type: {}".format(args.scheduler))

    logger.info("***** Running training *****")
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, & accumulation) = %d",
        args.per_gpu_train_batch_size * get_world_size() *
        args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step, global_loss, global_acc = 0, 0.0, 0.0
    model.zero_grad()
    eval_log = []
    best_score = 0
    tb_writer = SummaryWriter(log_dir=args.output_dir)
    for epoch in range(int(args.num_train_epochs)):
        for step, (img_keys, batch) in enumerate(train_dataloader):
            batch = tuple(t.to(args.device) for t in batch)

            model.train()
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2],
                'pixel_values': batch[3],
                'masked_pos': batch[4],
                'masked_ids': batch[5]
            }
            outputs = model(**inputs)
            loss, logits = outputs[:2]
            masked_ids = inputs['masked_ids']
            masked_ids = masked_ids[masked_ids != 0]
            batch_score = compute_score_with_logits(logits, masked_ids)
            batch_acc = torch.sum(batch_score.float()) / torch.sum(
                inputs['masked_pos'])

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           args.max_grad_norm)
            global_loss += loss.item()
            global_acc += batch_acc
            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                if global_step % args.logging_steps == 0:
                    logger.info("Epoch: {}, global_step: {}, lr: {:.6f}, loss: {:.4f} ({:.4f}), "
                                "score: {:.4f} ({:.4f})".format(epoch, global_step,
                                                                optimizer.param_groups[0]["lr"], loss, global_loss /
                                                                global_step,
                                                                batch_acc, global_acc / global_step)
                                )
                    tb_writer.add_scalar(
                        'train/global_loss', global_loss / global_step, global_step)
                    tb_writer.add_scalar(
                        'train/batch_loss', loss,  global_step)
                    tb_writer.add_scalar(
                        'train/global_acc', global_acc / global_step, global_step
                    )
                    tb_writer.add_scalar(
                        'train/batch_acc', batch_acc, global_step
                    )
                    tb_writer.add_scalar(
                        'lr', scheduler.get_last_lr()[0], global_step
                    )

                if (args.save_steps > 0 and global_step % args.save_steps == 0) or \
                        global_step == t_total:
                    checkpoint_dir = save_checkpoint(model, tokenizer, args,
                                                     epoch, global_step)
                    # evaluation
                    if args.evaluate_during_training:
                        logger.info("Perform evaluation at step: %d" %
                                    (global_step))
                        evaluate_file = evaluate(args, val_dataloader, model,
                                                 tokenizer, checkpoint_dir)
                        with open(evaluate_file, 'r') as f:
                            res = json.load(f)
                        for k, v in res.items():
                            tb_writer.add_scalar('val/' + k, v, global_step)
                        best_score = max(best_score, res['CIDEr'])
                        res['best_CIDEr'] = best_score
                        res['epoch'] = epoch
                        res['global_step'] = global_step
                        eval_log.append(res)

                        with open(os.path.join(args.output_dir, 'eval_logs.json'), 'w') as f:
                            json.dump(eval_log, f)
                if global_step > t_total:
                    return checkpoint_dir

    return checkpoint_dir


def get_predict_filename(output_dir, json_file, args):
    cc = ['pred']
    # make sure it works with/without / in end of the path.
    data = op.basename(op.join(args.data_dir, '')[:-1])
    split = op.basename(json_file)
    assert split.endswith('.json')
    split = split[:-5]
    cc.append(data)
    cc.append(split)
    cc.append('beam{}'.format(args.num_beams))
    cc.append('max{}'.format(args.max_gen_length))
    if args.add_concepts:
        cc.append('concepts')
    if args.num_keep_best != 1:
        cc.append('best{}'.format(args.num_keep_best))
    if args.output_hidden_states:
        cc.append('hidden')
    return op.join(output_dir, '{}.tsv'.format('.'.join(cc)))


def get_evaluate_filename(predict_file):
    assert predict_file.endswith('.tsv')
    fpath = op.splitext(predict_file)[0]
    return fpath + '.eval.json'


def evaluate(args, val_dataloader, model, tokenizer, output_dir):
    predict_file = get_predict_filename(output_dir,
                                        val_dataloader.dataset.captions_file, args)
    test(args, val_dataloader, model, tokenizer, predict_file)

    if get_world_size() > 1:
        torch.distributed.barrier()
    evaluate_file = get_evaluate_filename(predict_file)
    if is_main_process():
        caption_file = val_dataloader.dataset.captions_file
        result = evaluate_on_coco_caption(predict_file,
                                          caption_file,
                                          outfile=evaluate_file)
        logger.info('evaluation result: {}'.format(str(result)))
        logger.info('evaluation result saved to {}'.format(evaluate_file))
    if get_world_size() > 1:
        torch.distributed.barrier()
    return evaluate_file


def test(args, test_dataloader, model, tokenizer, predict_file):
    cls_token_id, sep_token_id, pad_token_id, mask_token_id = \
        tokenizer.convert_tokens_to_ids([tokenizer.cls_token, tokenizer.sep_token,
                                         tokenizer.pad_token, tokenizer.mask_token])
    world_size = get_world_size()
    if world_size == 1:
        cache_file = predict_file
    else:
        cache_file = op.splitext(predict_file)[0] + '_{}_{}'.format(
            get_rank(), world_size) + op.splitext(predict_file)[1]

    model.eval()
    inputs_param = {
        'is_decode': True,
        'do_sample': False,
        'bos_token_id': cls_token_id,
        'pad_token_id': pad_token_id,
        'eos_token_ids': [sep_token_id],
        'mask_token_id': mask_token_id,
        # for adding concepts
        'add_concepts': args.add_concepts,
        'concepts_start_posid': args.max_seq_a_length,

        # hyperparameters of beam search
        'max_length': args.max_gen_length,
        'num_beams': args.num_beams,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "length_penalty": args.length_penalty,
        "num_return_sequences": args.num_return_sequences,
        "num_keep_best": args.num_keep_best,
    }

    def gen_rows():
        time_meter = 0

        with torch.no_grad():
            for step, (img_keys, batch) in enumerate(tqdm(test_dataloader)):
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],
                    'pixel_values': batch[3],
                    'masked_pos': batch[4],
                }
                inputs.update(inputs_param)
                tic = time.time()
                # captions, logprobs
                outputs = model(**inputs)
                time_meter += time.time() - tic
                all_caps = outputs[0]  # batch_size * num_keep_best * max_len
                all_confs = torch.exp(outputs[1])

                for img_key, caps, confs in zip(img_keys, all_caps, all_confs):
                    res = []
                    for cap, conf in zip(caps, confs):
                        cap = tokenizer.decode(cap.tolist(),
                                               skip_special_tokens=True)
                        res.append({'caption': cap, 'conf': conf.item()})
                    if isinstance(img_key, torch.Tensor):
                        img_key = img_key.item()
                    yield img_key, json.dumps(res)

        logger.info(
            "Inference model computing time: {} seconds per batch".format(
                time_meter / (step + 1)))

    tsv_writer(gen_rows(), cache_file)
    if world_size > 1:
        torch.distributed.barrier()
    if world_size > 1 and is_main_process():
        cache_files = [op.splitext(predict_file)[0] + '_{}_{}'.format(i, world_size) +
                       op.splitext(predict_file)[1] for i in range(world_size)]
        concat_tsv_files(cache_files, predict_file)
        delete_tsv_files(cache_files)
        reorder_tsv_keys(predict_file, test_dataloader.dataset.image_keys,
                         predict_file)
    if world_size > 1:
        torch.distributed.barrier()


def restore_training_settings(args):
    if args.do_train:
        if args.model_name_or_path is None:
            return args
        checkpoint = args.model_name_or_path
    else:
        assert args.do_test or args.do_eval
        checkpoint = args.eval_model_dir
        assert op.isdir(checkpoint)
    # restore training settings, check hasattr for backward compatibility
    train_args = torch.load(op.join(checkpoint, 'training_args.bin'))
    if hasattr(train_args, 'max_seq_a_length'):
        max_concepts_len = train_args.max_seq_length - train_args.max_seq_a_length
        max_seq_length = args.max_gen_length + max_concepts_len
        args.max_seq_length = max_seq_length
        logger.warning(
            'Override max_seq_length to {} = max_gen_length:{} + max_concepts_len:{}'
            .format(max_seq_length, args.max_gen_length, max_concepts_len))

    override_params = [
        'max_seq_a_length', 'do_lower_case', 'add_concepts',
        'img_seq_length'
    ]
    for param in override_params:
        if hasattr(train_args, param):
            train_v = getattr(train_args, param)
            test_v = getattr(args, param)
            if train_v != test_v:
                logger.warning('Override {} with train args: {} -> {}'.format(
                    param, test_v, train_v))
                setattr(args, param, train_v)
    return args


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def ensure_init_process_group(local_rank=None, port=12345):
    # init with env
    world_size = int(
        os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    if world_size > 1 and not dist.is_initialized():
        assert local_rank is not None
        print("Init distributed training on local rank {}".format(local_rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
    return local_rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        default='0',
        type=str,
        help="GPU id.")
    parser.add_argument("--data_dir",
                        default='dataset/',
                        type=str,
                        required=False,
                        help="The input data dir with all required files.")
    parser.add_argument(
        "--train_json",
        default='dataset/caption_prediction_train_coco.json',
        type=str,
        required=False,
        help="json (in coco format) file for training.")
    parser.add_argument(
        "--val_json",
        default='dataset/caption_prediction_valid_coco.json',
        type=str,
        required=False,
        help="json (in coco format) file used for validation during training.")
    parser.add_argument(
        "--concepts_csv",
        default='dataset/concepts.csv',
        type=str,
        required=False,
        help="csv file with the concepts and their cuis.")
    parser.add_argument(
        "--train_concepts_csv",
        default='dataset/concept_detection_train.csv',
        type=str,
        required=False,
        help="csv file with the concepts of every training image.")
    parser.add_argument(
        "--val_concepts_csv",
        default='dataset/concept_detection_valid.csv',
        type=str,
        required=False,
        help="csv file with the concepts of every validation image.")
    parser.add_argument("--model_name_or_path",
                        default=None,
                        type=str,
                        required=False,
                        help="Path to pre-trained model or model type.")
    parser.add_argument(
        "--output_dir",
        default='results',
        type=str,
        required=False,
        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--loss_type",
                        default='sfmx',
                        type=str,
                        help="Loss function types: support kl, x2, sfmx")
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name.")
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name."
    )
    parser.add_argument(
        "--max_seq_length",
        default=150,
        type=int,
        help="The maximum total input sequence length after tokenization. "
        "Sequences longer than this will be truncated, "
        "sequences shorter will be padded.")
    parser.add_argument("--max_seq_a_length",
                        default=100,
                        type=int,
                        help="The maximum sequence length for caption.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run inference.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run evaluation.")
    parser.add_argument(
        "--do_lower_case",
        action='store_true',
        help="Set this flag if you are using an uncased model.")
    parser.add_argument(
        "--mask_prob",
        default=0.15,
        type=float,
        help="Probability to mask input sentence during training.")
    parser.add_argument("--max_masked_tokens",
                        type=int,
                        default=3,
                        help="The max number of masked tokens per sentence.")
    parser.add_argument("--add_concepts",
                        default=False,
                        action='store_true',
                        help="Whether to add concept tags or not")
    parser.add_argument("--drop_out",
                        default=0.1,
                        type=float,
                        help="Drop out in BERT.")
    parser.add_argument("--img_size",
                        default=224,
                        type=int,
                        help="Input image size.")
    parser.add_argument("--num_channels",
                        default=3,
                        type=int,
                        help="Number of channels of input images.")
    parser.add_argument("--patch_size",
                        default=16,
                        type=int,
                        help="Patch size for image patches.")
    parser.add_argument("--img_embed_dim",
                        default=768,
                        type=int,
                        help="Embedding dimension of images.")
    parser.add_argument(
        "--tie_weights",
        default=False,
        action='store_true',
        help="Whether to tie decoding weights to that of encoding")
    parser.add_argument("--freeze_embedding",
                        default=False,
                        action='store_true',
                        help="Whether to freeze word embeddings in Bert")
    parser.add_argument("--label_smoothing", default=0, type=float, help=".")
    parser.add_argument("--drop_worst_ratio", default=0, type=float, help=".")
    parser.add_argument("--drop_worst_after", default=0, type=int, help=".")
    parser.add_argument("--per_gpu_train_batch_size",
                        default=16,
                        type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size",
                        default=16,
                        type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument(
        "--output_mode",
        default='classification',
        type=str,
        help="output mode, support classification or regression.")
    parser.add_argument(
        "--num_labels",
        default=2,
        type=int,
        help="num_labels is 2 for classification and 1 for regression.")
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help="Number of updates steps to accumulate before backward.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial lr.")
    parser.add_argument("--weight_decay",
                        default=0.05,
                        type=float,
                        help="Weight deay.")
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for Adam.")
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps",
                        default=0,
                        type=int,
                        help="Linear warmup.")
    parser.add_argument("--scheduler",
                        default='linear',
                        type=str,
                        help="constant or linear or")
    parser.add_argument("--num_workers",
                        default=4,
                        type=int,
                        help="Workers in dataloader.")
    parser.add_argument("--num_train_epochs",
                        default=30,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="Total number of training steps. Override num_train_epochs.")
    parser.add_argument('--logging_steps',
                        type=int,
                        default=20,
                        help="Log every X steps.")
    parser.add_argument(
        '--save_steps',
        type=int,
        default=-1,
        help="Save checkpoint every X steps.")
    parser.add_argument(
        "--evaluate_during_training",
        action='store_true',
        help="Run evaluation during training at each save_steps.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=0,
                        help="For distributed training.")
    parser.add_argument('--seed',
                        type=int,
                        default=88,
                        help="random seed for initialization.")
    
    # for generation
    parser.add_argument("--eval_model_dir",
                        type=str,
                        default='',
                        help="Model directory for evaluation.")
    parser.add_argument('--max_gen_length',
                        type=int,
                        default=150,
                        help="max length of generated sentences")
    parser.add_argument('--output_hidden_states',
                        action='store_true',
                        help="Turn on for fast decoding")
    parser.add_argument('--num_return_sequences',
                        type=int,
                        default=1,
                        help="repeating times per image")
    parser.add_argument('--num_beams',
                        type=int,
                        default=1,
                        help="beam search width")
    parser.add_argument('--num_keep_best',
                        type=int,
                        default=1,
                        help="number of hypotheses to keep in beam search")
    parser.add_argument('--temperature',
                        type=float,
                        default=1,
                        help="temperature in softmax for sampling")
    parser.add_argument('--top_k',
                        type=int,
                        default=0,
                        help="filter distribution for sampling")
    parser.add_argument('--top_p',
                        type=float,
                        default=1,
                        help="filter distribution for sampling")
    parser.add_argument(
        '--repetition_penalty',
        type=int,
        default=1,
        help="repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)"
    )
    parser.add_argument('--length_penalty',
                        type=int,
                        default=1,
                        help="beam search length penalty")
    args = parser.parse_args()

    args.img_seq_length = int(args.img_size / args.patch_size) ** 2

    global logger

    # Setup CUDA, GPU & distributed training
    local_rank = ensure_init_process_group(local_rank=args.local_rank)
    args.local_rank = local_rank
    args.num_gpus = get_world_size()
    args.distributed = args.num_gpus > 1
    args.device = torch.device('cuda:' + args.gpu)
    synchronize()

    if args.do_train:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.output_dir = os.path.join(args.output_dir, timestamp)
        mkdir(args.output_dir)
    else:
        args.output_dir = args.eval_model_dir

    logger = setup_logger("captioning", args.output_dir, args.local_rank,
                          filename='train_log.txt' if args.do_train else 'test_log.txt')
    logger.warning("Device: %s, n_gpu: %s", args.device, args.num_gpus)
    set_seed(args.seed, args.num_gpus)
    args = restore_training_settings(args)

    # Save training parameters
    if args.do_train:
        with open(os.path.join(args.output_dir, "train_params.txt"), "w") as f:
            f.write(str(args))

    # Load pretrained model and tokenizer
    config_class, model_class, tokenizer_class = BertConfig, BertForImageCaptioning, BertTokenizer
    if args.do_train:
        assert args.model_name_or_path is not None
        config = config_class.from_pretrained(args.config_name if args.config_name else
                                              args.model_name_or_path, num_labels=args.num_labels, finetuning_task='image_captioning')

        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name
                                                    else args.model_name_or_path, do_lower_case=args.do_lower_case)

        config.image_size = args.img_size
        config.patch_size = args.patch_size
        config.img_seq_len = args.img_seq_length
        config.num_channels = args.num_channels
        config.hidden_size = args.img_embed_dim
        config.hidden_dropout_prob = args.drop_out
        config.loss_type = args.loss_type
        config.tie_weights = args.tie_weights
        config.freeze_embedding = args.freeze_embedding
        config.label_smoothing = args.label_smoothing
        config.drop_worst_ratio = args.drop_worst_ratio
        config.drop_worst_after = args.drop_worst_after
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool('.ckpt' in args.model_name_or_path),
            config=config)
    else:
        checkpoint = args.eval_model_dir
        assert op.isdir(checkpoint)
        config = config_class.from_pretrained(checkpoint)
        config.output_hidden_states = args.output_hidden_states
        tokenizer = tokenizer_class.from_pretrained(checkpoint)
        logger.info("Evaluate the following checkpoint: %s", checkpoint)
        model = model_class.from_pretrained(checkpoint, config=config)

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    val_transform = T.Compose([T.CenterCrop(args.img_size), T.ToTensor()])

    if args.do_train:
        train_transform = T.Compose(
            [T.RandomResizedCrop(args.img_size), T.ToTensor()])

        train_dataloader = make_data_loader(args=args,
                                            captions_file=args.train_json,
                                            concept_map_file=args.concepts_csv,
                                            concept_imgs_file=args.train_concepts_csv,
                                            tokenizer=tokenizer,
                                            transform=train_transform,
                                            is_distributed=args.distributed,
                                            is_train=True)
        val_dataloader = None
        if args.evaluate_during_training:
            val_dataloader = make_data_loader(args=args,
                                              captions_file=args.val_json,
                                              concept_map_file=args.concepts_csv,
                                              concept_imgs_file=args.val_concepts_csv,
                                              tokenizer=tokenizer,
                                              transform=val_transform,
                                              is_distributed=args.distributed,
                                              is_train=False)
        last_checkpoint = train(args, train_dataloader, val_dataloader, model,
                                tokenizer)

        # test the last checkpoint after training
        if args.do_test:
            logger.info("Evaluate on dataset: " + args.val_json)
            test_dataloader = make_data_loader(args=args,
                                               captions_file=args.val_json,
                                               concept_map_file=args.concepts_csv,
                                               concept_imgs_file=args.val_concepts_csv,
                                               tokenizer=tokenizer,
                                               transform=val_transform,
                                               is_distributed=args.distributed,
                                               is_train=False)
            evaluate(args, test_dataloader, model, tokenizer, last_checkpoint)

    # inference and evaluation
    elif args.do_test or args.do_eval:
        logger.info("Test on dataset: " + args.val_json)
        test_dataloader = make_data_loader(args=args,
                                           captions_file=args.val_json,
                                           concept_map_file=args.concepts_csv,
                                           concept_imgs_file=args.val_concepts_csv,
                                           tokenizer=tokenizer,
                                           transform=val_transform,
                                           is_distributed=args.distributed,
                                           is_train=False)

        if not args.do_eval:
            predict_file = get_predict_filename(checkpoint,
                                                test_dataloader.dataset.captions_file,
                                                args)
            test(args, test_dataloader, model, tokenizer, predict_file)
            logger.info("Prediction results saved to: {}".format(predict_file))
        else:
            evaluate_file = evaluate(args, test_dataloader, model, tokenizer,
                                     checkpoint)
            logger.info(
                "Evaluation results saved to: {}".format(evaluate_file))


if __name__ == "__main__":
    main()
