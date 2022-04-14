# import fastseq
from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    EarlyStoppingCallback,
    GPT2Tokenizer,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from scst_trainer import CustomTrainer
import datetime
import argparse
import os
from dataset import Dataset
import torchvision.transforms as T
from transformers.data.data_collator import default_data_collator
import shutil

# Reproducibility
import torch

torch.manual_seed(42)
torch.backends.cudnn.benchmark = True
import random

random.seed(42)
import numpy as np

np.random.seed(0)


def _create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_path = os.path.join(folder, timestamp)

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    return timestamp, results_path


def count_parameters(model):
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad), sum(p.numel() for p in model.parameters()
                                        if not p.requires_grad)


def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    outputs = token_ids_0 + [self.eos_token_id]
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments to run the script.")

    # Processing parameters
    parser.add_argument(
        "--gpu",
        type=str,
        default="1",
        help="Which gpus to use in CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument("--num_workers",
                        type=int,
                        default=4,
                        help="Number of workers for dataloader.")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        action="store_true",
        help="Use 16-bit floating-point precision.",
    )
    parser.add_argument(
        "--no-fp16",
        dest="fp16",
        action="store_false",
        help="Use 16-bit floating-point precision.",
    )
    parser.set_defaults(fp16=False)

    # Directories and paths
    parser.add_argument(
        "--logdir",
        type=str,
        default=
        "/media/TOSHIBA6T/ICC2022/captioning/baseline_without_concepts/",
        help="Directory where logs and models are to be stored.",
    )
    # Model
    parser.add_argument(
        "--encoder",
        type=str,
        default="facebook/deit-tiny-patch16-224",
        choices=[
            "microsoft/beit-base-patch16-224-pt22k-ft22k",
            "google/vit-base-patch16-224", "facebook/deit-tiny-patch16-224"
        ],
        help="Encoder model to load.",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        default="distilgpt2",
        choices=[
            "bert-base-uncased",
            "gpt2",
            "distilgpt2",
        ],
        help="Decoder model to load.",
    )
    parser.add_argument(
        "--model_max_length",
        type=int,
        default=100,
        help="Max length of inputs.",
    )

    # Training
    parser.add_argument("--scst",
                        action="store_true",
                        help="Do Self-Critical Sequence Training.")
    parser.add_argument("--ckpt",
                        type=str,
                        default=None,
                        help="Load model from this checkpoint.")
    parser.add_argument("--resume",
                        action="store_true",
                        help="Resume training from checkpoint.")
    parser.add_argument("--load_pretrained",
                        action="store_true",
                        help="Load pre trained model for fine tuning.")

    parser.add_argument("--epochs",
                        type=int,
                        default=20,
                        help="Number of epochs.")
    parser.add_argument(
        "--bs",
        type=int,
        default=32,
        help="Batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=1e-6,
        help="Learning rate.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience",
    )

    # Beam Search
    parser.add_argument(
        "--gen_max_length",
        type=int,
        default=150,
        help="Max length of generated sequence.",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=3,
        help="Min length of generated sequence.",
    )

    args = parser.parse_args()
    if args.resume or args.load_pretrained:
        assert args.resume != args.load_pretrained, "Options resume and load_pretrained are mutually exclusive. Please choose only one."
    if args.ckpt:
        assert (
            args.resume or args.load_pretrained
        ), "When resuming training or loading pretrained model, you need to provide a checkpoint. When a checkpoint is provided, you need to select either the resume or the load_pretrained options."
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device("cuda:" +
                          args.gpu if torch.cuda.is_available() else "cpu")

    timestamp, path = _create_folder(args.logdir)

    with open(os.path.join(path, "train_params.txt"), "w") as f:
        f.write(str(args))

    tokenizer_path = args.ckpt if args.ckpt else args.decoder
    feature_extractor_path = args.ckpt if args.ckpt else args.encoder

    if "gpt2" in args.decoder:
        GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)

        if not args.ckpt:
            special_tokens_dict = {
                'additional_special_tokens': ["<|startoftext|>", "<|pad|>"]
            }
            tokenizer.add_special_tokens(special_tokens_dict)
            tokenizer.bos_token = "<|startoftext|>"
            tokenizer.pad_token = "<|pad|>"
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if not args.ckpt:
            # user BERT's CLS token as BOS and SEP as EOS
            tokenizer.bos_token = tokenizer.cls_token
            tokenizer.eos_token = tokenizer.sep_token

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        feature_extractor_path)
    size = feature_extractor.size

    # Data
    preprocess = T.Compose([T.RandomResizedCrop(size), T.ToTensor()])
    tr_dtset = Dataset(
        "/media/TOSHIBA6T/ICC2022/dataset/train_resized/",
        "/media/TOSHIBA6T/ICC2022/dataset/caption_prediction_train_coco.json",
        tokenizer,
        args.model_max_length,
        feature_extractor,
        transform=preprocess,
    )
    val_dtset = Dataset(
        "/media/TOSHIBA6T/ICC2022/dataset/valid_resized/",
        "/media/TOSHIBA6T/ICC2022/dataset/caption_prediction_valid_coco.json",
        tokenizer,
        args.model_max_length,
        feature_extractor,
        transform=preprocess,
    )

    # Model
    if args.ckpt:
        model = VisionEncoderDecoderModel.from_pretrained(args.ckpt)
        model.encoder.pooler = None  # TODO REMOVE
        print("Model weights loaded from " + args.ckpt)
    else:
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            args.encoder,
            args.decoder,
            encoder_add_pooling_layer=False,
        )
        model.decoder.resize_token_embeddings(len(tokenizer))
        del model.config.encoder.label2id
        del model.config.encoder.id2label

        # set special tokens
        model.config.decoder_start_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

        # parameters for generation
        model.config.vocab_size = model.config.decoder.vocab_size
        model.config.max_length = args.gen_max_length
        model.config.min_length = args.min_length
        model.config.early_stopping = True

    model.train()

    model.config.to_json_file(os.path.join(path, "conf.json"))

    trainable_params, non_trainable_params = count_parameters(model.encoder)
    print("Encoder params\n\t Trainable: %d \n\t Non trainable: %d\n" %
          (trainable_params, non_trainable_params))

    trainable_params, non_trainable_params = count_parameters(model.decoder)
    print("Decoder params\n\t Trainable: %d \n\t Non trainable: %d\n" %
          (trainable_params, non_trainable_params))

    # Trainer
    training_args = Seq2SeqTrainingArguments(
        output_dir=path,
        evaluation_strategy="epoch",
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        learning_rate=args.lr,
        weight_decay=args.decay,
        num_train_epochs=args.epochs,
        logging_dir=os.path.join(path, "runs"),
        logging_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        fp16=args.fp16,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=False,
        load_best_model_at_end=True,
        log_level="debug",
    )

    if (args.scst):
        # for param in model.encoder.parameters():
        #     param.requires_grad = False
        trainer = CustomTrainer(
            model=model,
            tokenizer=
            tokenizer,  # to ensure it is saved (useful later for loading during generation)
            data_collator=
            default_data_collator,  # when a tokenizer is passed the data_collator defaults to DataCollatorWithPadding, which throws an error, so we simply override this behaviour by passing the default collator
            args=training_args,
            train_dataset=tr_dtset,
            eval_dataset=val_dtset,
            callbacks=[EarlyStoppingCallback(args.patience)],
        )
        trainer.init_loss(model=model)
    else:
        trainer = Seq2SeqTrainer(
            model=model,
            tokenizer=
            tokenizer,  # to ensure it is saved (useful later for loading during generation)
            data_collator=
            default_data_collator,  # when a tokenizer is passed the data_collator defaults to DataCollatorWithPadding, which throws an error, so we simply override this behaviour by passing the default collator
            args=training_args,
            train_dataset=tr_dtset,
            eval_dataset=val_dtset,
            callbacks=[EarlyStoppingCallback(args.patience)],
        )
    trainer.train(resume_from_checkpoint=args.ckpt if args.resume else False)

    # save best model
    print("Saving best model...")
    best_model_dir = trainer.state.best_model_checkpoint
    shutil.copytree(best_model_dir, os.path.join(path, "checkpoint-best"))
    feature_extractor.save_pretrained(os.path.join(path, "checkpoint-best"))
