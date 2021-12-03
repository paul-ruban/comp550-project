import argparse
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelWithLMHead
from src.aug.utils import hf_masked_encode, hf_reconstruction_prob_tok, fill_batch


def gen_neighborhood(
    shard=0,
    num_shards=1,
    seed=None,
    model="bert-base-uncased",
    tokenizer=None,
    in_file=None,
    label_file=None,
    output_path=None,
    noise_prob=0.15,
    random_token_prob=0.1,
    leave_unmasked_prob=0.1,
    batch=8,
    num_samples=4,
    max_tries=10,
    min_len=4,
    max_len=512,
    topk=-1,
):
    """
    batch : # examples generated per input example
    """

    # initialize seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # load model and tokenizer
    r_model = AutoModelWithLMHead.from_pretrained(model)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    r_model.eval()
    if torch.cuda.is_available():
        r_model.cuda()

    # remove unused vocab and special ids from sampling
    softmax_mask = np.full(len(tokenizer.vocab), False)
    softmax_mask[tokenizer.all_special_ids] = True
    for k, v in tokenizer.vocab.items():
        if "[unused" in k:
            softmax_mask[v] = True

    # load the inputs and labels
    lines = [tuple(s.strip().split("\t")) for s in open(in_file).readlines()]
    num_lines = len(lines)
    lines = [[[s] for s in s_list] for s_list in list(zip(*lines))]

    # load label file if it exists
    if label_file:
        labels = [s.strip() for s in open(label_file).readlines()]
        output_labels = True
    else:
        labels = [0] * num_lines
        output_labels = False

    # shard the input and labels
    if num_shards > 0:
        shard_start = (int(num_lines / num_shards) + 1) * shard
        shard_end = (int(num_lines / num_shards) + 1) * (shard + 1)
        lines = [s_list[shard_start:shard_end] for s_list in lines]
        labels = labels[shard_start:shard_end]

    # open output files
    if num_shards != 1:
        output_text_path = os.path.join(output_path + "_" + str(shard))
        s_rec_file = open(output_text_path, "w")
        if output_labels:
            ouput_label_path = os.path.join(
                output_path + "_" + str(shard) + ".label"
            )
            l_rec_file = open(ouput_label_path, "w")
    else:
        output_text_path = os.path.join(output_path)
        s_rec_file = open(output_text_path, "w")
        if output_labels:
            ouput_label_path = os.path.join(output_path + ".label")
            l_rec_file = open(ouput_label_path, "w")

    # sentences and labels to process
    sents = []
    l = []

    # number sentences generated
    num_gen = []

    # sentence index to noise from
    gen_index = []

    # number of tries generating a new sentence
    num_tries = []

    # next sentence index to draw from
    next_sent = 0

    # generated_sentences and labels, keep these for postprocessing 
    # (to conserve the order )

    sents, l, next_sent, num_gen, num_tries, gen_index = fill_batch(
        batch,
        min_len,
        max_len,
        tokenizer,
        sents,
        l,
        lines,
        labels,
        next_sent,
        num_gen,
        num_tries,
        gen_index,
    )

    # main augmentation loop
    while sents != []:

        # remove any sentences that are done generating and dump to file
        for i in range(len(num_gen))[::-1]:
            if num_gen[i] == num_samples or num_tries[i] > max_tries:

                # get sent info
                gen_sents = sents.pop(i)
                num_gen.pop(i)
                gen_index.pop(i)
                label = l.pop(i)

                # write generated sentences
                # don't dump the original sentence!
                for sg in gen_sents[1:]:  # the dump is done here!
                    s_rec_file.write("\t".join([repr(val)[1:-1] for val in sg]) + "\n")
                    if output_labels:
                        l_rec_file.write(label + "\n")

        # fill batch
        sents, l, next_sent, num_gen, num_tries, gen_index = fill_batch(
            batch,
            min_len,
            max_len,
            tokenizer,
            sents,
            l,
            lines,
            labels,
            next_sent,
            num_gen,
            num_tries,
            gen_index,
        )

        # break if done dumping
        if len(sents) == 0:
            break

        # build batch
        toks = []
        masks = []

        for i in range(len(gen_index)):
            s = sents[i][gen_index[i]]
            tok, mask = hf_masked_encode(
                tokenizer,
                *s,
                noise_prob=noise_prob,
                random_token_prob=random_token_prob,
                leave_unmasked_prob=leave_unmasked_prob,
            )
            toks.append(tok)
            masks.append(mask)

        # pad up to max len input
        max_len = max([len(tok) for tok in toks])
        pad_tok = tokenizer.pad_token_id

        # Pad at the end with the padding token which is 0
        toks = [
            F.pad(tok, (0, max_len - len(tok)), "constant", pad_tok) for tok in toks
        ]
        masks = [
            F.pad(mask, (0, max_len - len(mask)), "constant", pad_tok) for mask in masks
        ]
        toks = torch.stack(toks)
        masks = torch.stack(masks)

        # load to GPU if available
        if torch.cuda.is_available():
            toks = toks.cuda()
            masks = masks.cuda()

        # predict reconstruction
        rec, rec_masks = hf_reconstruction_prob_tok(
            toks,
            masks,
            tokenizer,
            r_model,
            softmax_mask,
            reconstruct=True,
            topk=topk,
        )

        # decode reconstructions and append to lists
        for i in range(len(rec)):
            rec_work = rec[i].cpu().tolist()
            s_rec = [
                s.strip()
                for s in tokenizer.decode(  # this is where you give it the new perturbed data
                    [val for val in rec_work if val != tokenizer.pad_token_id][
                        1:-1
                    ]  # 1:-1 gets rid of <s>
                ).split(
                    tokenizer.sep_token
                )
            ]
            s_rec = tuple(s_rec)

            # check if identical reconstruction or empty
            if s_rec not in sents[i] and "" not in s_rec:
                sents[i].append(s_rec)
                num_gen[i] += 1
                num_tries[i] = 0
                gen_index[i] = 0

            # otherwise try next sentence
            else:
                num_tries[i] += 1
                gen_index[i] += 1
                if gen_index[i] == len(sents[i]):
                    gen_index[i] = 0

        # clean up tensors
        del toks
        del masks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--shard",
        type=int,
        default=0,
        help="Shard of input to process. Output filename "
        "will have _${shard} appended.",
    )

    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of shards to shard input file with.",
    )

    parser.add_argument(
        "--seed", type=int, help="Random seed to use for reconstruction and noising."
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="bert-base-uncased",
        help="Name of HuggingFace BERT model to use for reconstruction,"
        " or filepath to local model directory.",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Name of HuggingFace tokenizer to use for vocabulary"
        " or filepath to local tokenizer. If None, uses the same"
        " as model.",
    )

    parser.add_argument(
        "-i",
        "--in-file",
        type=str,
        help="Path of input text file for augmentation."
        " Inputs should be separated by newlines with tabs indicating"
        " BERT <SEP> tokens.",
    )

    parser.add_argument(
        "-l",
        "--label-file",
        type=str,
        default=None,
        help="Path of input label file for augmentation if using "
        " label preservation.",
    )

    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        help="Path for output files, including augmentations and" " preserved labels.",
    )

    parser.add_argument(
        "-p",
        "--noise-prob",
        type=float,
        default=0.15,
        help="Probability for selecting a token for noising."
        " Selected tokens are then masked, randomly replaced,"
        " or left the same.",
    )

    parser.add_argument(
        "-r",
        "--random-token-prob",
        type=float,
        default=0.1,
        help="Probability of a selected token being replaced"
        " randomly from the vocabulary.",
    )

    parser.add_argument(
        "-u",
        "--leave-unmasked-prob",
        type=float,
        default=0.1,
        help="Probability of a selected token being left" " unmasked and unchanged.",
    )

    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        default=8,
        help="Batch size of inputs to reconstruction model.",
    )

    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=4,
        help="Number of augmented samples to generate for each" " input example.",
    )

    parser.add_argument(
        "-t",
        "--max-tries",
        type=int,
        default=10,
        help="Number of tries to generate a unique sample" " before giving up.",
    )

    parser.add_argument(
        "--min-len", type=int, default=4, help="Minimum length input for augmentation."
    )

    parser.add_argument(
        "--max-len",
        type=int,
        default=512,
        help="Maximum length input for augmentation.",
    )

    parser.add_argument(
        "--topk",
        "-k",
        type=int,
        default=-1,
        help="Top k to use for sampling reconstructed tokens from"
        " the BERT model. -1 indicates unrestricted sampling.",
    )

    args = parser.parse_args()

    if args.shard >= args.num_shards:
        raise Exception(
            "Shard number {} is too large for the number"
            " of shards {}".format(args.shard, args.num_shards)
        )

    if not args.tokenizer:
        args.tokenizer = args.model

    gen_neighborhood(
        shard=args.shard,
        num_shards=args.num_shards,
        seed=args.seed,
        model=args.model,
        tokenizer=args.tokenizer,
        in_file=args.in_file,
        label_file=args.label_file,
        output_path=args.output_path,
        noise_prob=args.noise_prob,
        random_token_prob=args.random_token_prob,
        leave_unmasked_prob=args.leave_unmasked_prob,
        batch=args.batch,
        num_samples=args.num_samples,
        max_tries=args.max_tries,
        min_len=args.min_len,
        max_len=args.max_len,
        topk=args.topk,
    )
