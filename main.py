#!/usr/bin/env python3
from pathlib import Path

import numpy as np, pandas as pd
from tqdm import tqdm

import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchaudio.datasets import LIBRISPEECH as LS

import s3prl.hub as hub

from src.utils import (
    second_to_vecidx as toIdx,
    add_to_vocab,
    path_to_info,
    collate_fn_factory,
)
from src.utils import aggr, different_occurrance_aggr


DATASET_ROOT = Path("/storage/LabJob/Projects/Data")
SPLIT = "train-clean-100"
BATCH_SIZE = 16  # 10 ~ 32
device = ("cuda" 
    if torch.cuda.is_available() else "cpu")


def initModel():
    """初始化模型 """
    model = hub.hubert().to(device).eval()
    return model

def initData():
    """ 初始化資料集，以 dataloader 形式輸出 """
    # TODO: add split
    TGTs: dict = {
        path_to_info(pathname): pathname 
        for pathname in DATASET_ROOT.glob(
            "LibriSpeech_Textgrids/"
           f"{SPLIT}/*/*.TextGrid")}
    dst = LS(DATASET_ROOT, SPLIT, "LibriSpeech")
    dldr = DataLoader(dst, 
        batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn_factory(TGTs, device))
    return dldr

def collectVectors(model, dldr):
    """ 把向量抓出來 """
    coll = {}

    with torch.no_grad():
        for batch in tqdm(dldr, desc='取向量'):
            waves, tgtins, texts, infos = batch
            outs = model(wavs=waves)
            outs = outs['hidden_states'][-1]
            # print(outs.shape)

            for utt_idx, (utt_tgt, out) in (
                    enumerate(zip(tgtins, outs))):
                for word_idx, (st, en, word) in (
                    enumerate(utt_tgt['word_tiers'])):
                    vecseq_word = (
                       out[toIdx(st):toIdx(en)]
                          .detach().to('cpu').numpy())
                    vec_word = aggr(vecseq_word)
                    add_to_vocab(coll, word, vec_word,
                                utt_idx, word_idx)


    """
    Usage:
    vecs, uttidcs, wordidcs = zip(*coll['chapter'])
    """

    for w in tqdm(coll, desc='整理向量'):
        vecs, uttidcs, wordidcs = zip(*coll[w])
        coll[w] = (np.array(vecs), uttidcs, wordidcs)
        
    for w in tqdm(coll, desc='合併向量'):
        # TODO: 可以合併到上面那步！
        vecs, uttidcs, wordidcs = coll[w]
        coll[w] = (
            different_occurrance_aggr(vecs), 
            uttidcs, 
            wordidcs)

    return coll


def main():
    model, dldr = initModel(), initData()
    coll = collectVectors(model, dldr)
    return coll, model, dldr


if __name__ == "__main__":
    coll, model, dldr = main()    
