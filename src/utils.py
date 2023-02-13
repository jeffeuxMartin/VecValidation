#!/usr/bin/env python3
from pathlib import Path

from tqdm import tqdm

import tgt


SAMPLE_RATE = 16_000


def datasetGatherer(dataset, index=None):
    """ 拿前段資料就好 """
    if index is not None:
        output = []
        for idx, item in tqdm(
            enumerate(dataset), 
            desc="Gathering data..."):
            if idx >= index: break
            output.append(item)
        return output
    else:
        return [i for i in tqdm(
            dataset, 
            desc="Gathering data...")]

def parseTgt(tgtfile):
    """ 剖析 forced alignment 資料 """
    tgtobj = tgt.read_textgrid(tgtfile)
    [word_tiers] = tgtobj.get_tiers_by_name("words")
    if tgtobj.start_time == 0.0:
        return dict(
            duration=tgtobj.end_time,
            word_tiers=[
                (
                    w.start_time,
                    w.end_time,
                    w.text,
                )
                for w in word_tiers
            ],
        )
    else:
        return dict(
            duration=
                tgtobj.end_time - tgtobj.start_time,
            word_tiers=[
                (
                    w.start_time - tgtobj.start_time,
                    w.end_time - tgtobj.start_time,
                    w.text,
                )
                for w in word_tiers
            ],
            start_time=tgtobj.start_time,
        )

def second_to_vecidx(time: float) -> int:
    """ 轉換秒數到 vector index """
    bias = 80
    sr = 16_000
    stride = 0.020
    
    return max(0, 
        int((time * sr - bias) // (sr * stride)))

def second_to_sample(time: float) -> int:
    """ 轉換秒數到 sample index """
    sr = 16_000
    
    return time * sr

def sample_to_vecidx(sample_idx: int) -> int:
    """ 轉換 sample index 到 vector index """
    bias = 80
    sr = 16_000
    stride = 0.020

    return int((sample_idx - bias) // (sr * stride))

def add_to_vocab(coll, word, vec, utt_idx, word_idx):
    """ 加入到 vocab，也許寫在 coll? [TODO!] """
    if word not in coll:
        coll[word] = []
    coll[word].append(
        (vec, utt_idx, word_idx))

def path_to_info(pathname: Path) -> tuple:
    """ 從路徑找到 infos """
    return *(int(num) 
               for num in pathname.stem.split('-')),

def collate_fn_factory(
    TGTs, 
    device, 
    SAMPLE_RATE=SAMPLE_RATE):
    """ 建造 dataloader 用的，應該比較快 """
    def collate_fn(batch):
        waves, srs, texts, *infos = zip(*batch)
        waves = [w.squeeze(0).to(device) 
                 for w in waves]
        assert all(i == SAMPLE_RATE for i in srs)
        *infos, = zip(*infos)
        tgtins = [parseTgt(TGTs[tgtfile]) 
            for tgtfile in infos]
        return (waves, tgtins, texts, infos)
    return collate_fn


def aggr(vecseq):
    """ Aggregation of vecs in speech sequence. """
    if len(vecseq) < 1:
        raise ValueError
    return vecseq.mean(0)
    
def different_occurrance_aggr(vecs):
    """ 
    aggregation of different occurance vectors. 
    """
    if len(vecs) < 1:
        raise ValueError
    return vecs.mean(0)
