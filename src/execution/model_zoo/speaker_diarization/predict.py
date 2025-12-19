import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from src.execution.model_zoo.speaker_diarization.cluster import \
    OptimizedAgglomerativeClustering
from src.execution.model_zoo.speaker_diarization.dataset import BaseLoad
from src.execution.model_zoo.speaker_diarization.model import Encoder
from src.execution.model_zoo.speaker_diarization.utils import get_timestamp, zcr_vad
from torch import onnx


class BasePredictor(BaseLoad):
    def __init__(self, config_path, max_frame, hop):
        config = torch.load(config_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        super().__init__(config.sr, config.n_mfcc)
        self.ndim = config.ndim
        self.max_frame = max_frame
        self.hop = hop
        
    @staticmethod
    def _plot_diarization(y, spans, speakers):
        c = y[0].cpu().numpy().copy()
        for (start, end), speaker in zip(spans, speakers):
            c[start:end] = speaker

        plt.figure(figsize=(15, 2))
        plt.plot(y[0], "k-")
        for idx, speaker in enumerate(set(speakers)):
            plt.fill_between(range(len(c)), -1, 1, where=(c==speaker), alpha=0.5, label=f"speaker_{speaker}")
        plt.legend(loc="upper center", ncol=idx+1, bbox_to_anchor=(0.5, -0.25))
        
        
class PyTorchPredictor(BasePredictor):
    def __init__(self, config_path, model_path, max_frame=45, hop=3):
        super().__init__(config_path, max_frame, hop)
        
        weight = torch.load(model_path, map_location="cpu")
        self.model = Encoder(self.ndim).to(self.device)
        self.model.load_state_dict(weight)
        self.model.eval()
    
    def predict(self, path, plot=False):        
        y = self._load(path, mfcc=False)
        activity = zcr_vad(y)
        spans = get_timestamp(activity)
        
        embed = []
        new_spans = []
        for span in spans:
            obtained_embed = self._encode_segment(y, span)
            if obtained_embed is not None:
                embed.append(obtained_embed)
                new_spans.append(span)
        
        print(f"Detected {len(embed)} speakers.")
        
        # 检查嵌入向量的数量
        if len(embed) == 0:
            # 如果没有有效的嵌入向量，返回空结果
            timestamp = np.array([]) / self.sr
            speakers = np.array([])
        elif len(embed) == 1:
            # 只有一个嵌入向量，直接赋予标签0
            speakers = np.array([0])
            timestamp = np.array(new_spans) / self.sr
        elif len(embed) == 2:
            # 有两个嵌入向量，直接进行判断
            speakers = np.array([0, 0])
            timestamp = np.array(new_spans) / self.sr
        else:
            # 有多个嵌入向量，进行聚类
            embed = torch.cat(embed).cpu().numpy()
            speakers = OptimizedAgglomerativeClustering().fit_predict(embed)
            timestamp = np.array(new_spans) / self.sr

        if plot and len(spans) > 0:
            self._plot_diarization(y, new_spans, speakers)
            
        return timestamp, speakers
    
    def _encode_segment(self, y, span):
        start, end = span
        mfcc = self._mfcc(y[:, start:end]).to(self.device)
        if mfcc.shape[2] < self.max_frame:
            return None
        mfcc = mfcc.unfold(2, self.max_frame, self.hop).permute(2, 0, 1, 3)
        with torch.no_grad():
            embed = self.model(mfcc).mean(0, keepdims=True)
        return embed
        
    def to_onnx(self, outdir="model/openvino"):
        os.makedirs(outdir, exist_ok=True)
        mfcc = torch.rand(1, 1, self.n_mfcc, self.max_frame).to(self.device)
        onnx.export(self.model, mfcc, f"{outdir}/diarization.onnx", input_names=["input"], output_names=["output"])
        print(f"model is exported as {outdir}/diarization.onnx")     
        
