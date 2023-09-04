import finance_client as fc
import torch
from torch import nn
import pandas as pd

from .base_model import ModelWrapper


class Transformer(ModelWrapper):
    def __init__(self, torch_model, observation_length, prediction_length, device=None) -> None:
        super().__init__(torch_model, observation_length, prediction_length, device=device)

    def get_required_process(self):
        return [fc.fprocess.TimeProcess]

    def get_arguments(self, observations, std_processes):
        kwargs = {}
        for process in std_processes:
            observations = process.run(observations)
            if isinstance(process, fc.fprocess.TimeProcess):
                freq = process.freq
                delta = pd.Timedelta(minutes=freq)
                time_column = process.time_column
                if time_column == "index":
                    last_time = observations.index[-1]
                else:
                    last_time = observations[time_column].iloc[-1]
                pre_index = [last_time + delta * i for i in range(self.prediction_length + 1)]
                pre_index = pd.DatetimeIndex(pre_index)
                pre_df = pd.DataFrame(index=pre_index)
                pre_df = process.run(pre_df)
                kwargs = {"tgt_time_srs": pre_df[time_column]}
                self.positional_index = time_column
        return observations, kwargs

    def predict(self, observations: pd.DataFrame, tgt_time_srs: pd.Series):
        """_summary_

        Args:
            observations (pd.DataFrame):
        """

        src_time_srs = observations[self.positional_index]
        columns_to_keep = [col for col in observations.columns if col != self.positional_index]
        src_df = observations[columns_to_keep]
        feature_size = len(columns_to_keep)

        src = self.convert_df_to_tensor(src_df, dtype=torch.float)
        src = src.unsqueeze(-1).transpose(1, 2)
        src_time = self.convert_df_to_tensor(src_time_srs, dtype=torch.int)
        src_time = src_time.unsqueeze(-1)

        tgt = torch.zeros(1, src.size(1), feature_size, device=self.device)
        tgt[0, :, :] = src[-1, :, :]

        tgt_time_tensor = self.convert_df_to_tensor(tgt_time_srs, dtype=torch.int)
        tgt_time_tensor = tgt_time_tensor.unsqueeze(-1)

        print(src.shape, src_time.shape, tgt.shape, tgt_time_tensor.shape)

        while tgt.size(0) <= self.prediction_length:
            current_length = tgt.size(0)
            tgt_time = tgt_time_tensor[:current_length, :]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(self.device)
            out = self.torch_model(src, src_time, tgt=tgt, tgt_time=tgt_time, mask_tgt=tgt_mask)
            tgt = torch.cat([tgt, out[-1:]], dim=0)

        pre_df = self.convert_tensor_to_df(tgt[:, 0, :], index=tgt_time_srs.index, columns=columns_to_keep)

        return pre_df
