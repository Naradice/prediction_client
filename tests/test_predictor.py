import json
import os
import sys
import unittest
from logging import config, getLogger

import finance_client as fc

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(module_path)
sys.path.append(module_path)

import prediction_client as pc
from basic_transformer_model import Seq2SeqTransformer

logger = getLogger("prediction_client.test")


class TestPrediction(unittest.TestCase):
    def create_basic_transformer(self, feature_size, time_size, nhead, dim_feedforward, num_encoder_layers, num_decoder_layers, device="cpu"):
        dropout = 0.1

        return Seq2SeqTransformer(
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            feature_size=feature_size,
            time_size=time_size,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            nhead=nhead,
        ).to(device)

    def test_create_ts_basic_model(self):
        torch_model = self.create_basic_transformer(
            feature_size=4, time_size=24 * 7 * 2, nhead=1, dim_feedforward=10, num_encoder_layers=2, num_decoder_layers=2
        )
        observation_length = 60
        prediction_length = 10

        model = pc.models.Transformer(torch_model, observation_length, prediction_length)

    def common_prediction(self, idc_process=None):
        torch_model = self.create_basic_transformer(
            feature_size=4, time_size=24 * 7 * 2, nhead=1, dim_feedforward=10, num_encoder_layers=2, num_decoder_layers=2, device="cuda"
        )
        observation_length = 60
        prediction_length = 10

        model = pc.models.Transformer(torch_model, observation_length, prediction_length)

        with open(f"{os.path.dirname(__file__)}/preprocess.json", mode="r") as fp:
            process_dict = json.load(fp)
        processes = fc.fprocess.load_preprocess(process_dict)
        std_processes = [
            *processes,
            fc.fprocess.WeeklyIDProcess(freq=30, time_column="index"),
        ]
        client = fc.CSVClient("L:\\data\\fx\\HistData_USDJPY_30min.csv", columns=["open", "high", "low", "close"])
        p_client = pc.Client(client, model, std_processes=std_processes)
        obs, pre = p_client.get_ohlc(idc_process=idc_process)
        return obs, pre

    def test_prediction_with_observation(self):
        obs, pre = self.common_prediction()
        print(pre)

    def test_prediction_with_indicators(self):
        idc_process = [fc.fprocess.MACDProcess(target_column="close")]
        obs, pre = self.common_prediction(idc_process=idc_process)
        print(pre)


if __name__ == "__main__":
    unittest.main()
