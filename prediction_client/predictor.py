import finance_client as fc
import pandas as pd

from . import models
from .models.base_model import ModelWrapper


class Client:
    def __init__(self, finance_client: fc.Client, model: ModelWrapper, std_processes: list) -> None:
        self.finance_client = finance_client
        self.model = model
        self.std_processes = std_processes
        required_processes = model.get_required_process()
        for process_type in required_processes:
            if not self.has_process(process_type, std_processes):
                raise Exception(f"finance client require {process_type} process")

    def has_process(self, process_type, processes):
        for process in processes:
            if isinstance(process, process_type):
                return True
        return False

    def get_ohlc(self, idc_process=None):
        """get ohlc data with predictions.
        Standalization processes should be specified in advance on this client.
        Technical indicators should be specidied in finance client in advance.

        Args:
            idc_process (list[fc.fprocess.ProcessBase], optional): After predictions is caliculated, apply processes if specified. Defaults to None.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: observations and predictions
        """
        required_length = [0]
        if self.std_processes is not None:
            for process in self.std_processes:
                required_length.append(process.get_minimum_required_length())
        required_length = max(required_length)

        ohlc_df = self.finance_client._get_ohlc_from_client(
            length=self.model.observation_length + required_length,
            symbols=self.finance_client.symbols,
            frame=self.finance_client.frame,
            columns=self.finance_client.out_ohlc_columns,
            index=None,  # use latest index only
            grouped_by_symbol=grouped_by_symbol,
        )
        kwargs = {}
        observations, args = self.model.get_arguments(ohlc_df, self.std_processes)
        observations = observations.iloc[-self.model.observation_length :]
        kwargs.update(args)
        predictions = self.model.predict(observations=observations, **kwargs)
        predictions = self.model.revert_predictions(ohlc_df, predictions, self.std_processes)
        ohlc_df = pd.concat([ohlc_df, predictions], axis=0)
        if idc_process is not None:
            for process in idc_process:
                ohlc_df = process.run(ohlc_df)
        observations = ohlc_df.loc[observations.index]
        predictions = ohlc_df.loc[predictions.index]
        return observations, predictions

    def __getattr__(self, name):
        if hasattr(self.finance_client, name) and callable(getattr(self.finance_client, name)):
            return getattr(self.finance_client, name)
        else:
            raise AttributeError(f"'{name}' is not defined")
