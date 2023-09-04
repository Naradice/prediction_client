import finance_client as fc

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

    def get_ohlc(
        self,
        symbols=None,
        frame: int = None,
        columns: list = None,
        index=None,
        grouped_by_symbol=True,
    ):
        ohlc_df = self.finance_client._get_ohlc_from_client(
            length=self.model.observation_length,
            symbols=self.finance_client.symbols,
            frame=frame,
            columns=columns,
            index=index,
            grouped_by_symbol=grouped_by_symbol,
        )
        kwargs = {}
        observations, args = self.model.get_arguments(ohlc_df, self.std_processes)
        kwargs.update(args)
        predictions = self.model.predict(observations=observations, **kwargs)
        predictions = self.model.revert_predictions(observations, predictions, self.std_processes)
        return observations, predictions

    def __getattr__(self, name):
        if hasattr(self.finance_client, name) and callable(getattr(self.finance_client, name)):
            return getattr(self.finance_client, name)
        else:
            raise AttributeError(f"'{name}' is not defined")
