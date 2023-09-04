import pandas as pd
import torch
from finance_client import fprocess


class ModelWrapper:
    def __init__(self, torch_model, observation_length, prediction_length, device=None) -> None:
        self.torch_model = torch_model
        self.observation_length = observation_length
        self.prediction_length = prediction_length
        self.device = device
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def convert_df_to_tensor(self, df, dtype=torch.float64):
        return torch.tensor(df.values, device=self.device, dtype=dtype)

    def convert_tensor_to_df(self, tensor: torch.tensor, index, columns):
        np_values = tensor.cpu().detach().numpy()
        return pd.DataFrame(np_values, index=index, columns=columns)

    def revert_predictions(self, observations: pd.DataFrame, predictions: pd.DataFrame, std_processes: list):
        # most process doesn't work as arguments are required.
        for p_index in range(len(std_processes)):
            r_index = len(std_processes) - 1 - p_index
            process = std_processes[r_index]
            if hasattr(process, "revert_params"):
                # print(f"currently: {r_data[0, 0]}")
                params = process.revert_params
                if len(params) == 1:
                    predictions = process.revert(predictions)
                else:
                    params = {}
                    if isinstance(process, fprocess.MinMaxPreProcess):
                        predictions = process.revert(predictions)
                    elif isinstance(process, fprocess.SimpleColumnDiffPreProcess):
                        close_column = process.base_column
                        if p_index > 0:
                            processes = std_processes[:p_index]
                            required_length = [1]
                            base_processes = []
                            for base_process in processes:
                                if close_column in base_process.columns:
                                    base_processes.append(base_process)
                                    required_length.append(base_process.get_minimum_required_length())
                            if len(base_processes) > 0:
                                raise Exception("Not implemented yet")
                        base_values = observations[close_column].iloc[-1:]
                        predictions = process.revert(predictions, base_value=base_values)
                    elif process.kinds == fprocess.DiffPreProcess.kinds:
                        target_columns = process.columns
                        if r_index > 0:
                            processes = std_processes[:r_index]
                            required_length = [process.get_minimum_required_length()]
                            base_processes = []
                            for base_process in processes:
                                if len(set(target_columns) & set(base_process.columns)) > 0:
                                    base_processes.append(base_process)
                                    required_length.append(base_process.get_minimum_required_length())
                            if len(base_processes) > 0:
                                required_length = max(required_length)
                                target_data = observations[target_columns].iloc[-required_length:]
                                for base_process in base_processes:
                                    target_data = base_process(target_data)
                                base_values = target_data.iloc[-1:]
                            else:
                                base_values = observations[target_columns].iloc[-1:]
                        else:
                            base_values = observations[target_columns].iloc[-1:]
                        predictions = process.revert(predictions, base_values=base_values)
                    else:
                        raise Exception(f"Not implemented: {process.kinds}")
        return predictions

    def get_required_process(self):
        return []

    def get_arguments(self, observations, finance_client):
        return observations, {}

    def predict(self, observations, **kwargs):
        pass
