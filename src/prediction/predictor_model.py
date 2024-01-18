import os
import warnings
import joblib
import numpy as np
import pandas as pd
from schema.data_schema import ForecastingSchema
from sklearn.exceptions import NotFittedError
from neuralforecast.models import NHITS
from neuralforecast import NeuralForecast
from pytorch_lightning.callbacks import EarlyStopping
import torch

warnings.filterwarnings("ignore")


PREDICTOR_FILE_NAME = "predictor.joblib"


class Forecaster:
    """A wrapper class for the NHiTS Forecaster.

    This class provides a consistent interface that can be used with other
    Forecaster models.
    """

    model_name = "NHiTS Forecaster"

    def __init__(
        self,
        data_schema: ForecastingSchema,
        history_forecast_ratio: int = None,
        lags_forecast_ratio: int = None,
        lags: int = None,
        stack_types: list = ["identity", "identity", "identity"],
        n_blocks: list = [1, 1, 1],
        mlp_units: list = [[512, 512], [512, 512], [512, 512]],
        n_pool_kernel_size: list = [2, 2, 1],
        n_freq_downsample: list = [4, 2, 1],
        pooling_mode: str = "MaxPool1d",
        interpolation_mode: str = "linear",
        dropout_prob_theta=0.0,
        activation="ReLU",
        max_steps: int = 1000,
        learning_rate: float = 0.001,
        num_lr_decays: int = -1,
        batch_size: int = 32,
        early_stopping: bool = False,
        early_stop_patience_steps: int = 50,
        min_delta: float = 0.0005,
        step_size: int = 1,
        local_scaler_type: str = None,
        use_exogenous: bool = True,
        random_state: int = 0,
        trainer_kwargs: dict = {},
        **kwargs,
    ):
        """Construct a new NHiTS Forecaster

        Args:

            data_schema (ForecastingSchema):
                Schema of training data.

            history_forecast_ratio (int):
                Sets the history length depending on the forecast horizon.
                For example, if the forecast horizon is 20 and the history_forecast_ratio is 10,
                history length will be 20*10 = 200 samples.

            lags_forecast_ratio (int):
                Sets the lags parameters depending on the forecast horizon.
                lags = forecast horizon * lags_forecast_ratio
                This parameters overides lags parameter and uses the most recent values as lags.

            lags (int): Lags of the target to use as features.

            stack_types (List[str]):
                stacks list in the form N * ['identity'], to be deprecated in favor of n_stacks.
                Note that len(stack_types)=len(n_freq_downsample)=len(n_pool_kernel_size).

            n_blocks (List[int]): Number of blocks for each stack. Note that len(n_blocks) = len(stack_types).

            mlp_units (List[List[int]]):
                Structure of hidden layers for each stack type.
                Each internal list should contain the number of units of each hidden layer. Note that len(n_hidden) = len(stack_types).

            n_freq_downsample (List[int]):
                list with the stack's coefficients (inverse expressivity ratios).
                Note that len(stack_types)=len(n_freq_downsample)=len(n_pool_kernel_size).

            interpolation_mode (str): interpolation basis from ['linear', 'nearest', 'cubic'].

            n_pool_kernel_size (List[int]): list with the size of the windows to take a max/avg over. Note that len(stack_types)=len(n_freq_downsample)=len(n_pool_kernel_size).

            pooling_mode (str): input pooling module from ['MaxPool1d', 'AvgPool1d'].

            dropout_prob_theta (float): Float between (0, 1). Dropout for NHITS basis.

            activation (str): Activation from ['ReLU', 'Softplus', 'Tanh', 'SELU', 'LeakyReLU', 'PReLU', 'Sigmoid'].

            max_steps (int): maximum number of training steps.

            learning_rate (float): Learning rate between (0, 1).

            num_lr_decays (int): Number of learning rate decays, evenly distributed across max_steps.

            batch_size (int): Number of different series in each batch.

            early_stopping (bool): If true, uses early stopping.

            early_stop_patience_steps (int): Number of validation iterations before early stopping.

            min_delta (float): Minimum improvement required by the early stopped.

            step_size (int): Step size between each window of temporal data.

            local_scaler_type (str):
                Scaler to apply per-serie to all features before fitting, which is inverted after predicting.
                Can be 'standard', 'robust', 'robust-iqr', 'minmax' or 'boxcox'

            use_exogenous (bool): If true, uses covariates in training.

            trainer_kwargs (dict): keyword trainer arguments inherited from PyTorch Lighning's trainer.

            random_state (int): Sets the underlying random seed at model initialization time.
        """
        self.data_schema = data_schema
        self.lags = lags
        self.use_exogenous = use_exogenous
        self.random_state = random_state
        self._is_trained = False
        self.kwargs = kwargs
        self.history_length = None

        if history_forecast_ratio:
            self.history_length = (
                self.data_schema.forecast_length * history_forecast_ratio
            )

        if lags_forecast_ratio:
            self.lags = lags_forecast_ratio * self.data_schema.forecast_length

        stopper = EarlyStopping(
            monitor="train_loss",
            patience=early_stop_patience_steps,
            min_delta=min_delta,
            verbose=True,
            mode="min",
        )

        if early_stopping:
            trainer_kwargs["callbacks"] = [stopper]

        if torch.cuda.is_available():
            print("GPU is available")
        else:
            print("GPU is not available")
            if trainer_kwargs.get("accelerator") == "gpu":
                trainer_kwargs.pop("accelerator")

        hist_exog_list = None
        stat_exog_list = None

        if use_exogenous:
            if data_schema.past_covariates:
                hist_exog_list = data_schema.past_covariates

            if data_schema.static_covariates:
                stat_exog_list = data_schema.static_covariates

        models = [
            NHITS(
                h=data_schema.forecast_length,
                hist_exog_list=hist_exog_list,
                stat_exog_list=stat_exog_list,
                input_size=self.lags,
                stack_types=stack_types,
                n_blocks=n_blocks,
                mlp_units=mlp_units,
                n_freq_downsample=n_freq_downsample,
                interpolation_mode=interpolation_mode,
                n_pool_kernel_size=n_pool_kernel_size,
                pooling_mode=pooling_mode,
                dropout_prob_theta=dropout_prob_theta,
                activation=activation,
                max_steps=max_steps,
                learning_rate=learning_rate,
                num_lr_decays=num_lr_decays,
                batch_size=batch_size,
                step_size=step_size,
                random_seed=random_state,
                **trainer_kwargs,
            )
        ]

        self.model = NeuralForecast(
            models=models,
            freq=self.map_frequency(data_schema.frequency),
            local_scaler_type=local_scaler_type,
        )

    def map_frequency(self, frequency: str) -> str:
        """
        Maps the frequency in the data schema to the frequency expected by neuralforecast.

        Args:
            frequency (str): The frequency from the schema.

        Returns (str): The mapped frequency.
        """

        frequency = frequency.lower()
        frequency = frequency.split("frequency.")[1]
        if frequency == "yearly":
            return "Y"
        if frequency == "quarterly":
            return "Q"
        if frequency == "monthly":
            return "M"
        if frequency == "weekly":
            return "W"
        if frequency == "daily":
            return "D"
        if frequency == "hourly":
            return "H"
        if frequency == "minutely":
            return "min"
        if frequency == ["secondly"]:
            return "S"
        else:
            return "S"

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares the training data by converting the index to datetime if available
        and drops or keeps other covariates depending on use_exogenous.

            Args:
                data (pd.DataFrame): The training data.
        """

        if self.data_schema.time_col_dtype in ["DATE", "DATETIME"]:
            data[self.data_schema.time_col] = pd.to_datetime(
                data[self.data_schema.time_col]
            )

        groups_by_ids = data.groupby(self.data_schema.id_col)
        all_ids = list(groups_by_ids.groups.keys())

        all_series = [groups_by_ids.get_group(id_).reset_index() for id_ in all_ids]

        if self.history_length:
            for index, series in enumerate(all_series):
                all_series[index] = series.iloc[-self.history_length :]
            data = pd.concat(all_series).drop(columns="index")

        if not self.use_exogenous:
            if self.data_schema.future_covariates:
                data.drop(columns=self.data_schema.future_covariates, inplace=True)

            if self.data_schema.static_covariates:
                data.drop(columns=self.data_schema.static_covariates, inplace=True)

            if self.data_schema.past_covariates:
                data.drop(columns=self.data_schema.past_covariates, inplace=True)

        data.rename(
            columns={
                self.data_schema.id_col: "unique_id",
                self.data_schema.time_col: "ds",
                self.data_schema.target: "y",
            },
            inplace=True,
        )

        return data

    def generate_static_exogenous(self, history: pd.DataFrame) -> pd.DataFrame:
        """
        Generate the dataframe of static covariates

        Args:
            history (pd.DataFrame): The prepared dataframe of history.

        Returns (pd.DataFrame): The static covariates dataframe expected by neuralforecast.
        """
        static_exogenous = history.groupby("unique_id").first().reset_index()
        static_exogenous = static_exogenous[
            ["unique_id"] + self.data_schema.static_covariates
        ]
        return static_exogenous

    def generate_future_exogenous_for_predict(
        self, test_data: pd.DataFrame
    ) -> pd.DataFrame:
        futr_df = test_data[["unique_id", "ds"] + self.data_schema.future_covariates]

        if self.data_schema.time_col_dtype in ["DATE", "DATETIME"]:
            futr_df["ds"] = pd.to_datetime(futr_df["ds"])

        return futr_df

    def fit(
        self,
        history: pd.DataFrame,
    ) -> None:
        """Fit the Forecaster to the training data.

        Args:
            history (pandas.DataFrame): The features of the training data.

        """
        np.random.seed(self.random_state)

        history = self.prepare_data(history)

        static_df = None
        if self.use_exogenous and len(self.data_schema.static_covariates) > 0:
            static_df = self.generate_static_exogenous(history)

        self.model.fit(df=history, static_df=static_df)
        self._is_trained = True
        self.history = history
        self.static_df = static_df

    def predict(
        self, test_data: pd.DataFrame, prediction_col_name: str
    ) -> pd.DataFrame:
        """Make the forecast of given length.

        Args:
            test_data (pd.DataFrame): Given test input for forecasting.
            prediction_col_name (str): Name to give to prediction column.
        Returns:
            pd.DataFrame: The prediction dataframe.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")

        test_data.rename(
            columns={
                self.data_schema.id_col: "unique_id",
                self.data_schema.time_col: "ds",
                self.data_schema.target: "y",
            },
            inplace=True,
        )
        futr_df = None
        if self.use_exogenous and len(self.data_schema.future_covariates) > 0:
            futr_df = self.generate_future_exogenous_for_predict(test_data)

        forecast = self.model.predict(
            df=self.history, futr_df=None, static_df=self.static_df
        )

        forecast[prediction_col_name] = forecast.drop(columns=["ds"]).mean(axis=1)
        forecast.reset_index(inplace=True)
        forecast["ds"] = test_data["ds"]
        forecast.rename(
            columns={
                "unique_id": self.data_schema.id_col,
                "ds": self.data_schema.time_col,
            },
            inplace=True,
        )
        return forecast

    def save(self, model_dir_path: str) -> None:
        """Save the Forecaster to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Forecaster":
        """Load the Forecaster from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Forecaster: A new instance of the loaded Forecaster.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return f"Model name: {self.model_name}"


def train_predictor_model(
    history: pd.DataFrame,
    data_schema: ForecastingSchema,
    hyperparameters: dict,
) -> Forecaster:
    """
    Instantiate and train the predictor model.

    Args:
        history (pd.DataFrame): The training data inputs.
        data_schema (ForecastingSchema): Schema of the training data.
        hyperparameters (dict): Hyperparameters for the Forecaster.

    Returns:
        'Forecaster': The Forecaster model
    """

    model = Forecaster(
        data_schema=data_schema,
        **hyperparameters,
    )
    model.fit(
        history=history,
    )
    return model


def predict_with_model(
    model: Forecaster, test_data: pd.DataFrame, prediction_col_name: str
) -> pd.DataFrame:
    """
    Make forecast.

    Args:
        model (Forecaster): The Forecaster model.
        test_data (pd.DataFrame): The test input data for forecasting.
        prediction_col_name (int): Name to give to prediction column.

    Returns:
        pd.DataFrame: The forecast.
    """
    return model.predict(test_data, prediction_col_name)


def save_predictor_model(model: Forecaster, predictor_dir_path: str) -> None:
    """
    Save the Forecaster model to disk.

    Args:
        model (Forecaster): The Forecaster model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Forecaster:
    """
    Load the Forecaster model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Forecaster: A new instance of the loaded Forecaster model.
    """
    return Forecaster.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Forecaster, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the Forecaster model and return the accuracy.

    Args:
        model (Forecaster): The Forecaster model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The labels of the test data.

    Returns:
        float: The accuracy of the Forecaster model.
    """
    return model.evaluate(x_test, y_test)
