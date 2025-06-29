import copy
import matplotlib.pyplot as plt
from prefect.artifacts import create_markdown_artifact, create_table_artifact
import io
import base64
import numpy as np
from torch.nn import Module
from argparse import Namespace
from pandas import DataFrame
    
class ResultLogger:
    """
    A logger class to record training and testing metrics across rounds and generate a markdown report with plots.
    """
    def __init__(self):
        """ Initialize lists to store metrics per round """
        self.rounds = []
        self.test_accuracy = []
        self.test_loss = []
        self.test_auc = []
        self.train_accuracy = []
        self.train_loss = []
        self.train_auc = []
        self.scalars = {}
        self.histograms = {}
        self.tables = {}

    def log(
        self,
        round_number: int,
        test_accuracy: float,
        test_loss: float,
        test_auc: float,
        train_accuracy: float,
        train_loss: float,
        train_auc: float
    ):
        """
        Record metrics for a given round.
        """
        self.add_scalar("test_accuracy", test_accuracy, round_number)
        self.add_scalar("test_loss", test_loss, round_number)
        self.add_scalar("test_auc", test_auc, round_number)
        self.add_scalar("train_accuracy", train_accuracy, round_number)
        self.add_scalar("train_loss", train_loss, round_number)
        self.add_scalar("train_auc", train_auc, round_number)

    def add_scalar(self, tag: str, value: int | float, step: int):
        """
        Log a scalar value.
        :param tag: Name of the metric.
        :param value: Scalar value.
        :param step: The round/iteration at which this value is logged.
        """
        if tag not in self.scalars:
            self.scalars[tag] = []
        self.scalars[tag].append((step, value))

    def add_histogram(self, tag: str, values: np.ndarray, bins=10):
        """
        Log a histogram with complete data.
        :param tag: Name of the metric.
        :param values: A 1D array of values representing the histogram.
        :param bins: (Optional) Number of bins for the histogram. Default is 10.
        """
        self.histograms[tag] = (values, bins)
        
    def add_table(self, tag: str, data: DataFrame):
        """
        Log a table of data.
        :param tag: Name of the table.
        :param data: A pandas DataFrame containing the data to be logged.
        """
        self.tables[tag] = data
        
    def save_raw_data(self, artifact_prefix: str):
        """
        Save raw data to prefect artifacts.
        """
        for tag, data in self.tables.items():
            try:
                create_table_artifact(
                    key=f"{artifact_prefix}-{tag}",
                    table=data.to_dict('records'),
                    description=f"Table for {tag}",
                )
            except Exception as e:
                print(f"Error saving table artifact for {tag}: {e}")
                raise e
        
        for tag, data in self.scalars.items():
            df = DataFrame(data, columns=["Round", tag])
            create_table_artifact(
                key=f"{artifact_prefix}-{tag}",
                table=df.to_dict('records'),
                description=f"Scalar data for {tag}",
            )
        
        for tag, (values, bins) in self.histograms.items():
            df = DataFrame({"Values": values})
            create_table_artifact(
                key=f"{artifact_prefix}-{tag}",
                table=df.to_dict('records'),
                description=f"Histogram data for {tag}",
            )
            
    def create_report(self):
        """
        Generate plots for the recorded metrics and create a markdown report that embeds the images directly as inline base64.
        """
        md_lines = ["# Training Report\n"]
        
        for tag, data in self.tables.items():
            md_lines.append(f"## {tag}\n")
            md_lines.append(data.to_markdown(index=False))
            md_lines.append("\n")

        for tag, data in self.scalars.items():
            data = sorted(data, key=lambda x: x[0])
            steps = [x[0] for x in data]
            values = [x[1] for x in data]
            plt.figure()
            plt.plot(steps, values, marker='o', linestyle='-')
            plt.xlabel("Round")
            plt.ylabel(tag)
            plt.title(f"{tag} over Rounds")
            
            # Save plot to a bytes buffer and encode as base64
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            encoded = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            md_lines.append(f"## {tag}\n")
            md_lines.append(f"![{tag} Plot](data:image/png;base64,{encoded})\n")

        for tag, (values, bins) in self.histograms.items():
            plt.figure()
            plt.hist(values, bins=bins)
            plt.xlabel(tag)
            plt.ylabel("Frequency")
            plt.title(f"{tag} Histogram")
            
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            encoded = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            md_lines.append(f"## {tag} Histogram\n")
            md_lines.append(f"![{tag} Histogram](data:image/png;base64,{encoded})\n")

        md = "\n".join(md_lines)
        
        create_markdown_artifact(
            key="pipeline-report",
            markdown=md,
            description="Result report",
        )