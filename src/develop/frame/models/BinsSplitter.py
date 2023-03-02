import pandas as pd

from src.develop.frame.utils import get_data
from src.segmentation import ImobsSegmentation


class BinsSplitter:

    def __init__(self):
        self.df = get_data("src/develop/frame/data/df_preds.csv")
        self.preprocess_data()

        self.partitioner = None

    def preprocess_data(self):

        self.df = self.df.rename(columns={"dt_calendar": "dt_ativacao"})
        self.df["dt_ativacao"] = pd.to_datetime(self.df["dt_ativacao"])

        print(self.df.shape)

    def segment_bins(self):
        partitioner = ImobsSegmentation()
        partitioner.fit(self.df)

        self.partitioner = partitioner
        print(f"\nSplits: {partitioner.splits}")
        # partitioner.binning_table()

        self.df = partitioner.get_optimal_segments(self.df)

        partitioner.calculate_auc_among_segments(self.df)
        partitioner.calculate_auc_within_segments(self.df, "optimal_bins")

    def export_bins(self):
        self.df.to_csv(f"src/develop/frame/data/df_bins.csv", index=False)

        return print("\n<<< Bins segmented >>>")