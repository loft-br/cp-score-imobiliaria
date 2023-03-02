import numpy as np
import pandas as pd
import datetime as dt

from optbinning import OptimalBinning
from sklearn.metrics import roc_auc_score


class ImobsSegmentation:

    def __init__(self):

        self.splits = None
        self.optb = OptimalBinning(
            name='score_imobiliaria',
            dtype="numerical",
            max_n_bins=3,
            min_n_bins=3,
            solver="mip",
            gamma=.5
        )

        self.inner_auc = None
        self.opt_among_auc = None

    def fit(self, df):

        x = df.predictions
        y = df.target
        
        self.optb.fit(x, y)
        self.splits = self.optb.splits
        
        return self.optb

    def binning_table(self, mode="build"):

        if mode == "plot":
            return self.optb.binning_table.plot(metric="event_rate")
            
        elif mode == "analysis":
            return self.optb.binning_table.analysis()

        else:
            return self.optb.binning_table.build()

    def get_optimal_segments(self, df):
        _df = df.copy()

        new_segments = dict(
            zip(
                ["1-baixo", "2-medio", "3-alto"],
                sorted(np.unique(
                    self.optb.transform(_df.predictions, metric="event_rate")
                ))
            )
        )
        _df['optimal_bins'] = self.optb.transform(_df.predictions, metric="event_rate")

        for key, value in new_segments.items():
            _df.loc[_df["optimal_bins"] == value, "optimal_segments"] = key

        return _df

    def calculate_auc_within_segments(self, df, segment_column):

        segments_auc = {}

        for rating in df[segment_column].sort_values().unique():
            rating_df = df[df[segment_column] == rating]
            
            auc = roc_auc_score(
                rating_df['target'], rating_df['predictions']
            ).round(3)

            segments_auc[rating] = auc

        inner_auc = pd.DataFrame(
            segments_auc.items(), columns=["rating", "auc"]
        )

        self.inner_auc = inner_auc
        print(f"\nAUC within segments (mean): {round(self.inner_auc.mean(numeric_only=True).auc, 3)}")
        
        return self.inner_auc

    def calculate_auc_among_segments(self, df, alt_segmentation=None):

        self.opt_among_auc = roc_auc_score(df['target'], df['optimal_bins']).round(3)

        print(f"AUC among segments: \
              \n- Optimal segments: {self.opt_among_auc}")
        
        if alt_segmentation:
            print(f"- Alternative segments: {roc_auc_score(df['target'], df[alt_segmentation].apply(ord)).round(3)}")
