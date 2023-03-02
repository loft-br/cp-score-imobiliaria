import pandas as pd
import datetime as dt

from ..utils import get_data
from risk_suite.economics import EconomicsCalculator


class EconCalculator:

    def __init__(self):

        self.df_bins = get_data("src/develop/frame/data/df_bins.csv")
        self.df_new_imobs = pd.read_csv("src/develop/frame/data/df_new_imobs.csv")
        self.df_scores = (pd.read_csv("src/develop/frame/data/scores.csv")
            [["dt_ativacao", "id_contrato", "faixa_score"]]
        )
        self.info_contracts = (pd.read_csv("src/develop/frame/data/info_contracts.csv")
            [["dt_ativacao", "id_imobiliaria", "id_contrato"]]
        )

        self.df = None
        self.pivot_tables = None
        self.calculator_basis = None

        print("\nPreparing data...")
        self.preprocess_data()
        print("Running calculator prerequisites...")
        self.load_calculator_prereqs()


    def preprocess_data(self):

        self.df_new_imobs["optimal_segments"] = "4-novas"
        self.df_scores = self.df_scores.rename(columns={
            "id_contrato": "contract_id",
            "faixa_score": "rating_4KST"
        })
        self.info_contracts = self.info_contracts.rename(columns={"id_contrato": "contract_id"})

        self.df_bins = pd.concat([self.df_bins, self.df_new_imobs], ignore_index=True)
        
        for df in [self.df_bins, self.df_new_imobs, self.df_scores, self.info_contracts]: 
            df["dt_ativacao"] = pd.to_datetime(df["dt_ativacao"]).dt.strftime("%Y-%m") 

        self.df = (
            self.info_contracts
            .merge(self.df_bins, on=["dt_ativacao", "id_imobiliaria"])
            .merge(self.df_scores, on=["dt_ativacao", "contract_id"], how="left")
        )

    def init_calculator(self):

        self.df = self.calculator_basis["contracts"].merge(self.df, on="contract_id")

        econ_calculator = EconomicsCalculator(
            self.df,
            self.calculator_basis["defaults"],
            self.calculator_basis["recoveries"],
            self.calculator_basis["revenues"],
            max_history_date='2023-01'
        )

        self.pivot_tables = {
            "defaults_pivot": econ_calculator._build_defaults_pivot(),
            "recoveries_pivot": econ_calculator._build_recoveries_pivot(),
            "revenues_pivot": econ_calculator._build_revenues_pivot()
        }

        return print("\n<<< Calculator is ready to use >>>")

    def build_report(self, grouped_by, pivot_values=None):

        report = (
            self.report_economics(grouped_by,
            self.pivot_tables["revenues_pivot"],
            self.pivot_tables["defaults_pivot"],
            self.pivot_tables["recoveries_pivot"])
        )

        if pivot_values:
            return self.pivot_report(report, pivot_values)
        
        return report
         
    def pivot_report(self, report, pivot_values):

        return pd.pivot_table(
            report,
            values=pivot_values, index="optimal_segments", columns="rating_4KST"
        )


    def load_calculator_prereqs(self):

        contracts = pd.read_parquet('/Users/raquel.camara/Documents/CredPago/risk_suite/data/contracts.parquet')
        defaults = pd.read_parquet('/Users/raquel.camara/Documents/CredPago/risk_suite/data/defaults.parquet')
        recoveries = pd.read_parquet('/Users/raquel.camara/Documents/CredPago/risk_suite/data/recoveries.parquet')
        revenues = pd.read_parquet('/Users/raquel.camara/Documents/CredPago/risk_suite/data/revenues.parquet')

        base_features = [
            'contract_id',
            'activation_date',
            'churn_date',
            'activation_month',
            'activation_quarter',
            'score_serasa',
            'rating',
            'rental_value',
        ]

        contracts = contracts[base_features]
        contracts = contracts.dropna(subset=['activation_date', 'rating'])
        contracts = contracts.loc[lambda x: x['activation_quarter'] >= pd.Period('2020Q1')]
        contracts = contracts.replace(['E1', 'E2', 'E3'], 'E')

        self.calculator_basis = {
            "contracts": contracts,
            "defaults": defaults,
            "recoveries": recoveries,
            "revenues": revenues
        }


    def report_economics(self, aggkeys, revenues_pivot, defaults_pivot, recoveries_pivot):
    
        economics_df = pd.DataFrame(
            {
                'n_contracts': revenues_pivot.groupby(aggkeys).size(),
                'revenue_value': (revenues_pivot.sum(axis=1)).groupby(aggkeys).mean(),
                'prob_default': (defaults_pivot.sum(axis=1) > 0).groupby(aggkeys).mean(),
                'default_value': (defaults_pivot.sum(axis=1)).groupby(aggkeys).mean(),
                'recovery_value': (recoveries_pivot.sum(axis=1)).groupby(aggkeys).mean()
            }
        )

        economics_df = (
            economics_df
            .assign(recovery_efficiency=lambda x: x['recovery_value'] / x['default_value'])
            .assign(unit_economics=lambda x: x['revenue_value'] - x['default_value'] + x['recovery_value'])
            .assign(aggregate_margin=lambda x: x['unit_economics'] * x['n_contracts'])
        )

        return economics_df.sort_index(ascending=[True, False])

        