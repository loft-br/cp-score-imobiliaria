import pandas as pd
from .params_dev import *

from .models.Optimizer import Optimizer
from .models.BinsSplitter import BinsSplitter
from .models.DataCustomizer import DataCustomizer
from .models.ModelGenerator import ModelGenerator
from .models.EconCalculator import EconCalculator
from .models.ReportGenerator import ReportGenerator

if __name__ == "__main__":

    print("\nPreparing Data...")
    customizer = DataCustomizer()
    customizer.create_target()
    customizer.shift_features(columns_to_shift=columns_to_shift)
    customizer.transform_features(**agg_params)
    customizer.filter_new_imobs()
    customizer.export_data(column_to_filter)

    with open(f'src/develop/frame/reports/{report_name}.txt', 'w') as outfile:

        report = ReportGenerator(outfile)
        report.add_header(report_description)

        report.add_line(f"""
Features:
{FEATURES}

Target Balance:
""")
        report.tabule_df(pd.DataFrame({
                "False": [customizer.df.groupby("target").size()[0]],
                "True": [customizer.df.groupby("target").size()[1]]
            }),
            format='simple'
        )

        print("\nOptimizing Hyperparameters...")
        optimizer = Optimizer()
        optimizer.find_best_params(**opt_params)
        optimizer.export_params()

        report.insert_break()
        report.add_line("<<< HYPEROPT RESULTS >>>")
        report.add_line(f"""
Best AUC: {optimizer.best_result}
Best Params: {optimizer.best_params}
""")
        print("\nCreating Model...")
        generator = ModelGenerator()
        generator.run_pipeline(**pipe_params)
                        
        report.insert_break()
        report.add_line("<<< MODEL'S RESULTS >>>")
        report.add_line(f"""
CV AUC: {generator.cv_result}
Testing AUC: {generator.teste_result}
""")
        print("\nFinding optimal segmentation...")
        splitter = BinsSplitter()
        splitter.segment_bins()
        splitter.export_bins()

        report.insert_break()                
        report.add_line("<<< BINS SEGMENTATION >>>")
        report.add_line(f"""
Splits: {splitter.partitioner.splits}
        """)
        report.tabule_df(splitter.partitioner.binning_table(), format='mixed_outline')
        report.add_line(f"""
AUC within segments (mean): {round(splitter.partitioner.inner_auc.mean(numeric_only=True).auc, 3)}
AUC among segments:
    - Optimal segments: {splitter.partitioner.opt_among_auc}
        """)
        report.tabule_df(splitter.partitioner.inner_auc, format='mixed_outline')
        
        # report.add_image(splitter.partitioner.binning_table(mode="plot"))

        print("\nCalculating Economics...")
        calculator = EconCalculator()
        calculator.init_calculator()

        report.insert_break()
        report.add_line("<<< ECONOMICS REPORTS >>>")
        report.tabule_df(calculator.build_report(unique_feat), format='mixed_outline', showindex=True)

        for values in pivot_values:
            report.add_line(f"\n{values.upper().replace('_', ' ')}")
            report.tabule_df(calculator.build_report(multiple_feat, values), format='mixed_outline', showindex=True)

        
        
