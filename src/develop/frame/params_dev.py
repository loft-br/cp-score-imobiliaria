######################################################################################################
#                                           REPORT PARAMS 
######################################################################################################
report_name = "teste"
report_description = "Add a description for the test here"
unique_feat = ["optimal_segments"]
multiple_feat = ["optimal_segments", "rating_4KST"]
pivot_values = ["prob_default", "recovery_efficiency", "unit_economics", "n_contracts"]

######################################################################################################
#                                       DATA CUSTOMIZER PARAMS 
######################################################################################################
agg_params = {
    "window": 6,
    "aggregate_list": {
        # "is_activated": "agg_activated_last_90days",
        "is_commun_first_90days": "agg_comun_last_90days",
        # "is_debelado_first_4months": "agg_deb_last_4months",
        # "exonerated_first_6months": "agg_exon_last_6months",
        "churn_rate": "agg_churn_rate",
        # "rating_A": "agg_rating_A",
        # "rating_B": "agg_rating_B",
        # "rating_C": "agg_rating_C",
        # "rating_D": "agg_rating_D",
        # "rating_E": "agg_rating_E"
    }
}

FEATURES = [
    "lat_imob",
    "long_imob",
    "is_activated",
    # "agg_activated_last_90days",
    "rating_A",
    "rating_B",
    "rating_C",
    "rating_D",
    "rating_E",
    "agg_churn_rate",
    "agg_comun_last_90days",
    # "is_debelado_first_4months",
    # "exonerated_first_6months",
    # "indemnity_value",
    # "vl_locacao",

]

columns_to_shift={
    1: ["is_churn"],
    3: ["is_commun_first_90days"],
    4: ["is_debelado_first_4months"],
    6: ["exonerated_first_6months"]

}
column_to_filter="exonerated_first_6months"

######################################################################################################
#                                        OPTIMIZER PARAMS 
######################################################################################################
opt_params = {
    "from_": "2021-06",
    "to_": "2021-11",
    "trials": 20
}

######################################################################################################
#                                     MODEL GENERATOR PARAMS 
######################################################################################################
pipe_params = {
    "lr": {
        'solver':'sag',
        'C': 1e6,
        'fit_intercept': True
    },
    "cv": {
        "from_": "2021-12",
        "to_": "2022-05"
    },
    "teste": {
    "cutoff_period": "2022-05"
    },
    "historic": ["2021-12", "2022-01", "2022-02", "2022-03", "2022-04", "2022-05"]
}
