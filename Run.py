import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Load the ready dataset
dataset=pd.read_csv('dataset_ready.csv')

measured_variables = dataset.columns.tolist()

import dowhy
from dowhy import CausalModel


graph_str = """
digraph {
    "distribution_channel" -> "lead_time";
    "financial_status_hidden" -> "distribution_channel";
    "distribution_channel" -> "is_canceled";
    "distribution_channel" -> "days_in_waiting_list";
    "distribution_channel" -> "different_room_assigned";
    "hotel" -> "booking_changes";
    "hotel" -> "total_of_special_requests";
    "hotel" -> "local_events_hidden";
    "hotel" -> "total_days";
    "hotel" -> "arrival_date_month";
    "lead_time" -> "is_canceled"; 
    "country" -> "lead_time";
    "lead_time" -> "days_in_waiting_list";
    "lead_time" -> "deposit_type";
    "local_events_hidden" -> "lead_time";
    "local_events_hidden" -> "booking_changes";
    "local_events_hidden" -> "is_canceled";
    "local_events_hidden" -> "deposit_type";
    "local_events_hidden" -> "total_days";
    "country" -> "local_events_hidden";
    "country" -> "hotel_policies_hidden";
    "different_room_assigned" -> "is_canceled"; 
    "total_of_special_requests" -> "different_room_assigned";
    "hotel_policies_hidden" -> "different_room_assigned";
    "total_guests" -> "different_room_assigned";
    "deposit_type" -> "different_room_assigned";
    "is_repeated_guest" -> "is_canceled";
    "days_in_waiting_list" -> "is_canceled";
    "hotel_policies_hidden" -> "days_in_waiting_list";
    "previous_bookings_not_canceled" -> "is_canceled";
    "previous_bookings_not_canceled" -> "is_repeated_guest";
    "financial_status_hidden" -> "deposit_type";
    "financial_status_hidden" -> "is_canceled";
    "arrival_date_month" -> "total_days";
    "is_repeated_guest" -> "is_canceled";
    "total_days" -> "is_canceled";
    "total_days" -> "agent";
    "agent" -> "is_canceled";
    "financial_status_hidden" -> "agent";
    "country" -> "agent";
    "agent" -> "days_in_waiting_list";
    "total_guests" -> "is_canceled";
    "previous_cancellations" -> "is_canceled";
    "previous_cancellations" -> "is_repeated_guest";
    "total_guests" -> "required_car_parking_spaces";
    "total_days" -> "required_car_parking_spaces";
    "total_of_special_requests" -> "is_canceled";
    "booking_changes" -> "different_room_assigned";
    "booking_changes" -> "is_canceled";
    "is_canceled" -> "adr";
    "total_of_special_requests" -> "days_in_waiting_list";
}
"""

dataset_modeling=dataset.copy()
model = CausalModel(
    data=dataset_modeling,
    treatment='different_room_assigned',
    outcome='is_canceled',
    graph=graph_str
)

# Identification
identified_estimand = model.identify_effect()

# Choosing backdoor criterion based on the identification
from pathlib import Path

identified_estimand.default_backdoor_id = 'backdoor'
desired_effect="ate"

import warnings
import sys
import io
import time

estimated_regular_ate = True

if not estimated_regular_ate:
    warnings.filterwarnings("ignore", category=FutureWarning)
    methods = ['backdoor.linear_regression', 'backdoor.propensity_score_weighting', 'backdoor.propensity_score_matching', 'backdoor.propensity_score_stratification']

    for method in methods:
        start_time = time.time()
        start_time_date = datetime.now()
        print(f'Start time {method}: {start_time_date.strftime("%Y-%m-%d %H:%M:%S")}\n')
        
        try:
            estimate = model.estimate_effect(
                identified_estimand,
                method_name=method,
                target_units=desired_effect,
                confidence_intervals=True,
                test_significance=True,
            )

            default_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()

            estimate.interpret()

            interpretation = buffer.getvalue()
            sys.stdout = default_stdout

            refute_placebo_treatment = model.refute_estimate(
                identified_estimand,
                estimate,
                method_name="placebo_treatment_refuter",
                placebo_type="permute")

            end_time = time.time()
            elapsed_time = end_time - start_time
            minutes, seconds = divmod(elapsed_time, 60)
            
            save_path = Path("/Users/shadi-omari/Repos/CausalInference/Results/ATE/Regular")
            save_path.mkdir(parents=True, exist_ok=True)
            file_path = save_path / f"{method}_ATE.txt"
            with open(file_path, "w") as file:
                file.write(str(estimate))
                file.write('\n\n\n -----------------------------------------------------------------------\n')
                file.write(interpretation)
                file.write('\n\n\n -----------------------------------------------------------------------\n')
                file.write('refute_placebo_treatment: ' + str(refute_placebo_treatment))
                file.write('\n\n\n -----------------------------------------------------------------------\n')
                file.write(f'Time taken to estimate: {int(minutes)} minutes and {seconds:.2f} seconds\n')

            print('\n\n\n -----------------------------------------------------------------------\n')
            end_time_date = datetime.now()
            print(f'Time taken to estimate {method}: {int(minutes)} minutes and {seconds:.2f} seconds. End Time: {end_time_date.strftime("%Y-%m-%d %H:%M:%S")}\n')
            
        except AttributeError as e:
            print(f"Error encountered with {method}: {e}")

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

import warnings
import time
import io
import sys

estimated_Slearner_ate = True 

if not estimated_Slearner_ate:
    warnings.filterwarnings(action='ignore', category=UserWarning)

    ml_models=[LinearRegression, Ridge, DecisionTreeRegressor, LogisticRegression, MLPRegressor, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, DecisionTreeClassifier]
    
    for ml_model in ml_models:
        start_time = time.time()
        start_time_date = datetime.now()
        print(f'Start time {ml_model.__name__}: {start_time_date.strftime("%Y-%m-%d %H:%M:%S")}\n')

        try:
            estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.econml.metalearners.SLearner",
                method_params={
                    "init_params": {'overall_model': ml_model()},
                    "fit_params": {}
                },
                target_units=desired_effect,
                test_significance=True,
            )
            
            default_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()

            estimate.interpret()

            interpretation = buffer.getvalue()
            sys.stdout = default_stdout

            refute_placebo_treatment = model.refute_estimate(
                identified_estimand,
                estimate,
                method_name="placebo_treatment_refuter",
                placebo_type="permute")
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            minutes, seconds = divmod(elapsed_time, 60)

            save_path = Path("/Users/shadi-omari/Repos/CausalInference/Results/ATE/SLearner")
            save_path.mkdir(parents=True, exist_ok=True)
            file_path = save_path / f"{ml_model.__name__}_ATE.txt"
            with open(file_path, "w") as file:
                file.write(str(estimate))
                file.write('\n\n\n -----------------------------------------------------------------------\n')
                file.write(interpretation)
                file.write('\n\n\n -----------------------------------------------------------------------\n')
                file.write('refute_placebo_treatment: ' + str(refute_placebo_treatment))
                file.write('\n\n\n -----------------------------------------------------------------------\n')
                file.write(f'Time taken to estimate: {int(minutes)} minutes and {seconds:.2f} seconds\n')

            print('\n\n\n -----------------------------------------------------------------------\n')
            end_time_date = datetime.now()
            print(f'Time taken to estimate {ml_model.__name__}: {int(minutes)} minutes and {seconds:.2f} seconds. End Time: {end_time_date.strftime("%Y-%m-%d %H:%M:%S")}\n')

        except AttributeError as e:
            print(f"Error encountered with {ml_model.__name__}: {e}")

import warnings
import time

estimated_Tlearner_ate = True

if not estimated_Tlearner_ate:
    warnings.filterwarnings(action='ignore', category=UserWarning)

    ml_models=[LinearRegression, Ridge, DecisionTreeRegressor, LogisticRegression, MLPRegressor, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, DecisionTreeClassifier]

    # Estimate the effect using a custom econml model
    for ml_model in ml_models:
        start_time = time.time()
        start_time_date = datetime.now()
        print(f'Start time {ml_model.__name__}: {start_time_date.strftime("%Y-%m-%d %H:%M:%S")}\n')
        
        try:
            estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.econml.metalearners.TLearner",
                method_params={
                    "init_params": {'models': [ml_model(), ml_model()]},
                    "fit_params": {}
                },
                target_units=desired_effect,
                test_significance=True,
            )

            default_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()

            estimate.interpret()

            interpretation = buffer.getvalue()
            sys.stdout = default_stdout

            refute_placebo_treatment = model.refute_estimate(
                identified_estimand,
                estimate,
                method_name="placebo_treatment_refuter",
                placebo_type="permute")
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            minutes, seconds = divmod(elapsed_time, 60)

            save_path = Path("/Users/shadi-omari/Repos/CausalInference/Results/ATE/TLearner")
            save_path.mkdir(parents=True, exist_ok=True)
            file_path = save_path / f"{ml_model.__name__}_ATE.txt"
            with open(file_path, "w") as file:
                file.write(str(estimate))
                file.write('\n\n\n -----------------------------------------------------------------------\n')
                file.write(interpretation)
                file.write('\n\n\n -----------------------------------------------------------------------\n')
                file.write('refute_placebo_treatment: ' + str(refute_placebo_treatment))
                file.write('\n\n\n -----------------------------------------------------------------------\n')
                file.write(f'Time taken to estimate: {int(minutes)} minutes and {seconds:.2f} seconds\n')
            
            print('\n\n\n -----------------------------------------------------------------------\n')
            end_time_date = datetime.now()
            print(f'Time taken to estimate {ml_model.__name__}: {int(minutes)} minutes and {seconds:.2f} seconds. End Time: {end_time_date.strftime("%Y-%m-%d %H:%M:%S")}\n')

        except AttributeError as e:
            print(f"Error encountered with {ml_model.__name__}: {e}")

# Choosing backdoor criterion based on the identification

identified_estimand.default_backdoor_id = 'backdoor'
desired_effect="att"

import warnings
import sys
import io
import time

estimated_regular_att = True

if not estimated_regular_att:
    warnings.filterwarnings("ignore", category=FutureWarning)
    methods = ['backdoor.linear_regression', 'backdoor.propensity_score_weighting', 'backdoor.propensity_score_matching', 'backdoor.propensity_score_stratification']

    for method in methods:
        start_time = time.time()
        start_time_date = datetime.now()
        print(f'Start time {method}: {start_time_date.strftime("%Y-%m-%d %H:%M:%S")}\n')
        
        try:
            estimate = model.estimate_effect(
                identified_estimand,
                method_name=method,
                target_units=desired_effect,
                confidence_intervals=True,
                test_significance=True,
            )

            default_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()

            estimate.interpret()

            interpretation = buffer.getvalue()
            sys.stdout = default_stdout

            refute_placebo_treatment = model.refute_estimate(
                identified_estimand,
                estimate,
                method_name="placebo_treatment_refuter",
                placebo_type="permute")
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            minutes, seconds = divmod(elapsed_time, 60)

            save_path = Path("/Users/shadi-omari/Repos/CausalInference/Results/ATT/Regular")
            save_path.mkdir(parents=True, exist_ok=True)
            file_path = save_path / f"{method}_ATT.txt"
            with open(file_path, "w") as file:
                file.write(str(estimate))
                file.write('\n\n\n -----------------------------------------------------------------------\n')
                file.write(interpretation)
                file.write('\n\n\n -----------------------------------------------------------------------\n')
                file.write('refute_placebo_treatment: ' + str(refute_placebo_treatment))
                file.write('\n\n\n -----------------------------------------------------------------------\n')
                file.write(f'Time taken to estimate: {int(minutes)} minutes and {seconds:.2f} seconds\n')

            print('\n\n\n -----------------------------------------------------------------------\n')
            end_time_date = datetime.now()
            print(f'Time taken to estimate {method}: {int(minutes)} minutes and {seconds:.2f} seconds. End Time: {end_time_date.strftime("%Y-%m-%d %H:%M:%S")}\n')
            
        except AttributeError as e:
            print(f"Error encountered with {method}: {e}")

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

import warnings
import time
import io
import sys

estimated_Slearner_att = True 

if not estimated_Slearner_att:
    warnings.filterwarnings(action='ignore', category=UserWarning)

    ml_models=[LinearRegression, Ridge, DecisionTreeRegressor, LogisticRegression, MLPRegressor, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, DecisionTreeClassifier]
    
    for ml_model in ml_models:
        start_time = time.time()
        start_time_date = datetime.now()
        print(f'Start time {ml_model.__name__}: {start_time_date.strftime("%Y-%m-%d %H:%M:%S")}\n')
        
        try:
            estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.econml.metalearners.SLearner",
                method_params={
                    "init_params": {'overall_model': ml_model()},
                    "fit_params": {}
                },
                target_units=desired_effect,
                test_significance=True,
            )
            
            default_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()

            estimate.interpret()

            interpretation = buffer.getvalue()
            sys.stdout = default_stdout

            refute_placebo_treatment = model.refute_estimate(
                identified_estimand,
                estimate,
                method_name="placebo_treatment_refuter",
                placebo_type="permute")
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            minutes, seconds = divmod(elapsed_time, 60)

            save_path = Path("/Users/shadi-omari/Repos/CausalInference/Results/ATT/SLearner")
            save_path.mkdir(parents=True, exist_ok=True)
            file_path = save_path / f"{ml_model.__name__}_ATT.txt"
            with open(file_path, "w") as file:
                file.write(str(estimate))
                file.write('\n\n\n -----------------------------------------------------------------------\n')
                file.write(interpretation)
                file.write('\n\n\n -----------------------------------------------------------------------\n')
                file.write('refute_placebo_treatment: ' + str(refute_placebo_treatment))
                file.write('\n\n\n -----------------------------------------------------------------------\n')
                file.write(f'Time taken to estimate: {int(minutes)} minutes and {seconds:.2f} seconds\n')

            print('\n\n\n -----------------------------------------------------------------------\n')
            end_time_date = datetime.now()
            print(f'Time taken to estimate {ml_model.__name__}: {int(minutes)} minutes and {seconds:.2f} seconds. End Time: {end_time_date.strftime("%Y-%m-%d %H:%M:%S")}\n')

        except AttributeError as e:
            print(f"Error encountered with {ml_model.__name__}: {e}")

import warnings
import time

estimated_Tlearner_att = True

if not estimated_Tlearner_att:
    warnings.filterwarnings(action='ignore', category=UserWarning)

    ml_models=[LinearRegression, Ridge, DecisionTreeRegressor, LogisticRegression, MLPRegressor, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, DecisionTreeClassifier]

    # Estimate the effect using a custom econml model
    for ml_model in ml_models:
        start_time = time.time()
        start_time_date = datetime.now()
        print(f'Start time {ml_model.__name__}: {start_time_date.strftime("%Y-%m-%d %H:%M:%S")}\n')
        
        try:
            estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.econml.metalearners.TLearner",
                method_params={
                    "init_params": {'models': [ml_model(), ml_model()]},
                    "fit_params": {}
                },
                target_units=desired_effect,
                test_significance=True,
            )

            default_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()

            estimate.interpret()

            interpretation = buffer.getvalue()
            sys.stdout = default_stdout

            refute_placebo_treatment = model.refute_estimate(
                identified_estimand,
                estimate,
                method_name="placebo_treatment_refuter",
                placebo_type="permute")
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            minutes, seconds = divmod(elapsed_time, 60)

            save_path = Path("/Users/shadi-omari/Repos/CausalInference/Results/ATT/TLearner")
            save_path.mkdir(parents=True, exist_ok=True)
            file_path = save_path / f"{ml_model.__name__}_ATT.txt"
            with open(file_path, "w") as file:
                file.write(str(estimate))
                file.write('\n\n\n -----------------------------------------------------------------------\n')
                file.write(interpretation)
                file.write('\n\n\n -----------------------------------------------------------------------\n')
                file.write('refute_placebo_treatment: ' + str(refute_placebo_treatment))
                file.write('\n\n\n -----------------------------------------------------------------------\n')
                file.write(f'Time taken to estimate: {int(minutes)} minutes and {seconds:.2f} seconds\n')
            
            print('\n\n\n -----------------------------------------------------------------------\n')
            end_time_date = datetime.now()
            print(f'Time taken to estimate {ml_model.__name__}: {int(minutes)} minutes and {seconds:.2f} seconds. End Time: {end_time_date.strftime("%Y-%m-%d %H:%M:%S")}\n')

        except AttributeError as e:
            print(f"Error encountered with {ml_model.__name__}: {e}")