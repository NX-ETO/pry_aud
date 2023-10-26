from teradataml import *
from aoa import (
    record_training_stats,
    save_plot,
    aoa_create_context,
    ModelContext
)
import numpy as np

def train(context: ModelContext, **kwargs):
    aoa_create_context()

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]

    # read training dataset from Teradata and convert to pandas
    train_df = DataFrame.from_query(context.dataset_info.sql)

    print("Starting training...")

    model = DecisionForest(data=train_df,
                            input_columns = feature_names, 
                            response_column = target_name, 
                            max_depth = context.hyperparams["max_depth"], 
                            num_trees = context.hyperparams["num_trees"], 
                            min_node_size = context.hyperparams["min_node_size"], 
                            mtry = context.hyperparams["mtry"], 
                            mtry_seed = context.hyperparams["mtry_seed"], 
                            seed = context.hyperparams["seed"], 
                            tree_type = context.hyperparams["tree_type"])
    
    model.result.to_sql(f"model_${context.model_version}", if_exists="replace")    
    print("Saved trained model")

    record_training_stats(
        train_df,
        features=feature_names,
        targets=[target_name],
        categorical=[target_name],
        context=context
    )