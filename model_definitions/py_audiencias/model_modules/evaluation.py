from sklearn.metrics import confusion_matrix
from teradataml import (
    copy_to_sql,
    DataFrame,
    DecisionForestPredict,
    ClassificationEvaluator,
    ConvertTo,
    ROC
)
from aoa import (
    record_evaluation_stats,
    save_plot,
    aoa_create_context,
    ModelContext
)
import json
import os

def plot_feature_importance(fi, img_filename):
    import pandas as pd
    import matplotlib.pyplot as plt
    feat_importances = pd.Series(fi)
    feat_importances.nlargest(10).plot(kind='barh').set_title('Feature Importance')
    fig = plt.gcf()
    fig.savefig(img_filename, dpi=500)
    plt.clf()
    
    
def plot_confusion_matrix(cf, img_filename):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(cf, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cf.shape[0]):
        for j in range(cf.shape[1]):
            ax.text(x=j, y=i,s=cf[i, j], va='center', ha='center', size='xx-large')
    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix');
    fig = plt.gcf()
    fig.savefig(img_filename, dpi=500)
    plt.clf()

    
def plot_roc_curve(roc_out, img_filename):
    import matplotlib.pyplot as plt
    auc = roc_out.result.to_pandas().reset_index()['AUC'][0]
    roc_results = roc_out.output_data.to_pandas()
    plt.plot(roc_results['fpr'], roc_results['tpr'], color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % 0.27)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    fig = plt.gcf()
    fig.savefig(img_filename, dpi=500)
    plt.clf()

    
def evaluate(context: ModelContext, **kwargs):

    aoa_create_context()

    model = DataFrame(f"model_${context.model_version}")

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]
    entity_key = context.dataset_info.entity_key

    test_df = DataFrame.from_query(context.dataset_info.sql)

    # Scaling the test set
    print ("Loading scaler...")
    predictions = DecisionForestPredict(
        object=model,
        newdata=test_df,
        id_column=entity_key,
        detailed=False,
        terms=[target_name],
        output_prob=True
    )

    predicted_data = ConvertTo(
        data = predictions.result,
        target_columns = [target_name,'prediction'],
        target_datatype = ["INTEGER"]
    )

    ClassificationEvaluator_obj = ClassificationEvaluator(
        data=predicted_data.result,
        observation_column=target_name,
        prediction_column='prediction',
        num_labels=2
    )

    metrics_pd = ClassificationEvaluator_obj.output_data.to_pandas()

    evaluation = {
        'Accuracy': '{:.2f}'.format(metrics_pd.MetricValue[0]),
        'Micro-Precision': '{:.2f}'.format(metrics_pd.MetricValue[1]),
        'Micro-Recall': '{:.2f}'.format(metrics_pd.MetricValue[2]),
        'Micro-F1': '{:.2f}'.format(metrics_pd.MetricValue[3]),
        'Macro-Precision': '{:.2f}'.format(metrics_pd.MetricValue[4]),
        'Macro-Recall': '{:.2f}'.format(metrics_pd.MetricValue[5]),
        'Macro-F1': '{:.2f}'.format(metrics_pd.MetricValue[6]),
        'Weighted-Precision': '{:.2f}'.format(metrics_pd.MetricValue[7]),
        'Weighted-Recall': '{:.2f}'.format(metrics_pd.MetricValue[8]),
        'Weighted-F1': '{:.2f}'.format(metrics_pd.MetricValue[9]),
    }

    with open(f"{context.artifact_output_path}/metrics.json", "w+") as f:
        json.dump(evaluation, f)
        
    cm = confusion_matrix(predicted_data.result.to_pandas()[target_name], predicted_data.result.to_pandas()['prediction'])
    plot_confusion_matrix(cm, f"{context.artifact_output_path}/confusion_matrix")

    roc_out = ROC(
        data=predictions.result,
        probability_column='prob',
        observation_column=target_name,
        positive_class='1',
        num_thresholds=1000
    )
    plot_roc_curve(roc_out, f"{context.artifact_output_path}/roc_curve")

    predictions_table = "predictions_tmp"
    copy_to_sql(df=predicted_data.result, table_name=predictions_table, index=False, if_exists="replace", temporary=True)

    # calculate stats if training stats exist
    if os.path.exists(f"{context.artifact_input_path}/data_stats.json"):
        record_evaluation_stats(
            features_df=test_df,
            predicted_df=DataFrame.from_query(f"SELECT * FROM {predictions_table}"),
            context=context
        )