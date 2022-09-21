# library doc string
"""
These modules are implemented to predict customer churn.
Models are created by following steps.
1. import data from csv
2. perform EDA
3. feature engineering
4. build machine learning model using Logistic Regression and Random Forest.
5. evaluate model performances and save models.
Author: Yusuke Nakano
Date: 2022/8/10
"""

# import libraries
import os
from typing import List, Dict, Any, Union
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth: str) -> pd.DataFrame:
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df_train = pd.read_csv(pth)
    df_train['Churn'] = df_train['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    df_train = df_train.drop(columns='Attrition_Flag')
    return df_train


def perform_eda(
    df_train: pd.DataFrame,
    out_plot_dir: str,
) -> None:
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe
            out_plot_dir: directory path to output EDA plot images

    output:
            None
    '''
    # translate Attrition_Flag to binary feature that value takes 1 when
    # customer churn occures.
    if 'Unnamed: 0' in df_train.columns:
        df_train = df_train.drop(columns=['Unnamed: 0'])
    if 'CLIENTNUM' in df_train.columns:
        df_train = df_train.drop(columns=['CLIENTNUM'])
    quant_columns = list(df_train.select_dtypes(include=np.number).columns)
    cat_columns = list(df_train.select_dtypes(exclude=np.number).columns)

    fig, axes = plt.subplots(
        len(quant_columns),
        1,
        figsize=(20, 10 * len(quant_columns))
    )

    # plot histograms for quantitative features
    for axis, col in zip(axes.flatten(), quant_columns):
        df_train[col].hist(ax=axis)
    fig.savefig(os.path.join(out_plot_dir, 'histograms.png'))

    fig, axes = plt.subplots(
        len(cat_columns), 1, figsize=(
            20, 10 * len(cat_columns)))

    # count plots for categorical features
    for axis, col in zip(axes.flatten(), cat_columns):
        df_train[col].value_counts('normalize').plot(kind='bar', ax=axis)
    fig.savefig(os.path.join(out_plot_dir, 'value_counts.png'))

    # plot heatmap that visualize correlation matrix
    fig = plt.figure(figsize=(20, 10))
    sns.heatmap(df_train.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    fig.savefig(os.path.join(out_plot_dir, 'correlations.png'))


def encoder_helper(
    df_train: pd.DataFrame,
    category_lst: List[str],
    response: str
) -> pd.DataFrame:
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst:
                list of columns that contain categorical features
            response:
                string of response name [optional argument
                that could be used for naming variables or index y column

    output:
            df: pandas dataframe with new columns for
    '''
    mean_df = pd.DataFrame()
    # target encoding for categorical features
    for col in category_lst:
        churn_means = (
            df_train.groupby(col)[response]
            .mean()
            .to_dict()
        )
        mean_values = df_train[col].map(churn_means)
        mean_df = (pd.concat(
            [mean_df, mean_values],
            axis=1
        )
            .rename(columns={
                    col: f'{col}_Churn'
                    }
                    )
        )
    df_train = pd.concat([df_train, mean_df], axis=1)
    return df_train


def perform_feature_engineering(
    df_train: pd.DataFrame,
    categories_to_encode: List[str],
    other_cols_to_keep: List[str],
    response: str
) -> pd.DataFrame:
    '''
    input:
              df: pandas dataframe
              categories_to_encode:
                    a list of feature names to perform target encoding.
                    encoded features are automatically added to training data.
              other_cols_to_keep:
                    a list of feature names to use as training features.
                    do not include features specified in categories_to_encode.
              response:
                    string of response name [optional argument
                    that could be used for naming variables or index y column

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    # create target variable that takes 1 if customer churn occures
    # perform target encoding
    df_train = encoder_helper(df_train, categories_to_encode, response)
    keep_cols = other_cols_to_keep + \
        [f'{col}_Churn' for col in categories_to_encode]
    y_train_test = df_train[response]
    x_train_test = df_train[keep_cols]
    x_train, x_test, y_train, y_test = train_test_split(
        x_train_test,
        y_train_test,
        test_size=0.3,
        random_state=42
    )
    return x_train, x_test, y_train, y_test


def create_classification_report(
    y_train: Union[pd.Series, np.array],
    y_test: Union[pd.Series, np.array],
    y_train_preds: Union[pd.Series, np.array],
    y_test_preds: Union[pd.Series, np.array],
    classifier: str,
) -> plt.figure:
    """
    create classification report for training and testing results
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds:  training predictions
            y_test_preds:   test predictions
            classifier:  classifier name used for prediction.
                         This string value is used for titles of the report.
    output:
            pyplot.figure: classification report figure
    """
    fig = plt.figure(figsize=(10, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old
    # approach
    plt.text(
        0.01,
        1.25,
        str(f'{classifier} Train'),
        {'fontsize': 10},
        fontproperties='monospace'
    )

    plt.text(
        0.01,
        0.05,
        str(classification_report(y_test, y_test_preds)),
        {'fontsize': 10},
        fontproperties='monospace'
    )  # approach improved by OP -> monospace!

    plt.text(
        0.01,
        0.6,
        str(f'{classifier} Test'),
        {'fontsize': 10},
        fontproperties='monospace'
    )

    plt.text(
        0.01,
        0.7,
        str(classification_report(y_train, y_train_preds)),
        {'fontsize': 10},
        fontproperties='monospace'
    )  # approach improved by OP -> monospace!

    plt.axis('off')
    return fig

def feature_importance_plot(
        model: ClassifierMixin,
        x_data: Union[pd.DataFrame, np.ndarray],
        output_dir: str) -> None:
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [x_data.columns[i] for i in indices]

    plt.figure(figsize=(20, 5))
    plt.title('Feature Importance')
    plt.ylabel('Importance')

    # feature importance bar plot
    plt.bar(range(x_data.shape[1]), importances[indices])
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(os.path.join(output_dir, 'feature_importances_rf.png'))


def train_models(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: Union[pd.Series, np.array],
    y_test: Union[pd.Series, np.array],
    param_grid: Dict[str, Any],
    out_plot_dir: str,
    out_model_dir: str,
):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
              param_grid:
                    a dictionary that specify the combination of parametars
                    to search optimal parameter set.
              out_plot_dir:
                    a directory path to save classification report,
                    feature importance plot and ROC curve
              out_pldel_dir:
                    a directory path to save Logistic Regression model
                    and Random Forest Classifier model.
    output:
              None
    '''
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    cv_rfc = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5
    )

    # build models
    cv_rfc.fit(x_train, y_train)
    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # create and save classification reports

    fig = create_classification_report(
        y_train,
        y_test,
        y_train_preds_lr,
        y_test_preds_lr,
        'Logistic Regression'
    )

    fig.savefig(os.path.join(out_plot_dir, 'classification_report_lr.png'))

    fig = create_classification_report(
        y_train,
        y_test,
        y_train_preds_rf,
        y_test_preds_rf,
        'Random Forest'
    )
    fig.savefig(os.path.join(out_plot_dir, 'classification_report_rf.png'))

    feature_importance_plot(
        cv_rfc.best_estimator_,
        x_test,
        out_plot_dir,
    )

    # ROC curve of logistic regression
    lrc_plot = plot_roc_curve(lrc, x_test, y_test)
    # convine ROC curves of both logistic and random forest
    plt.figure(figsize=(15, 8))
    ax_roc = plt.gca()
    plot_roc_curve(
        cv_rfc.best_estimator_,
        x_test,
        y_test,
        ax=ax_roc,
        alpha=0.8)
    lrc_plot.plot(ax=ax_roc, alpha=0.8)
    plt.savefig(os.path.join(out_plot_dir, 'roc_curve.png'))
    #del rfc_disp

    # save models
    joblib.dump(
        lrc,
        os.path.join(out_model_dir, 'logistic_model.pkl')
    )
    joblib.dump(
        cv_rfc.best_estimator_,
        os.path.join(out_model_dir, 'rfc_model.pkl')
    )


if __name__ == '__main__':
    # load parameters from parameters.yml
    with open('./parameters.yml') as f:
        parameters = yaml.safe_load(f.read())

    df = import_data('./data/bank_data.csv')

    perform_eda(
        df,
        './images',
    )

    X_train_churn, X_test_churn, y_train_churn, y_test_churn = perform_feature_engineering(
        df, parameters['categories_to_encode'], parameters['other_cols_to_keep'], 'Churn')

    train_models(
        X_train_churn,
        X_test_churn,
        y_train_churn,
        y_test_churn,
        parameters['grid_search_params'],
        './images',
        './models'
    )
