import enum
import io
import sys
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn import ensemble, linear_model, tree, model_selection, svm, naive_bayes, neighbors, neural_network, metrics, \
    feature_selection, pipeline
from sklearn.tree import _tree

PATH_IMAGES = '../tex/iterations/iteration_3/images/'

databases_path = './datasets/'


class DataFramesCSV(enum.Enum):
    ALCOHOL_CONSUMPTION_CSV = f"{databases_path}4_total-alcohol-consumption-per-capita-litres-of-pure-alcohol.csv"
    COUNTRY_MASTER_CSV = f"{databases_path}0_master_country_codes.csv"
    WHO_OBESITY_CSV = f"{databases_path}1_who_obesity.csv"
    MEAT_CONSUMPTION_CSV = f"{databases_path}2_meat_consumption.csv"
    HUNGER_CSV = f"{databases_path}5_global_hunger_index.csv"
    SMOKING_CSV = f"{databases_path}6_share-of-adults-who-smoke.csv"


class DataFramesXLSX(enum.Enum):
    HAPPINESS_REPORT_XLSX = f"{databases_path}3_happiness_report.xlsx"


COLUMN_RENAME_BY_DATASET = {
    DataFramesCSV.WHO_OBESITY_CSV: {
        'Numeric': 'percentage_obesity',
        'Countries, territories and areas': 'country',
        'WHO region': 'region',
        'Year': 'year',
    },
    DataFramesXLSX.HAPPINESS_REPORT_XLSX: {
        'year': 'year',
        'Country name': 'country',
        "Life Ladder": 'life_ladder',
        "Social support": 'social_support',
        "Freedom to make life choices": "freedom_to_make_life_choices",
        "Generosity": "generosity",
        "Perceptions of corruption": "perceptions_of_corruption",
        "Positive affect": "positive_affect",
        "Negative affect": "negative_affect",
    },
    DataFramesCSV.MEAT_CONSUMPTION_CSV: {
        'Code': 'country_code',
        'Year': 'year',
        "Meat, poultry | 00002734 || Food available for consumption | 0645pc || kilograms per year per capita": "poultry",
        "Meat, beef | 00002731 || Food available for consumption | 0645pc || kilograms per year per capita": "beef",
        "Meat, sheep and goat | 00002732 || Food available for consumption | 0645pc || kilograms per year per capita": "sheep_and_goat",
        "Meat, pig | 00002733 || Food available for consumption | 0645pc || kilograms per year per capita": "pig",
        "Fish and seafood | 00002960 || Food available for consumption | 0645pc || kilograms per year per capita": "fish_and_seafood",
    },
    DataFramesCSV.COUNTRY_MASTER_CSV: {
        'alpha-3': 'country_code',
        'name': 'country'
    },
    DataFramesCSV.HUNGER_CSV: {
        'Entity': 'country',
        'Year': 'year',
        'Global Hunger Index (2021)': 'hunger_index',
    },
    DataFramesCSV.SMOKING_CSV: {
        'Entity': 'country',
        'Year': 'year',
        'Prevalence of current tobacco use (% of adults)': 'prevalence_smoking',
    },
    DataFramesCSV.ALCOHOL_CONSUMPTION_CSV: {
        'Entity': 'country',
        'Year': 'year',
        'liters_of_pure_alcohol_per_capita': 'liters_of_pure_alcohol_per_capita',
    },
}


def capture_console_output_to_image(name_file: str, data_to_print):
    # Redirect console output to a StringIO buffer
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    print(data_to_print)

    # Get the console output
    console_output = sys.stdout.getvalue()

    # Reset the standard output
    sys.stdout = old_stdout

    # Create a Matplotlib figure and save the console output as an image
    fig, ax = plt.subplots()
    ax.text(0.1, 0.5, console_output, fontsize=12, va='center')
    ax.axis('off')
    plt.savefig(f'{PATH_IMAGES}{name_file}.png', bbox_inches='tight', dpi=200)
    plt.close()


class DataFramePreviousFieldNameOptions(enum.Enum):
    IS_NULL = 'isnull'
    D_TYPES = 'dtypes'
    COUNT = 'count'


def capture_get_dataframe_info_image(
        table_name: str,
        name_file: str,
        dataframe: pd.DataFrame,
        previous_data: pd.Series = None,
        previous_data_name: DataFramePreviousFieldNameOptions = None,
        figure_size_height=10.0,
        figure_size_width=5.0,
):
    info_object = {
        'columns': dataframe.columns.str[0:30].tolist(),
        'dtypes': dataframe.dtypes.tolist(),
        'count': dataframe.count().tolist(),
        'isnull': dataframe.isnull().sum().tolist(),
    }
    dataframe_info = pd.DataFrame(info_object)

    if previous_data is not None:
        current_data = dataframe_info[previous_data_name.value]
        dataframe_info[f'old {previous_data_name.value}'] = previous_data
        dataframe_info['change'] = previous_data - current_data

    fig, ax = plt.subplots(figsize=(figure_size_width, figure_size_height))
    ax.axis('off')
    ax.axis('tight')
    dataframe_info.reset_index(inplace=True)
    col_widths = [0.08, 0.35, 0.15, 0.15, 0.15, 0.15, 0.15]
    data_table = ax.table(
        cellText=dataframe_info.values,
        colLabels=[' '.join(col.split('_')) for col in dataframe_info.columns],
        colWidths=col_widths,
        loc='center'
    )

    if previous_data is not None:
        for (i, j), val in np.ndenumerate(dataframe_info.values):
            if j == 6 and val != 0:  # We look into the second column (j==1), and search for zero values
                data_table[(i + 1, j)].set_facecolor("red")
                data_table[(i + 1, j)].set_text_props(color='white', weight='bold')

    num_rows = dataframe.shape[0] - 1
    data_table.auto_set_font_size(False)
    data_table.set_fontsize(6)
    plt.title(f'{table_name} (Records: {num_rows})')
    plt.tight_layout()
    plt.savefig(f"{PATH_IMAGES}{name_file}.png", dpi=200, bbox_inches='tight')
    plt.close()


def capture_table_dataframe_image(
        table_name: str,
        name_file: str,
        col_widths: List[float],
        dataframe: pd.DataFrame,
        figure_size_height=10.0,
        figure_size_width=5.0,
        font_size=4,
        head=None,
        show_records=True,
):
    fig, ax = plt.subplots(figsize=(figure_size_width, figure_size_height))
    ax.axis('off')
    ax.axis('tight')
    dataframe.reset_index(inplace=True)
    data_table = ax.table(
        cellText=dataframe.head(head).values if head is not None else dataframe.values,
        colLabels=[' '.join(col.split('_')) for col in dataframe.columns],
        colWidths=col_widths,
        loc='center'
    )
    num_rows = dataframe.shape[0] - 1
    records = f'(Records: {num_rows})' if show_records else ''
    plt.title(f'{table_name} {records}')
    data_table.auto_set_font_size(False)
    data_table.set_fontsize(font_size)
    fig.tight_layout()

    plt.savefig(f"{PATH_IMAGES}{name_file}.png", dpi=200, bbox_inches='tight')
    plt.close()


def capture_confusion_matrix_image(
        table_name: str,
        name_file: str,
        dataframe: pd.DataFrame,
        figure_size_height=10.0,
        figure_size_width=5.0,
        font_size=4,
        head=None,
        accuracy: float = None,
        precision: float = None,
        recall: float = None,
        f1_score: float = None,
        best_params=None,
        best_training_accuracy=None,
        best_test_accuracy=None,
        tuning_scoring='precision',
):
    fig, (ax, ax2) = plt.subplots(figsize=(figure_size_width, figure_size_height), nrows=2)
    ax.axis('off')
    ax.axis('tight')
    dataframe.reset_index(inplace=True)
    df_cm = ax.table(
        cellText=dataframe.head(head).values if head is not None else dataframe.values,
        colLabels=[' '.join(col.split('_')) for col in dataframe.columns],
        colWidths=[0.2, 0.25, 0.25],
        loc='center'
    )
    ax.set_title(table_name)
    df_cm.auto_set_font_size(False)
    df_cm.set_fontsize(font_size)

    ax2.axis('off')
    ax2.axis('tight')
    if best_params:
        df_best_parameters = pd.DataFrame(list(best_params.items()), columns=['parameter', 'best'])
        data_table_best_params = ax2.table(
            cellText=df_best_parameters.head(head).values if head is not None else df_best_parameters.values,
            colLabels=[' '.join(col.split('_')) for col in df_best_parameters.columns],
            colWidths=[0.5, 0.3],
            loc='center',
        )
        ax2.set_title(f"Tuning Process for {tuning_scoring}:")
        data_table_best_params.auto_set_font_size(False)
        data_table_best_params.set_fontsize(font_size)

    increase_lines = - 0.13
    current_line = -0.2
    font_size_text = font_size + 1
    text_h_pos_m_r = 0.4
    text_h_pos_m_t = 1

    plt.text(
        text_h_pos_m_r,
        current_line,
        'Prediction Metrics:',
        ha='right',
        va='center',
        transform=plt.gca().transAxes,
        fontsize=font_size_text + 2,
    )

    plt.text(
        text_h_pos_m_t,
        current_line,
        'Training Metrics:',
        ha='right',
        va='center',
        transform=plt.gca().transAxes,
        fontsize=font_size_text + 2,
    )
    current_line = current_line + increase_lines
    if best_training_accuracy is not None:
        plt.text(
            text_h_pos_m_t, current_line, f'Best Training Accuracy: {round(best_training_accuracy * 100, 2)}%',
            ha='right',
            va='center',
            transform=plt.gca().transAxes,
            fontsize=font_size_text,
        )

    if accuracy is not None:
        plt.text(
            text_h_pos_m_r, current_line, f'Accuracy: {round(accuracy * 100, 2)}%',
            ha='right',
            va='center',
            transform=plt.gca().transAxes,
            fontsize=font_size_text,
        )
    current_line = current_line + increase_lines
    if precision is not None:
        plt.text(
            text_h_pos_m_r, current_line, f'Precision: {round(precision * 100, 2)}%',
            ha='right',
            va='center',
            transform=plt.gca().transAxes,
            fontsize=font_size_text,
        )

    if best_test_accuracy is not None:
        plt.text(
            text_h_pos_m_t, current_line, f'Best Test Accuracy: {round(best_test_accuracy * 100, 2)}%',
            ha='right',
            va='center',
            transform=plt.gca().transAxes,
            fontsize=font_size_text,
        )

    current_line = current_line + increase_lines
    if recall is not None:
        plt.text(
            text_h_pos_m_r, current_line, f'Recall: {round(recall * 100, 2)}%',
            ha='right',
            va='center',
            transform=plt.gca().transAxes,
            fontsize=font_size_text,
        )
    current_line = current_line + increase_lines
    if f1_score is not None:
        plt.text(
            text_h_pos_m_r, current_line, f'F1-score: {round(f1_score * 100, 2)}%',
            ha='right',
            va='center',
            transform=plt.gca().transAxes,
            fontsize=font_size_text,
        )

    # fig.tight_layout()
    plt.savefig(f"{PATH_IMAGES}{name_file}.png", dpi=200, bbox_inches='tight')
    plt.close()


def capture_summary_dataset_to_image(
        name_file: str,
        dataset: pd.DataFrame,
        dataset_name: str,
        figure_size_height=3,
        font_size=7,
):
    desc = dataset.describe().round(2).T

    fig, ax = plt.subplots(figsize=(7, figure_size_height))
    new_order = ['min', '25%', '50%', '75%', 'max', 'mean', 'count', 'std']
    desc = desc[new_order]
    # Hide axes
    ax.axis('off')
    ax.axis('tight')
    col_widths = [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08]

    data_table = ax.table(
        cellText=desc.values,
        colLabels=desc.columns,
        rowLabels=[name[:30] for name in desc.index],
        cellLoc='center',
        loc='center',
        colWidths=col_widths,
    )
    data_table.auto_set_font_size(False)
    data_table.set_fontsize(font_size)
    fig.tight_layout()

    plt.title(f"Descriptive Statistics {dataset_name}")
    plt.savefig(f"{PATH_IMAGES}{name_file}.png", dpi=200, bbox_inches='tight')
    plt.close()


def du_data_exploration_basics(
        dataset: pd.DataFrame,
        metric_name_plot: str,
        metric_label: str,
        dataset_name: str,
        country_label: str = 'country',
        prefix_file_name: str = 'du',
        year_label: str = 'year',
):
    metric_name_file = '_'.join([word[0:2] for word in metric_label.split('_')])
    base_name_file = f'{prefix_file_name}_{dataset_name}_{metric_name_file}'
    base_name_file_with_path = f'{PATH_IMAGES}{base_name_file}'

    fig, (ax1, ax2) = plt.subplots(figsize=(20, 10), nrows=2, ncols=1)
    dataset.boxplot(column=metric_label, by=year_label, grid=False, ax=ax1, rot=90)
    ax1.set_title(f'Yearly Spread of {metric_name_plot}')
    ax1.set_xlabel('Year')
    ax1.set_ylabel(metric_name_plot)

    average_obesity_per_year = dataset.groupby(year_label)[metric_label].mean()
    average_obesity_per_year.plot(linestyle='-', marker='o', color='b', label=f'Average {metric_name_plot}', ax=ax2)
    ax2.set_title(f'Average {metric_name_plot} Over Years')
    ax2.set_ylabel(metric_name_plot)
    ax2.set_xlabel('Year')

    plt.suptitle('')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{base_name_file_with_path}_y_trend.png', bbox_inches='tight', dpi=200)
    plt.close()

    plt.figure(figsize=(10, 3))
    dataset[metric_label].hist(bins=30, edgecolor='black', alpha=0.7)
    plt.title(f'Distribution of {metric_name_plot}')
    plt.xlabel(metric_name_plot)
    plt.ylabel('Frequency')
    plt.savefig(f'{base_name_file_with_path}_freq.png', bbox_inches='tight', dpi=200)
    plt.close()

    ##########

    dataset_grouped_by_country_mean = dataset.groupby(country_label)[metric_label].mean()

    top_countries_index = dataset_grouped_by_country_mean.nlargest(20).index
    top_countries_filtered_data = dataset[dataset[country_label].isin(top_countries_index)]
    smallest_countries_index = dataset_grouped_by_country_mean.nsmallest(20).index
    smallest_countries_filtered_data = dataset[dataset[country_label].isin(smallest_countries_index)]

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(18, 12))
    fig.tight_layout(pad=10.0)

    top_countries_filtered_data.boxplot(column=metric_label, by=country_label, ax=ax1, rot=90)
    ax1.set_title(f'{metric_name_plot} for Top 20 Countries')
    ax1.set_xlabel('Country')
    ax1.set_ylabel(metric_name_plot)

    smallest_countries_filtered_data.boxplot(column=metric_label, by=country_label, ax=ax2, rot=90)
    ax2.set_title(f'{metric_name_plot} for Lowest 20 Countries')
    ax2.set_xlabel('Country')
    ax2.set_ylabel(metric_name_plot)

    plt.suptitle('')
    plt.savefig(f'{base_name_file_with_path}_cou_t_l_20.png', bbox_inches='tight', dpi=200)
    plt.close()

    dataset_top_lowest = pd.concat([top_countries_filtered_data, smallest_countries_filtered_data])
    plt.figure(figsize=(20, 15))
    avg_obesity_per_country = dataset_top_lowest.groupby(country_label)[metric_label].mean().sort_values()
    avg_obesity_per_country.plot(kind='barh', color='skyblue')
    plt.title(f'Average {metric_name_plot} by Country 20 top and 20 lowest')
    plt.xlabel(metric_name_plot)
    plt.ylabel('Country')
    plt.savefig(f'{base_name_file_with_path}_cou_t_l_20_v2.png', bbox_inches='tight', dpi=200)
    plt.close()


class AvailableModels(enum.Enum):
    RANDOM_FOREST = 'Random Forest'
    DECISION_TREE = 'Decision Tree'
    LOGISTIC_REGRESSION = 'Logistic Regression'
    SVM = 'SVM'
    BERNOULLI_NB = 'Bernoulli NB'
    K_NEIGHBORS = 'K Neighbors'
    MLP = 'MLP'
    GRADIENT_BOOSTING = 'Gradient Boosting'
    TESTING = 'Testing'


@dataclass
class ModelFitter:
    dataset: pd.DataFrame
    label: str
    models_to_fit: List[AvailableModels]
    predictions_models: Dict[AvailableModels, pd.Series] = None
    tuning: bool = False
    tuning_scoring: str = 'accuracy'
    _x_train: pd.DataFrame = None
    _y_train: pd.DataFrame = None
    _x_test: pd.Series = None
    _y_test: pd.Series = None

    def __post_init__(self):
        X = self.dataset.drop(columns=[self.label])
        y = self.dataset[self.label]
        self._x_train, self._x_test, self._y_train, self._y_test = model_selection.train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )
        self.predictions_models = {}

    def _best_model_params(self, params, classifier):
        print("TUNING...")
        grid = model_selection.GridSearchCV(
            estimator=classifier,
            param_grid=params,
            scoring=self.tuning_scoring,
            n_jobs=-1,
            cv=5
        )
        grid.fit(X=self._x_train, y=self._y_train)

        best_gbc = grid.best_estimator_
        training_accuracy = best_gbc.score(self._x_train, self._y_train)
        test_accuracy = best_gbc.score(self._x_test, self._y_test)
        best_params = grid.best_params_
        return best_params, training_accuracy, test_accuracy, grid.best_estimator_

    def fit_models(self) -> None:
        fit_function_mapper = {
            AvailableModels.LOGISTIC_REGRESSION: self._fit_logistic_regression,
            AvailableModels.DECISION_TREE: self._fit_decision_tree,
            AvailableModels.RANDOM_FOREST: self._fit_random_forest,
            AvailableModels.SVM: self._fit_svm,
            AvailableModels.BERNOULLI_NB: self._fit_bernoulli_nb,
            AvailableModels.K_NEIGHBORS: self._fit_k_neighbors,
            AvailableModels.TESTING: self._fit_testing,
            AvailableModels.MLP: self._fit_mlp,
            AvailableModels.GRADIENT_BOOSTING: self._fit_gradient_boosting,
        }
        for model_enum in self.models_to_fit:
            print(f"Fitting : {model_enum.value}...")
            fit_function_mapper.get(model_enum)()

            accuracy = np.mean(self.predictions_models[model_enum] == self._y_test)
            print(f"Accuracy {model_enum.value}: {accuracy}")

    def generate_confusion_matrix(
            self,
            enum_model: AvailableModels,
            best_params=None,
            best_training_accuracy=None,
            best_test_accuracy=None,
    ):
        y_pred = self.predictions_models[enum_model]
        cm = metrics.confusion_matrix(self._y_test, y_pred)
        df_cm = pd.DataFrame(cm, columns=["Predicted: 0", "Predicted: 1"], index=["Actual: 0", "Actual: 1"])
        model_name = enum_model.value
        accuracy = metrics.accuracy_score(self._y_test, y_pred)
        recall = metrics.recall_score(self._y_test, y_pred)
        precision_score = metrics.precision_score(self._y_test, y_pred)
        f1_score = metrics.f1_score(self._y_test, y_pred)
        capture_confusion_matrix_image(
            table_name=f'Confusion Matrix: {model_name}',
            dataframe=df_cm,
            font_size=6,
            name_file=f"dm_confu_mat_{'_'.join([word.lower()[0:4] for word in model_name.split(' ')])}",
            figure_size_width=4,
            figure_size_height=3,
            accuracy=accuracy,
            precision=precision_score,
            recall=recall,
            f1_score=f1_score,
            best_params=best_params,
            best_training_accuracy=best_training_accuracy,
            best_test_accuracy=best_test_accuracy,
        )

    def _predictor_resume_logistic_regression(self, fitted_classifier):
        best_rfe = fitted_classifier.named_steps['Feature_Selection']
        selected_features = best_rfe.support_
        feature_ranking = best_rfe.ranking_

        feature_selection_df = pd.DataFrame({
            'Feature_Name': self._x_test.columns,
            'Selected': selected_features,
            'Ranking': feature_ranking
        })

        best_logreg = fitted_classifier.named_steps['Model']

        importance_feature_df = pd.DataFrame({
            'Feature': feature_selection_df[feature_selection_df['Selected'] == True]['Feature_Name'],
            'Importance': best_logreg.coef_[0]
        })

        capture_table_dataframe_image(
            dataframe=feature_selection_df.sort_values(by='Ranking'),
            table_name='Feature Selection: Logistic Regresion',
            font_size=5,
            name_file='dm_featu_sele_log_regr',
            col_widths=[0.3, 1, 0.5, 0.5],
            figure_size_height=3.5,
            show_records=False
        )

        capture_table_dataframe_image(
            dataframe=importance_feature_df.sort_values(by='Importance'),
            table_name='Feature Importance: Logistic Regresion',
            font_size=5,
            name_file='dm_featu_imp_logi_regr',
            col_widths=[0.2, 0.8, 0.6],
            figure_size_height=1,
            show_records=False
        )

        # r2 = metrics.r2_score(self._y_test, y_pred)
        # print(f"\nR^2 Score: {r2:.4f}")
        #
        # data = dict(
        #     coefficient=self._x_test.columns,
        #     values=best_logreg.coef_[0]
        # )
        #
        # # print(f'Intercept: {intercept}')
        # df = pd.DataFrame(data)
        # df.sort_values(by='values', ascending=False, inplace=True)
        # print(df)

    def _predictor_resume_gradient_boosting(self, fitted_classifier):
        feature_importances = fitted_classifier.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': self._x_test.columns,
            'Importance': feature_importances
        })
        feature_importance_df = feature_importance_df.round(5)
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        capture_table_dataframe_image(
            dataframe=feature_importance_df,
            table_name='Feature Importance: Gradient Boosting',
            font_size=5,
            name_file='dm_featu_imp_grad_boost',
            col_widths=[0.2, 1.2, 0.6],
            figure_size_height=3.5,
            figure_size_width=8,
            show_records=False
        )

    def _predictor_resume_decision_tree(self, fitted_classifier):
        feature_importances = fitted_classifier.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': self._x_test.columns,
            'Importance': feature_importances
        })
        feature_importance_df = feature_importance_df.round(5)
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        capture_table_dataframe_image(
            dataframe=feature_importance_df,
            table_name='Feature Importance: Decision Tree',
            font_size=5,
            name_file='dm_featu_imp_deci_tree',
            col_widths=[0.2, 1.2, 0.6],
            figure_size_height=3.5,
            figure_size_width=8,
            show_records=False
        )
        plt.figure(figsize=(20, 10))
        tree.plot_tree(fitted_classifier, filled=True,
                       class_names=['Fail', 'Obesity Target'], rounded=True)
        plt.savefig(f"{PATH_IMAGES}decision_tree_tree.png", dpi=500, bbox_inches='tight')

    def _predictor_resume_bernoulli_nb(self, fitted_classifier):
        feature_log_prob = fitted_classifier.feature_log_prob_
        feature_prob = np.exp(feature_log_prob)
        feature_importance_df = pd.DataFrame({
            'Feature': self._x_test.columns,
            'Importance': feature_prob[0]
        })
        feature_importance_df = feature_importance_df.round(5)
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        capture_table_dataframe_image(
            dataframe=feature_importance_df,
            table_name='Feature Importance: Bernoulli NB',
            font_size=5,
            name_file='dm_featu_imp_bern_nb',
            col_widths=[0.2, 1.2, 0.6],
            figure_size_height=3.5,
            figure_size_width=8,
            show_records=False
        )

    def _predictor_resume_random_forest(self, fitted_classifier):
        feature_importances = fitted_classifier.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': self._x_test.columns,
            'Importance': feature_importances
        })
        feature_importance_df = feature_importance_df.round(5)
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        capture_table_dataframe_image(
            dataframe=feature_importance_df,
            table_name='Feature Importance: Random Forest',
            font_size=5,
            name_file='dm_featu_imp_random_forest',
            col_widths=[0.2, 1.2, 0.6],
            figure_size_height=3.5,
            figure_size_width=8,
            show_records=False
        )

    def _predictor_resume_svc(self, fitted_classifier):
        feature_importances = abs(fitted_classifier.coef_[0])
        feature_importance_df = pd.DataFrame({
            'Feature': self._x_test.columns,
            'Importance': feature_importances
        })
        feature_importance_df = feature_importance_df.round(5)
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        capture_table_dataframe_image(
            dataframe=feature_importance_df,
            table_name='Feature Importance: SVM Model',
            font_size=5,
            name_file='dm_featu_imp_svm',
            col_widths=[0.2, 1.2, 0.6],
            figure_size_height=3.5,
            figure_size_width=8,
            show_records=False
        )

    def _fit_predict_classifier(
            self,
            classifier,
            enum_model: AvailableModels,
            best_params=None,
            best_training_accuracy: float = None,
            best_test_accuracy: float = None,
    ):
        classifier.fit(X=self._x_train, y=self._y_train)
        mapper_resume_function = {
            enum_model.LOGISTIC_REGRESSION: self._predictor_resume_logistic_regression,
            enum_model.RANDOM_FOREST: self._predictor_resume_random_forest,
            enum_model.GRADIENT_BOOSTING: self._predictor_resume_gradient_boosting,
            enum_model.DECISION_TREE: self._predictor_resume_decision_tree,
            enum_model.BERNOULLI_NB: self._predictor_resume_bernoulli_nb,
            # enum_model.SVM: self._predictor_resume_svc,
        }
        self.predictions_models[enum_model] = classifier.predict(self._x_test)
        resume_function = mapper_resume_function.get(enum_model, None)
        if resume_function is not None:
            resume_function(fitted_classifier=classifier)
        self.generate_confusion_matrix(
            enum_model=enum_model,
            best_params=best_params,
            best_training_accuracy=best_training_accuracy,
            best_test_accuracy=best_test_accuracy,
        )

    def _fit_decision_tree(self) -> None:
        params = {
            'criterion': [
                'entropy',
                'gini',
                'log_loss',
            ],
            'class_weight': ["balanced", None],
            'max_depth': [10, 20, 30, 40],
            'min_samples_split': [2, 10],
            'min_samples_leaf': [1, 8],
        }
        classifier = tree.DecisionTreeClassifier()
        best_params, training_accuracy, test_accuracy = None, None, None
        if self.tuning:
            best_params, training_accuracy, test_accuracy, classifier = self._best_model_params(
                params=params,
                classifier=classifier
            )
        self._fit_predict_classifier(
            classifier=classifier,
            enum_model=AvailableModels.DECISION_TREE,
            best_params=best_params,
            best_training_accuracy=training_accuracy,
            best_test_accuracy=test_accuracy,
        )

    def _fit_logistic_regression(self) -> None:
        classifier = linear_model.LogisticRegression(max_iter=100000)
        rfe = feature_selection.RFE(estimator=classifier, step=1)
        pipelines = pipeline.Pipeline(steps=[('Feature_Selection', rfe), ('Model', classifier)])
        param_grid = {
            'Feature_Selection__n_features_to_select': [1, 2, 3],
            'Model__C': [0.01, 0.1, 1, 10],
            'Model__penalty': ['l1', 'l2'],
            'Model__solver': ['lbfgs'],
            'Model__class_weight': ['balanced', None],
        }

        best_params, training_accuracy, test_accuracy, classifier = self._best_model_params(
            classifier=pipelines,
            params=param_grid
        )

        self._fit_predict_classifier(
            classifier=classifier,
            enum_model=AvailableModels.LOGISTIC_REGRESSION,
            best_params=best_params,
            best_training_accuracy=training_accuracy,
            best_test_accuracy=test_accuracy,
        )

    def _fit_gradient_boosting(self) -> None:
        classifier = ensemble.GradientBoostingClassifier(
            random_state=42
        )
        best_params, training_accuracy, test_accuracy = None, None, None
        if self.tuning:
            best_params, training_accuracy, test_accuracy, classifier = self._best_model_params(
                classifier=classifier,
                params={
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1, 0.5, 1],
                    'max_depth': [3, 4, 5],
                    'min_samples_split': [2, 3, 4],
                    'min_samples_leaf': [1, 2, 3],
                    'subsample': [0.8, 0.9, 1]
                }
            )
        self._fit_predict_classifier(
            classifier=classifier,
            enum_model=AvailableModels.GRADIENT_BOOSTING,
            best_params=best_params,
            best_test_accuracy=test_accuracy,
            best_training_accuracy=training_accuracy,
        )

    def _fit_random_forest(self) -> None:
        classifier = ensemble.RandomForestClassifier(
            n_jobs=-1
        )
        best_params, training_accuracy, test_accuracy = None, None, None
        if self.tuning:
            best_params, training_accuracy, test_accuracy, classifier = self._best_model_params(
                classifier=classifier,
                params={
                    "class_weight": ['balanced', 'balanced_subsample', None],
                    "n_estimators": [100, 300, 500, 700, 1000],
                    "max_depth": [1, 3, 5, 10]
                },
            )
        self._fit_predict_classifier(
            classifier=classifier,
            enum_model=AvailableModels.RANDOM_FOREST,
            best_params=best_params,
            best_test_accuracy=test_accuracy,
            best_training_accuracy=training_accuracy,
        )

    def _fit_svm(self) -> None:
        classifier = svm.SVC(random_state=42)
        best_params, training_accuracy, test_accuracy = None, None, None
        if self.tuning:
            best_params, training_accuracy, test_accuracy, classifier = self._best_model_params(
                classifier=classifier,
                params={
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf'],
                    'degree': [1, 2],
                    'gamma': ['scale', 'auto']
                },
            )
        self._fit_predict_classifier(
            classifier=classifier,
            enum_model=AvailableModels.SVM,
            best_params=best_params,
            best_test_accuracy=test_accuracy,
            best_training_accuracy=training_accuracy,
        )

    def _fit_bernoulli_nb(self) -> None:
        classifier = naive_bayes.BernoulliNB()
        best_params, training_accuracy, test_accuracy = None, None, None
        if self.tuning:
            best_params, training_accuracy, test_accuracy, classifier = self._best_model_params(
                classifier=classifier,
                params={
                    'alpha': [0.0, 0.5, 1.0],
                    'binarize': [0.0, 0.5, 1.0],
                    'fit_prior': [True, False]
                },
            )
        self._fit_predict_classifier(
            classifier=classifier,
            enum_model=AvailableModels.BERNOULLI_NB,
            best_params=best_params,
            best_test_accuracy=test_accuracy,
            best_training_accuracy=training_accuracy,
        )

    def _fit_k_neighbors(self) -> None:
        classifier = neighbors.KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
        best_params, training_accuracy, test_accuracy, classifier = self._best_model_params(
            classifier=classifier,
            params={
                "n_neighbors": [1, 2, 3, 4],
                "n_estimators": [100, 300, 500, 700, 1000],
                "max_depth": [1, 3, 5, 10]
            },
        )
        self._fit_predict_classifier(
            classifier=classifier,
            enum_model=AvailableModels.K_NEIGHBORS,
            best_params=best_params,
            best_training_accuracy=training_accuracy,
            best_test_accuracy=test_accuracy,
        )

    def _fit_testing(self) -> None:
        classifier = neighbors.KNeighborsClassifier(n_neighbors=3)
        self._fit_predict_classifier(classifier=classifier, enum_model=AvailableModels.TESTING)

    def _fit_mlp(self) -> None:
        classifier = neural_network.MLPClassifier(
            hidden_layer_sizes=(10, 10),
            max_iter=1000,
            activation='relu',
            solver='adam',
            random_state=42
        )
        self._fit_predict_classifier(classifier=classifier, enum_model=AvailableModels.MLP)
