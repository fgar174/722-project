import random
import asyncio
import sys

from typing import List

import pandas as pd

from dataclasses import dataclass

from matplotlib import pyplot as plt
from sklearn import ensemble

import project_helper as pj

random.seed(722)

PATH_IMAGES = '../tex/iterations/iteration_4/images/'


@dataclass
class ProjectManager:
    missing_countries: pd.DataFrame = None
    country_master: pd.DataFrame = None
    integrated_dataset: pd.DataFrame = None
    generate_images_du_02: bool = False

    def __post_init__(self):
        country_master = pd.read_csv(pj.DataFramesCSV.COUNTRY_MASTER_CSV.value)
        country_master = country_master[["alpha-3", "name"]]
        country_master.rename(
            inplace=True,
            columns={
                'alpha-3': 'country_code',
                'name': 'country'
            }
        )
        self.country_master = country_master

    def _function_mapper_du_02(self, dataset: pj.DataFramesCSV or pj.DataFramesXLSX):
        mapper = {
            pj.DataFramesCSV.MEAT_CONSUMPTION_CSV: self._du_02_meat_consumption,
            pj.DataFramesCSV.WHO_OBESITY_CSV: self._du_02_obesity,
            pj.DataFramesCSV.HUNGER_CSV: self._du_02_hunger,
            pj.DataFramesCSV.SMOKING_CSV: self._du_02_smoking,
            pj.DataFramesCSV.ALCOHOL_CONSUMPTION_CSV: self._du_02_alcohol_consumption,
            pj.DataFramesXLSX.HAPPINESS_REPORT_XLSX: self._du_02_happiness,
        }

        return mapper.get(dataset)

    def _capture_get_dataframe_info_image(
            self,
            table_name: str,
            name_file: str,
            dataframe: pd.DataFrame,
            figure_size_height=10.0,
            figure_size_width=5.0,
            force_save_image=False,
            previous_data: pd.Series = None,
            previous_data_name: pj.DataFramePreviousFieldNameOptions = None,
    ):
        if self.generate_images_du_02 or force_save_image:
            pj.capture_get_dataframe_info_image(
                table_name=table_name,
                name_file=name_file,
                dataframe=dataframe,
                figure_size_height=figure_size_height,
                figure_size_width=figure_size_width,
                previous_data_name=previous_data_name,
                previous_data=previous_data,
            )

    def _capture_table_dataframe_image(
            self,
            table_name: str,
            name_file: str,
            col_widths: List[float],
            dataframe: pd.DataFrame,
            figure_size_height=10.0,
            figure_size_width=5.0,
            font_size=4,
            head=None
    ):
        if self.generate_images_du_02:
            pj.capture_table_dataframe_image(
                table_name=table_name,
                name_file=name_file,
                col_widths=col_widths,
                dataframe=dataframe,
                figure_size_width=figure_size_width,
                figure_size_height=figure_size_height,
                font_size=font_size,
                head=head
            )

    def _capture_summary_dataset_to_image(
            self,
            name_file: str,
            dataset: pd.DataFrame,
            dataset_name: str,
            figure_size_height=3,
            font_size=7,
    ):
        if self.generate_images_du_02:
            pj.capture_summary_dataset_to_image(
                name_file=name_file,
                dataset=dataset,
                dataset_name=dataset_name,
                figure_size_height=figure_size_height,
                font_size=font_size,
            )

    def _du_data_exploration_basics(
            self,
            dataset: pd.DataFrame,
            metric_name_plot: str,
            metric_label: str,
            dataset_name: str,
            country_label: str = 'country',
            prefix_file_name: str = 'du',
            year_label: str = 'year',
    ):
        if self.generate_images_du_02:
            pj.du_data_exploration_basics(
                dataset_name=dataset_name,
                dataset=dataset,
                metric_label=metric_label,
                metric_name_plot=metric_name_plot,
                country_label=country_label,
                prefix_file_name=prefix_file_name,
                year_label=year_label,
            )

    def get_crosstab_missing_countries(self, without_replacement=False, save_table=False, head=20) -> pd.DataFrame:
        filtered_dataframe = self.missing_countries
        if without_replacement:
            filtered_dataframe = filtered_dataframe[
                filtered_dataframe["replacement"].apply(lambda x: x is None)
            ]
        missing_countries = pd.crosstab(
            filtered_dataframe["missing"],
            filtered_dataframe["dataset"],
            margins=True
        )
        if save_table:
            if head:
                file_name = f'du_missing_countries_per_dataset_head_{head}'
            else:
                file_name = 'du_missing_countries_per_dataset'
            self._capture_table_dataframe_image(
                table_name='Missing countries by datasets (20 firsts)',
                dataframe=missing_countries,
                col_widths=[0.35, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.07],
                name_file=file_name,
                head=head,
                font_size=5,
                figure_size_height=4.5
            )
        return missing_countries

    def _generate_missing_countries_replacement_table(self):
        # Remove groups of countries
        self.missing_countries = self.missing_countries[
            ~self.missing_countries["missing"].str.startswith('OWID', na=False)
        ]
        self.missing_countries = self.missing_countries[
            ~self.missing_countries["missing"].str.contains('WB', na=False)
        ]
        # Remove groups of countries by income
        self.missing_countries = self.missing_countries[
            ~self.missing_countries["missing"].str.contains('income', na=False)
        ]
        # Remove some that are not countries, or have been dissolved, or are present in just one database
        excluded_countries = [
            'European Union (27)',
            'World',
            'Somaliland region',
            'Kosovo',  # Present in just one database
            'Sudan (former)',  # Present in just one database
            'ANT',  # Was dissolved in 2010
            'CIV',  # Present in just one database
        ]
        self.missing_countries = self.missing_countries[
            ~self.missing_countries["missing"].isin(excluded_countries)
        ]

        # Add the replacement with countries
        missing_replacement = [
            ('Bosnia and Herzegovina', 'Bosnia And Herzegovina'),
            ('Czechia', 'Czech Republic'),
            ('North Macedonia', 'Macedonia'),
            ('Guinea-Bissau', 'Guinea Bissau'),
            ('Cape Verde', 'Cabo Verde'),
            ('East Timor', 'Timor-Leste'),
            ('North Korea', 'Korea'),
            ('Democratic Republic of Congo', 'Congo (Democratic Republic Of The)'),
            ('Brunei', 'Brunei Darussalam'),
            ('Viet Nam', 'Vietnam'),
            ('Hong Kong S.A.R. of China', 'Hong Kong'),
            ("Lao People's Democratic Republic", 'Laos'),
            ('United States of America', 'United States'),
            ('Democratic Republic of the Congo', 'Congo (Democratic Republic Of The)'),
            ('Congo (Kinshasa)', 'Congo'),
            ('Syrian Arab Republic', 'Syria'),
            ('United Kingdom of Great Britain and Northern Ireland', 'United Kingdom'),
            ('Bolivia (Plurinational State of)', 'Bolivia'),
            ('Ethiopia (former)', 'Ethiopia'),
            ('State of Palestine', 'Palestine'),
            ('Taiwan Province of China', 'Taiwan'),
            ('Turkiye', 'Turkey'),
            ('Türkiye', 'Turkey'),
            ("Democratic People's Republic of Korea", 'Korea'),
            ('United Republic of Tanzania', 'Tanzania'),
            ('Iran (Islamic Republic of)', 'Iran'),
            ('Republic of Moldova', 'Moldova'),
            ('Venezuela (Bolivarian Republic of)', 'Venezuela'),
            ('Russian Federation', 'Russia'),
            ('Micronesia (country)', 'Micronesia (Federated States of)'),
            ('Republic of Korea', 'Korea'),
            ("Cote de Ivoire", "Côte D'Ivoire"),
            ("Cote d'Ivoire", "Côte D'Ivoire"),
            ("Côte d'Ivoire", "Côte D'Ivoire"),
            ('Ivory Coast', "Côte D'Ivoire"),
        ]

        for missing, replacement in missing_replacement:
            self.missing_countries.loc[self.missing_countries["missing"] == missing, "replacement"] = replacement

        self._capture_table_dataframe_image(
            table_name='Table of replacement for missing countries (First 20)',
            dataframe=self.missing_countries,
            name_file='dp_missing_countries_replacement',
            col_widths=[0.1, 0.25, 0.15, 0.15, 0.25],
            head=20,
            figure_size_height=5,
            figure_size_width=6
        )

    def _merge_by_country_code(self, dataset_name, target_dataset: pd.DataFrame):
        not_found = target_dataset.merge(
            self.country_master,
            on='country_code',
            how='outer',
            indicator=True
        )
        not_found = not_found[not_found._merge == "left_only"][['country_code']].squeeze()
        not_found = not_found.unique()

        data = {
            'missing': not_found,
            'dataset': [dataset_name for _ in range(len(not_found))],
            'value': 1,
            'replacement': None,
        }
        self.missing_countries = pd.concat(
            [self.missing_countries, pd.DataFrame(data)],
            ignore_index=True
        )

    def _merge_by_country_name(self, dataset_name, target_dataset: pd.DataFrame):
        not_found = target_dataset.merge(
            self.country_master,
            on='country',
            how='outer',
            indicator=True
        )
        not_found = not_found[not_found._merge == "left_only"][['country']].squeeze()
        not_found = not_found.unique()
        data = {
            'missing': not_found,
            'dataset': [dataset_name for _ in range(len(not_found))],
            'value': 1,
            'replacement': None,
        }

        self.missing_countries = pd.concat(
            [self.missing_countries, pd.DataFrame(data)],
            ignore_index=True
        )

    # def _set_replacement_for_missing_countries_by_code(self):
    def _find_missing_countries(
            self,
            target_dataset_path: (pj.DataFramesCSV, pj.DataFramesXLSX),
            merged_by_in_country_master: str = 'country',
    ):
        missing_countries = self.missing_countries.rename(
            columns={
                'missing': merged_by_in_country_master,
            }
        )
        missing_countries = missing_countries[[merged_by_in_country_master, 'replacement']]
        missing_countries = missing_countries.drop_duplicates()

        if target_dataset_path in pj.DataFramesCSV:
            target_dataset = pd.read_csv(target_dataset_path.value)
        else:
            target_dataset = pd.read_excel(target_dataset_path.value)

        target_dataset.rename(
            columns=pj.COLUMN_RENAME_BY_DATASET.get(target_dataset_path),
            inplace=True,
        )
        target_dataset = self._du_03_filter_data(df_enum_path=target_dataset_path, dataframe=target_dataset)

        desired_columns = [column for _, column in pj.COLUMN_RENAME_BY_DATASET.get(target_dataset_path).items()]
        columns_to_remove = [column for column in target_dataset.columns if column not in desired_columns]
        target_dataset.drop(columns=columns_to_remove, inplace=True)

        country_master = self.country_master[['country', 'country_code']]
        existing = target_dataset.merge(
            country_master,
            on=merged_by_in_country_master,
        )

        merged_missing = target_dataset.merge(
            missing_countries,
            on=merged_by_in_country_master,
        )
        merged_missing['country'] = merged_missing['replacement']
        merged_missing.drop(columns=['replacement'], inplace=True)
        merged_missing = merged_missing[merged_missing['country'].notnull()]
        full_countries = pd.concat(
            [existing, merged_missing],
            ignore_index=True
        )
        columns_to_remove = [
            column for column in full_countries.columns if column[0:12] == 'country_code'
        ]
        full_countries.drop(columns=columns_to_remove, inplace=True)
        return full_countries

    def _du_02_obesity(self):
        obesity_dataset = pd.read_csv(pj.DataFramesCSV.WHO_OBESITY_CSV.value)
        self._capture_summary_dataset_to_image(
            dataset=obesity_dataset,
            dataset_name='obesity',
            name_file='du_obesity_summary'
        )
        self._capture_get_dataframe_info_image(
            table_name='Obesity Dataset',
            name_file='du_obesity_dataset',
            dataframe=pd.DataFrame(obesity_dataset),
            figure_size_height=3.3
        )

        # Sex if filtered to match both
        obesity_dataset = obesity_dataset[obesity_dataset.Sex == "Both sexes"]
        obesity_dataset = obesity_dataset[["Numeric", "Countries, territories and areas", "WHO region", 'Year']]
        obesity_dataset.rename(
            inplace=True,
            columns=pj.COLUMN_RENAME_BY_DATASET.get(pj.DataFramesCSV.WHO_OBESITY_CSV)
        )

        self._du_data_exploration_basics(
            dataset=obesity_dataset,
            metric_name_plot='Percentage Obesity',
            metric_label='percentage_obesity',
            dataset_name='obesity',
        )

    def _du_02_country_master(self):
        country_master = pd.read_csv(pj.DataFramesCSV.COUNTRY_MASTER_CSV.value)
        self._capture_get_dataframe_info_image(
            table_name='Countries Dataset',
            name_file='du_country_dataset',
            dataframe=pd.DataFrame(country_master),
            figure_size_height=2.5
        )

    def _du_02_meat_consumption(self):
        meat_consumption = pd.read_csv(pj.DataFramesCSV.MEAT_CONSUMPTION_CSV.value)
        self._capture_get_dataframe_info_image(
            table_name='Meat Consumption Dataset',
            name_file='du_meat_consumption_dataset',
            dataframe=pd.DataFrame(meat_consumption),
            figure_size_height=2
        )
        self._capture_summary_dataset_to_image(
            dataset=meat_consumption,
            dataset_name='Meat Consumption',
            name_file='du_meat_consumption_summary',
            figure_size_height=2
        )

        meat_consumption.rename(
            inplace=True,
            columns=pj.COLUMN_RENAME_BY_DATASET.get(pj.DataFramesCSV.MEAT_CONSUMPTION_CSV)
        )
        self._merge_by_country_code(dataset_name='meat_consumption', target_dataset=meat_consumption)
        self._du_data_exploration_basics(
            dataset=meat_consumption,
            metric_name_plot='Kg./Year per Capita - Beef consumption',
            metric_label='beef',
            dataset_name='meat_beef',
            country_label='country_code',
        )
        self._du_data_exploration_basics(
            dataset=meat_consumption,
            metric_name_plot='Kg./Year per Capita - Poultry consumption',
            metric_label='poultry',
            dataset_name='meat_poultry',
            country_label='country_code',
        )
        self._du_data_exploration_basics(
            dataset=meat_consumption,
            metric_name_plot='Kg./Year per Capita - Sheep and Goat consumption',
            metric_label='sheep_and_goat',
            dataset_name='meat_sheep',
            country_label='country_code',
        )
        self._du_data_exploration_basics(
            dataset=meat_consumption,
            metric_name_plot='Kg./Year per Capita - Pig consumption',
            metric_label='pig',
            dataset_name='meat_pig',
            country_label='country_code',
        )
        self._du_data_exploration_basics(
            dataset=meat_consumption,
            metric_name_plot='Kg./Year per Capita - Fish and Seafood consumption',
            metric_label='fish_and_seafood',
            dataset_name='meat_fish_seafood',
            country_label='country_code',
        )

    def _du_02_hunger(self):
        hunger = pd.read_csv(pj.DataFramesCSV.HUNGER_CSV.value)
        self._capture_get_dataframe_info_image(
            table_name='Hunger Dataset',
            name_file='du_hunger_dataset',
            dataframe=pd.DataFrame(hunger),
            figure_size_height=1.5
        )
        self._capture_summary_dataset_to_image(
            dataset=hunger,
            dataset_name='Hunger',
            name_file='du_hunger_summary',
            figure_size_height=1
        )
        hunger = hunger[["Entity", "Year", "Global Hunger Index (2021)"]]
        hunger.rename(
            inplace=True,
            columns=pj.COLUMN_RENAME_BY_DATASET.get(pj.DataFramesCSV.HUNGER_CSV)
        )

        self._merge_by_country_name(dataset_name='hunger', target_dataset=hunger)
        self._du_data_exploration_basics(
            dataset=hunger,
            metric_name_plot='Global Hunger Index',
            metric_label='hunger_index',
            dataset_name='hunger',
            country_label='country',
        )

    def _du_02_smoking(self):
        smoking = pd.read_csv(pj.DataFramesCSV.SMOKING_CSV.value)
        self._capture_get_dataframe_info_image(
            table_name='Smoking Dataset',
            name_file='du_smoking_dataset',
            dataframe=pd.DataFrame(smoking),
            figure_size_height=1
        )
        self._capture_summary_dataset_to_image(
            dataset=smoking,
            dataset_name='Smoking',
            name_file='du_smoking_summary',
            figure_size_height=1
        )
        smoking.drop(inplace=True, columns=["Code"])
        smoking.rename(
            inplace=True,
            columns=pj.COLUMN_RENAME_BY_DATASET.get(pj.DataFramesCSV.SMOKING_CSV)
        )
        self._du_data_exploration_basics(
            dataset=smoking,
            metric_name_plot='Percentage Prevalence Tobacco use Adults',
            metric_label='prevalence_smoking',
            dataset_name='smoking',
        )
        self._merge_by_country_name(dataset_name='smoking', target_dataset=smoking)

    def _du_02_alcohol_consumption(self):
        alcohol_consumption = pd.read_csv(pj.DataFramesCSV.ALCOHOL_CONSUMPTION_CSV.value)
        self._capture_get_dataframe_info_image(
            table_name='Alcohol Consumption Dataset',
            name_file='du_alcohol_consumption_dataset',
            dataframe=pd.DataFrame(alcohol_consumption),
            figure_size_height=1.2
        )
        self._capture_summary_dataset_to_image(
            dataset=alcohol_consumption,
            dataset_name='Alcohol Consumption',
            name_file='du_alcohol_summary',
            figure_size_height=1
        )
        alcohol_consumption.drop(inplace=True, columns=["Code"])

        alcohol_consumption.rename(
            inplace=True,
            columns=pj.COLUMN_RENAME_BY_DATASET.get(pj.DataFramesCSV.ALCOHOL_CONSUMPTION_CSV)
        )
        self._du_data_exploration_basics(
            dataset=alcohol_consumption,
            metric_name_plot='Liters of Pure Alcohol per Capita',
            metric_label='liters_of_pure_alcohol_per_capita',
            dataset_name='alcohol',
        )
        self._merge_by_country_name(dataset_name='alcohol_consumption', target_dataset=alcohol_consumption)

    def _du_02_happiness(self):
        happiness_record = pd.read_excel(pj.DataFramesXLSX.HAPPINESS_REPORT_XLSX.value)
        self._capture_get_dataframe_info_image(
            table_name='Happiness Report Dataset',
            name_file='du_happiness_dataset',
            dataframe=pd.DataFrame(happiness_record),
            figure_size_height=2.5
        )
        self._capture_summary_dataset_to_image(
            dataset=happiness_record,
            dataset_name='Happiness',
            name_file='du_happiness_summary',
            figure_size_height=2.3
        )
        new_columns = [col.split(',', 2)[-1].strip() for col in happiness_record.columns]
        happiness_record.columns = new_columns

        happiness_record.rename(
            inplace=True,
            columns=pj.COLUMN_RENAME_BY_DATASET.get(pj.DataFramesXLSX.HAPPINESS_REPORT_XLSX)
        )
        self._merge_by_country_name(dataset_name='happiness', target_dataset=happiness_record)
        self._du_data_exploration_basics(
            dataset=happiness_record,
            metric_name_plot='Life Ladder',
            metric_label='life_ladder',
            dataset_name='happiness',
            country_label='country',
        )
        self._du_data_exploration_basics(
            dataset=happiness_record,
            metric_name_plot='Social Support',
            metric_label='social_support',
            dataset_name='happiness',
            country_label='country',
        )
        self._du_data_exploration_basics(
            dataset=happiness_record,
            metric_name_plot='Freedom to Make Life Choices',
            metric_label='freedom_to_make_life_choices',
            dataset_name='happiness',
            country_label='country',
        )
        self._du_data_exploration_basics(
            dataset=happiness_record,
            metric_name_plot='Generosity',
            metric_label='generosity',
            dataset_name='happiness',
            country_label='country',
        )
        self._du_data_exploration_basics(
            dataset=happiness_record,
            metric_name_plot='Perceptions of Corruption',
            metric_label='perceptions_of_corruption',
            dataset_name='happiness',
            country_label='country',
        )
        self._du_data_exploration_basics(
            dataset=happiness_record,
            metric_name_plot='Positive Affect',
            metric_label='positive_affect',
            dataset_name='happiness',
            country_label='country',
        )
        self._du_data_exploration_basics(
            dataset=happiness_record,
            metric_name_plot='Negative affect',
            metric_label='negative_affect',
            dataset_name='happiness',
            country_label='country',
        )

    def _du_02_run_processes(self):
        self._function_mapper_du_02(dataset=pj.DataFramesCSV.WHO_OBESITY_CSV)()
        self._function_mapper_du_02(dataset=pj.DataFramesCSV.MEAT_CONSUMPTION_CSV)()
        self._function_mapper_du_02(dataset=pj.DataFramesCSV.HUNGER_CSV)()
        self._function_mapper_du_02(dataset=pj.DataFramesCSV.SMOKING_CSV)()
        self._function_mapper_du_02(dataset=pj.DataFramesCSV.ALCOHOL_CONSUMPTION_CSV)()
        self._function_mapper_du_02(dataset=pj.DataFramesXLSX.HAPPINESS_REPORT_XLSX)()

    def du_02(self):
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.expand_frame_repr', False)

        self._du_02_country_master()
        self._du_02_run_processes()

        self.get_crosstab_missing_countries(save_table=True).info()

    def _du_03_filter_data(self, df_enum_path: (pj.DataFramesCSV, pj.DataFramesXLSX), dataframe: pd.DataFrame):
        if df_enum_path == pj.DataFramesCSV.WHO_OBESITY_CSV:
            return dataframe[(dataframe.Sex == "Both sexes") & (dataframe.year >= 2000) & (dataframe.year <= 2016)]
        return dataframe

    def _du_03_remove_rows(self):
        self.integrated_dataset = self.integrated_dataset[
            ~(
                    pd.isnull(self.integrated_dataset.prevalence_smoking) &
                    pd.isnull(self.integrated_dataset.liters_of_pure_alcohol_per_capita) &
                    pd.isnull(self.integrated_dataset.poultry) &
                    pd.isnull(self.integrated_dataset.beef) &
                    pd.isnull(self.integrated_dataset.sheep_and_goat) &
                    pd.isnull(self.integrated_dataset.pig) &
                    pd.isnull(self.integrated_dataset.fish_and_seafood) &
                    pd.isnull(self.integrated_dataset.life_ladder) &
                    pd.isnull(self.integrated_dataset.social_support) &
                    pd.isnull(self.integrated_dataset.freedom_to_make_life_choices) &
                    pd.isnull(self.integrated_dataset.generosity) &
                    pd.isnull(self.integrated_dataset.perceptions_of_corruption) &
                    pd.isnull(self.integrated_dataset.positive_affect) &
                    pd.isnull(self.integrated_dataset.negative_affect))
        ]

    def _du_03_impute_missing_grouped_country_year(self):
        happiness_columns = [
            'life_ladder',
            'social_support',
            'freedom_to_make_life_choices',
            'generosity',
            'perceptions_of_corruption',
            'positive_affect',
            'negative_affect'
        ]
        others_columns = [
            'hunger_index',
            'prevalence_smoking',
            'liters_of_pure_alcohol_per_capita',

        ]
        meat_columns = [
            'poultry',
            'beef',
            'sheep_and_goat',
            'pig',
            'fish_and_seafood',
        ]

        for column in happiness_columns + others_columns + meat_columns:
            old_null_values = self.integrated_dataset.isnull().sum().tolist()
            # Forward fill to fill NaNs with the closest past values
            self.integrated_dataset[column] = self.integrated_dataset.sort_values(by='year').groupby('country')[
                column].fillna(method='ffill')
            # Backward fill to fill any remaining NaNs with the closest future values
            self.integrated_dataset[column] = self.integrated_dataset.sort_values(by='year').groupby('country')[
                column].fillna(method='bfill')

            self._capture_get_dataframe_info_image(
                table_name=f'Imputation: By Group Country Year: {column}',
                name_file=f'dp_imputation_c_y_{column}',
                dataframe=self.integrated_dataset,
                figure_size_height=4.3,
                previous_data=old_null_values,
                previous_data_name=pj.DataFramePreviousFieldNameOptions.IS_NULL,
            )

    def _du_03_impute_missing_important_features(self):
        old_null_values = self.integrated_dataset.isnull().sum().tolist()
        self.integrated_dataset = self.integrated_dataset[
            ~(self.integrated_dataset['pig'].isnull()) & ~(self.integrated_dataset['life_ladder'].isnull())]
        self._capture_get_dataframe_info_image(
            table_name='Imputation: Rows Without Important Features',
            name_file='dp_imput_impor_feat',
            dataframe=self.integrated_dataset,
            figure_size_height=4.3,
            previous_data=old_null_values,
            previous_data_name=pj.DataFramePreviousFieldNameOptions.IS_NULL,
        )

    def _du_03_impute_missing_pig(self):
        old_null_values = self.integrated_dataset.isnull().sum().tolist()
        # Filling with 0 those countries that does not consume pig
        condition = (~self.integrated_dataset['beef'].isnull()) & (~self.integrated_dataset['poultry'].isnull()) & (
            self.integrated_dataset['pig'].isnull())
        self.integrated_dataset.loc[condition, 'pig'] = 0
        self._capture_get_dataframe_info_image(
            table_name='Imputation: Pig',
            name_file='dp_imput_pig',
            dataframe=self.integrated_dataset,
            figure_size_height=4.3,
            previous_data=old_null_values,
            previous_data_name=pj.DataFramePreviousFieldNameOptions.IS_NULL,
        )

    def _du_03_impute_by_prediction(self, columns_to_impute: List[str]):
        columns_to_drop = ['percentage_obesity', 'country', 'region', 'hunger_index'] + columns_to_impute
        for feature in columns_to_impute:
            old_null_values = self.integrated_dataset.isnull().sum().tolist()
            train_data = self.integrated_dataset.dropna()
            test_data = self.integrated_dataset[self.integrated_dataset[feature].isnull()]

            X_train = train_data.drop(columns=columns_to_drop)
            y_train = train_data[feature]

            X_test = test_data.drop(columns=columns_to_drop)

            model = ensemble.RandomForestRegressor()
            model.fit(X_train, y_train)
            predicted_values = model.predict(X_test)

            self.integrated_dataset.loc[self.integrated_dataset[feature].isnull(), feature] = predicted_values

            self._capture_get_dataframe_info_image(
                table_name=f'Imputation by Prediction: {feature.capitalize()}',
                name_file=f'dp_imput_{feature}',
                dataframe=self.integrated_dataset,
                figure_size_height=4.3,
                previous_data=old_null_values,
                previous_data_name=pj.DataFramePreviousFieldNameOptions.IS_NULL,
            )

    def _du_03_impute_missing(self):
        self._du_03_impute_missing_grouped_country_year()
        self._du_03_impute_missing_pig()
        self._du_03_impute_missing_important_features()
        self._du_03_impute_by_prediction(
            columns_to_impute=[
                'perceptions_of_corruption',
                'prevalence_smoking',
                'social_support',
                'generosity',
                'positive_affect'
            ]
        )

    def _dp_03_create_expected_obesity_rate_attribute(self) -> None:
        threshold_in_hunger = 20
        self.integrated_dataset.loc[self.integrated_dataset['percentage_obesity'] <= 20, 'expected_obesity_rate'] = 1
        self.integrated_dataset['expected_obesity_rate'].fillna(0, inplace=True)
        self.integrated_dataset.loc[self.integrated_dataset['hunger_index'] >= threshold_in_hunger, 'in_hunger'] = 1
        self.integrated_dataset['in_hunger'].fillna(0, inplace=True)
        other_countries_in_hunger = ['Burundi', 'Guinea', 'Libya', 'Niger', 'Uganda', 'Zambia', 'Zimbabwe']
        self.integrated_dataset.loc[self.integrated_dataset["country"].isin(other_countries_in_hunger), 'in_hunger'] = 1
        self.integrated_dataset.loc[
            (self.integrated_dataset['expected_obesity_rate'] == 1) & (
                    self.integrated_dataset['in_hunger'] == 1), 'expected_obesity_rate'
        ] = 0
        self.integrated_dataset.drop(columns=['in_hunger', 'percentage_obesity', 'hunger_index'], inplace=True)
        self._capture_get_dataframe_info_image(
            table_name='New Feature: Obesity Rate Attribute',
            name_file='dt_new_feat_obes_rate_attribute',
            dataframe=self.integrated_dataset,
            figure_size_height=4.3,
        )

    def _dp_03_generate_merged_dataset(self):
        merged_dataset = country_manager._find_missing_countries(
            target_dataset_path=pj.DataFramesCSV.WHO_OBESITY_CSV
        )

        for df_enum_path in [
            pj.DataFramesCSV.HUNGER_CSV,
            pj.DataFramesCSV.SMOKING_CSV,
            pj.DataFramesCSV.ALCOHOL_CONSUMPTION_CSV,
            pj.DataFramesCSV.MEAT_CONSUMPTION_CSV,
            pj.DataFramesXLSX.HAPPINESS_REPORT_XLSX,
        ]:
            is_meat_consumption_dataset = df_enum_path == pj.DataFramesCSV.MEAT_CONSUMPTION_CSV
            country_column = 'country_code' if is_meat_consumption_dataset else 'country'
            dataset = country_manager._find_missing_countries(
                target_dataset_path=df_enum_path,
                merged_by_in_country_master=country_column
            )

            merged_dataset = merged_dataset.merge(
                dataset,
                on=['country', 'year'],
                how='left',
                validate='one_to_one',
                indicator=True,

            )

            merged_dataset.drop(columns=['_merge'], inplace=True)
            merged_dataset.reset_index(drop=True)
            self._capture_get_dataframe_info_image(
                table_name='Merged Dataset 1',
                name_file='dp_merged_dataset_01',
                dataframe=merged_dataset,
                figure_size_height=4.3,
            )
        self.integrated_dataset = merged_dataset

    def _dp_03_remove_attributes(
            self,
            dataset: pd.DataFrame,
            enum: (pj.DataFramesCSV, pj.DataFramesXLSX)
    ) -> pd.DataFrame:
        desired_columns = [column for _, column in pj.COLUMN_RENAME_BY_DATASET.get(enum)]
        columns_to_drop = [column for column in dataset.columns if column not in desired_columns]
        return dataset.drop(columns=columns_to_drop, inplace=True)

    def dp_03(self):
        self._generate_missing_countries_replacement_table()
        self._dp_03_generate_merged_dataset()
        self._du_03_remove_rows()
        self._du_03_impute_missing()
        self._dp_03_create_expected_obesity_rate_attribute()

    def _dt_04_get_feature_distribution(self):
        values_counts = self.integrated_dataset.groupby('expected_obesity_rate')['expected_obesity_rate'].value_counts()
        values_counts.plot(kind='bar', color=['skyblue', 'salmon'])
        plt.title('Distribution of Target Label')
        plt.xlabel('Label')
        plt.ylabel('Count')
        plt.xticks(rotation=0)  # keep the x-axis labels horizontal
        plt.savefig(f'{PATH_IMAGES}dt_feature_balance.png', bbox_inches='tight', dpi=200)

    def dt_04(self):
        self._dt_04_get_feature_distribution()

    def dm_07(self):
        dataset_to_fit = self.integrated_dataset.drop(columns=['country', 'year', 'region'])
        fitter = pj.ModelFitter(
            dataset=dataset_to_fit,
            label='expected_obesity_rate',
            models_to_fit=[
                pj.AvailableModels.DECISION_TREE,
                # pj.AvailableModels.GRADIENT_BOOSTING,
                # pj.AvailableModels.BERNOULLI_NB,
                # pj.AvailableModels.RANDOM_FOREST,
                # pj.AvailableModels.SVM,
                # pj.AvailableModels.LOGISTIC_REGRESSION,
            ],
            tuning_scoring='precision',
            tuning=True,
        )
        fitter.fit_models()
        loop = asyncio.get_event_loop()
        loop.close()
        sys.exit()


country_manager = ProjectManager(
    generate_images_du_02=True,
)

country_manager.du_02()
country_manager.dp_03()
country_manager.dt_04()
country_manager.dm_07()
