import pandas as pd
from minepy import MINE
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import preprocessing
import numpy as np
import datetime
from utils.utils import now


class calculate_importance():

    def __init__(self, variables, purpose, para=["rf", "etr", "corr", "mic"]):
        '''
        目的変数に対する重要度や相関係数を算出するクラス

        Args:
            variables (pandas.DataFrame): 説明変数を格納したdataframe
            purpose (pandas.DataFrame): 目的変数を格納したdataframe
        '''
        self._all_rf_importance = pd.DataFrame()
        self._all_etr_importance = pd.DataFrame()
        self._all_mic = pd.DataFrame()
        self._all_corr = pd.DataFrame()
        # 特徴量と目的変数に分ける
        X = variables.copy()
        y = purpose.copy()

        # 特徴量名を取り出しておく
        feature_x = X.columns

        for column in y.columns:
            print(column)
            if "rf" in para:
                rf_importance = self.__calculate_rf_importance(X, y[column])
                rf_importance = pd.Series(rf_importance, index=feature_x, name=column)
                self._all_rf_importance = pd.concat([self._all_rf_importance, rf_importance], axis=1, sort=False)
                print(" rf")
            if "etr" in para:
                etr_importance = self.__calculate_etr_importance(X, y[column])
                etr_importance = pd.Series(etr_importance, index=feature_x, name=column)
                self._all_etr_importance = pd.concat([self._all_etr_importance, etr_importance], axis=1, sort=False)
                print(" etr")
            if "mic" in para:
                mic_list = self.__mic_calculation(X, y[column])
                mic = pd.Series(mic_list, index=feature_x, name=column)
                self._all_mic = pd.concat([self._all_mic, mic], axis=1, sort=False)
                print(" mic")
            if "corr" in para:
                corr_list = self.__corr_calculation(X, y[column])
                corr = pd.Series(corr_list, index=feature_x, name=column)
                self._all_corr = pd.concat([self._all_corr, corr], axis=1, sort=False)
                print(" corr")

        self._all_corr = self._all_corr.fillna(0)


    def __calculate_rf_importance(self, X, y):
        '''
        目的変数に対する説明変数の需要度を算出する

        Args:
            X (pandas.DataFrame): 説明変数を格納したdataframe
            y (pandas.Series): 目的変数を格納したSeries
        Return:
            
        '''
        reg_rf = RandomForestRegressor(random_state=0, n_estimators=100)
        reg_rf.fit(X=X, y=y)
        return reg_rf.feature_importances_


    def __calculate_etr_importance(self, X, y):
        '''
        目的変数に対する説明変数の需要度を算出する

        Args:
            X (pandas.DataFrame): 説明変数を格納したdataframe
            y (pandas.Series): 目的変数を格納したSeries
        '''
        reg_etr = ExtraTreesRegressor(random_state=0, n_estimators=100)
        reg_etr.fit(X=X, y=y)
        return reg_etr.feature_importances_


    def __mic_calculation(self, X, y):
        '''
        目的変数に対する説明変数の非線形相関係数を算出する

        Args:
            X (pandas.DataFrame): 説明変数を格納したdataframe
            y (pandas.Series): 目的変数を格納したSeries
        '''
        mine = MINE()
        mic_list = np.array([])
        for n, column in enumerate(X.columns):
            mine.compute_score(y, X[column])
            mic_list = np.append(mic_list, mine.mic())
        return mic_list


    def __corr_calculation(self,X, y):
        '''
        目的変数に対する説明変数の相関係数を算出する

        Args:
            X (pandas.DataFrame): 説明変数を格納したdataframe
            y (pandas.Series): 目的変数を格納したSeries
        '''
        corr_list = np.array([])
        for n, column in enumerate(X.columns):
            np.seterr(divide='ignore', invalid='ignore')
            corr_list = np.append(corr_list, np.corrcoef(y, X[column])[0,1])
        return corr_list


    def __rank_importance(self, df, column):
        df = df.sort_values(column, ascending=False)
        df["rank"] = range(1, len(df)+1)
        df["rank"] = df["rank"]/len(df)
        df["rank"] = df["rank"].where(df["rank"] > 0.1, 5)
        df["rank"] = df["rank"].where(df["rank"] > 0.2, 4)
        df["rank"] = df["rank"].where(df["rank"] > 0.3, 3)
        df["rank"] = df["rank"].where(df["rank"] > 0.4, 2)
        df["rank"] = df["rank"].where(df["rank"] > 2, 1)
        return df["rank"]


    def rank_feature(self, output_name):
        '''
        総関係数や重要度から算出したスコアで説明変数に順位をつける

        Args:
            output_name (String): 保存するエクセルのファイル名
        '''
        corr = self._all_corr
        mic = self._all_mic
        rf = self._all_rf_importance
        etr = self._all_etr_importance
        rank_calc = lambda x: 5 if x > 0.6 else 4 if x > 0.4 else 3 if x > 0.25 else 2 if x > 0.16 else 1
        writer = pd.ExcelWriter(f"{now()}_{output_name}")
        for column in corr.columns:
            rank = pd.DataFrame(index=corr.index)
            rank["corr_rank"] = corr[column].abs().apply(rank_calc)
            rank["corr"] = corr[column]
            rank["mic_rank"] = mic[column].apply(rank_calc)
            rank["mic"] = mic[column]
            rank["rf_rank"] = self.__rank_importance(rf, column).astype("int")
            rank["rf"] = rf[column]
            rank["etr_rank"] = self.__rank_importance(etr, column).astype("int")
            rank["etr"] = rf[column]
            rank_index = [column for column in corr.index]
            print(rank)
            rank = rank.loc[rank_index][["corr", "mic", "rf", "etr"]].abs()
            mm = preprocessing.MinMaxScaler()
            print(rank)
            rank = pd.DataFrame(mm.fit_transform(rank), index=rank.index, columns=rank.columns)
            print(rank)
            rank["score"] = rank.sum(axis=1)
            rank["score"] = rank["score"].rank(ascending=False)
            rank.to_excel(writer, sheet_name=column)
            writer.save()
        self._rank = rank