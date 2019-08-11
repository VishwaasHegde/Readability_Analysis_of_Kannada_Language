from DataCleaner import DataCleaner
from FeatureExtractor import FeatureExtractor
from Model import KRNN
import pandas as pd
import os.path
from os import path
import os
import argparse

class ScorePredictor:

    def clean(self, df):
        dc=DataCleaner(df)
        return dc.clean_gadya()

    def feature_selector(self, df):
        cols  = df.columns
        if 'message' in cols:
            df=df.drop('message', 1)
        if 'stem_words' in cols:
            df = df.drop('stem_words', 1)
        if 'message_sw' in  cols:
            df = df.drop('message_sw', 1)
        if 'avg_dp_stem' in  cols:
            df = df.drop('avg_dp_stem', 1)
        if 'avg_dist_stem' in  cols:
            df = df.drop('avg_dist_stem', 1)
        if 'ottakshara_avg' in  cols:
            df = df.drop('ottakshara_avg', 1)

        return df
    def pre_process(self, df):

        featureExtractor= FeatureExtractor(df)
        # featureExtractor.combine_all_data()
        featureExtractor.add_features()
        featureExtractor.stem_gadya()
        featureExtractor.cnt_remove_sw()
        featureExtractor.get_hard_word_cnt()
        # featureExtractor.remove_outliers()
        # featureExtractor.get_local_word_cnt()
        featureExtractor.build_word_vec_map()
        featureExtractor.get_word2vec()
        featureExtractor.get_word_dist()
        featureExtractor.build_pos_cols()
        return featureExtractor.get_df()

    def build_model(self):
        self.krnn=KRNN()

    def train(self, X_train, y_train):
        self.krnn.train(X_train, y_train)

    def test(self, X_test):
        return self.krnn.test(X_test)

    def main(self):
        parser = argparse.ArgumentParser(description="Kannada Readability Score Predictor")

        # defining arguments for parser object
        parser.add_argument("-tf", "--testfile", type=str, nargs=1,
                            metavar="test_file", default=None,
                            help="Reads the csv file as test data")

        parser.add_argument("-tt", "--testtext", type=str, nargs=1,
                            metavar="test_text", default=None,
                            help="Reads the Kannada text input string as input data")

        parser.add_argument("-t", "--trainfile", type=str, nargs=1,
                            metavar="train_file", default=None,
                            help="Reads the csv file as train data")

        args = parser.parse_args()
        self.build_model()
        if args.testfile != None:
            testfile = args.testfile[0]
            if not path.exists(testfile):
                raise ValueError("The given path is not a file")
            extension = os.path.splitext(testfile)[1]
            if extension != '.csv':
                raise ValueError("The file should be a csv")
            df = pd.read_csv(extension)
            df = self.clean(df)
            df = self.pre_process(df)
            df = self.feature_selector(df)
            pred = self.test(df)
            print(pred)
        elif args.testtext != None:
            text=args.testtext[0]
            df = pd.DataFrame({'message':[text]})
            df = self.clean(df)
            df = self.pre_process(df)
            df = self.feature_selector(df)
            pred = self.test(df)
            print(pred)
        elif args.trainfile != None:
            trainfile = args.trainfile[0]
            if not path.exists(trainfile):
                raise ValueError("The given path is not a file")
            extension = os.path.splitext(trainfile)[1]
            if extension != '.csv':
                raise ValueError("The file should be a csv")
            df = pd.read_csv(trainfile)
            df = self.clean(df)
            df = self.pre_process(df)
            df = self.feature_selector(df)
            self.train(df.iloc[:,:-1], df.iloc[:,-1])

if __name__ == "__main__":
    score_predictor=ScorePredictor()
    # calling the main function
    score_predictor.main()