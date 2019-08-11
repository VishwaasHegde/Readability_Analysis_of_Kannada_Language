import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Lars, RidgeClassifier
from sklearn.svm import SVR, NuSVR
from sklearn.svm import SVR, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier
import scipy as sp
import seaborn as sns
from scipy.stats import chi2
from sklearn.metrics import confusion_matrix,f1_score, accuracy_score, recall_score

class KRNN():

    batch_size = 250
    y_classes = [1, 2, 3, 4, 6, 7, 8, 9, 10]

    def __init__(self):
        self.build_graph()

    def normalize(self, df):
        ss = StandardScaler()
        # df_new = df.drop('target', 1)
        cols = df.iloc[:,1:].columns
        df_new = ss.fit_transform(df.iloc[:,1:])
        return df_new

    def train(self, X_train, y_train):
        X_train = self.normalize(X_train)
        losses = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for e in range(550):
                for a in range(0,X_train.shape[0], self.batch_size):
                    Xi = X_train[a:a + self.batch_size, :]
                    yi = np.array(y_train[a:a + self.batch_size])
                    _, l = sess.run([self.train_op, self.loss], feed_dict={self.X: Xi, self.y: yi, self.is_training: True})
                losses.append(l)
                if e % 50 == 0:
                    print('loss: ',l)
            self.saver.save(sess, './model.ckpt')

    def test(self, X_test):
        X_test=self.normalize(X_test)
        with tf.Session() as sess:
            self.saver.restore(sess, './model.ckpt')
            nnpred = self.d7.eval(feed_dict={self.X: X_test, self.is_training: False})
        return nnpred.flatten()

    def build_graph(self):
        tf.reset_default_graph()
        self.X = tf.placeholder(shape=[None, 15], dtype=tf.float32)
        self.y = tf.placeholder(shape=[None], dtype=tf.float32)
        self.is_training = tf.placeholder(shape=[], dtype=tf.bool)
        # d1=tf.layers.dense(X, units=128, activation=tf.nn.relu)
        # dr1 =tf.layers.dropout(d1, training=is_training, rate=0.4)

        # d2=tf.layers.dense(dr1, units=64, activation=tf.nn.relu)
        # dr2 =tf.layers.dropout(d2, training=is_training, rate=0.4)

        # d3 = tf.layers.dense(self.X, units=75, activation=tf.nn.relu)
        # dr3 = tf.layers.dropout(d3, training=is_training, rate=0.9)

        # d4 = tf.layers.dense(self.X, units=50, activation=tf.nn.leaky_relu)
        # dr4 = tf.layers.dropout(d4, training=is_training, rate=0.9)

        d5 = tf.layers.dense(self.X, units=30, activation=tf.nn.leaky_relu)
        dr5 = tf.layers.dropout(d5, training=self.is_training, rate=0.4)

        d6 = tf.layers.dense(dr5, units=18, activation=tf.nn.leaky_relu)
        dr6 = tf.layers.dropout(d6, training=self.is_training, rate=0.1)

        self.d7 = tf.layers.dense(dr6, units=1, activation=None)

        self.loss = self.pearson_corr(self.y, self.d7)

        opt = tf.train.AdamOptimizer()
        # opt=tf.train.GradientDescentOptimizer(learning_rate=0.01)
        self.train_op = opt.minimize(self.loss)
        self.saver = tf.train.Saver()

    def vish_preds(self, npred):
        vish_corr = []
        var = []
        N = []
        xm = []
        # for cl in self.y_classes:
        #     x = npred[y_test == cl]
        #     xm.append(np.mean(x))
        #     v = np.var(x)
        #     var.append(v)
        #     N.append(len(x))
        # print(N)
        for j in range(200):
            corr_avg = []
            varc = np.var(self.y_classes)
            for i in range(1, 31):
                gga = get_group_avg(npred, i)
                corr_avg.append(np.square(np.corrcoef(gga, y_classes)[0][1]))
            vish_corr.append(corr_avg)
        return np.mean(vish_corr, axis=0)

    def get_group_avg(self, y_pred, grp=10):
        grp_avg = []
        for a in y_classes:
            choice = np.random.choice(y_pred[y_test == a], grp, False)
            grp_avg.append(np.mean(choice))
        return grp_avg

    def pearson_corr(self, y_true, y_pred):
        original_loss = self.batch_size * tf.reduce_sum(tf.multiply(y_true, tf.transpose(y_pred))) - (
        tf.reduce_sum(y_true) * tf.reduce_sum(y_pred))
        divisor = tf.sqrt(
            (self.batch_size * tf.reduce_sum(tf.square(y_true)) - tf.square(tf.reduce_sum(y_true))) *
            (self.batch_size * tf.reduce_sum(tf.square(y_pred)) - tf.square(tf.reduce_sum(y_pred)))
        )
        return 1 - tf.truediv(original_loss, divisor)

