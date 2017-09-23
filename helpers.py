import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support as prf_score
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from datetime import datetime 
import requests
import pickle
import time
import matplotlib.pylab as pylab
from matplotlib.ticker import MaxNLocator
from keras.callbacks import Callback

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def print_runtime(start):
    end = time.time()
    print('Runtime: %d min %d sec' % ((end-start)//60, (end-start)%60))
    
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def get_all_currencies(search_data, booking_data):
    gr1 = search_data.groupby('currency')
    gr2 = booking_data.groupby('currency')
    s1 = set(gr1.groups.keys())
    s2 = set(gr2.groups.keys())
    return s1.union(s2)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def get_conversion_rates(search_data, booking_data):
    import requests
    
    rates = dict()
    for ccc in get_all_currencies(search_data, booking_data):
        url = 'http://www.xe.com/currencyconverter/convert/?Amount=1&From=%s&To=USD' % (ccc) 
        r = requests.get(url) 
        rates[ccc] = float(r.text.split('uccResultAmount')[1].split('span')[0][2:-2])
        print("from:%s to:USD --- rate:%10.7f" % (ccc, rates[ccc]))
        
    return rates

def convert_fare_currency(df, booking_time, rates):
    df[booking_time] = pd.to_datetime(df[booking_time])
    df['currency'] = df['currency'].map(lambda s: s.upper())
    curr = df['currency'].values.astype(type('string_type'))
    fare = df['fare'].values

    fare_USD = np.zeros(len(fare)) # initialize
    
    for i, token in enumerate(zip(fare,curr)):
        fare_USD[i] = token[0] * rates[token[1]]
        
    return fare_USD

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def get_dict_airport(airport_data):
    dict_airport = {}
    gr = airport_data.groupby('iata_code')

    for k, rowlabel in gr.groups.items():
        vals = airport_data.iloc[rowlabel][['latitude', 'longitude']].values[0]
        dict_airport[k] = vals
    return dict_airport

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


def haversine_distance(tup1, tup2):
    from math import sin, cos, sqrt, atan2, radians, acos, pi, asin
    
    lat1, lon1 = tup1
    lat2, lon2 = tup2
    
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    return km


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def add_distance_column(df, dict_airport):
    origin = df['origin']
    destination = df['destination']
    distance = np.zeros(len(df))
    for i, token in enumerate(zip(origin, destination)):
        try:
            distance[i] = haversine_distance(dict_airport[token[0]], dict_airport[token[1]])
        except:
            distance[i] = None

    df['distance'] = distance
    df['distance'].replace(np.nan, df['distance'].mean(), inplace=True)
    
    return df
    
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#
# pre-processing airport_data 
# fill in the latitude and longitude NaN's, with country averages.

def nan_fill_airport(airport_data):
    for ix in airport_data.index:    
        if np.isnan(airport_data.loc[ix, 'latitude']):
            df = airport_data[airport_data['country'] == airport_data.loc[ix,'country']] 
            lat_mean = df['latitude'].mean()
            lon_mean = df['longitude'].mean()
            airport_data.loc[ix, 'latitude'] = lat_mean
            airport_data.loc[ix, 'longitude'] = lon_mean
        

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


def plotter(xlabel=None, ylabel=None, title=None, xlim=None, ylim=None):
    fig = plt.figure()
    ax = plt.gca()
    fig.set_size_inches((15,5))
    plt.grid('on')
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
    if xlim: plt.xlim((0, xlim));
    if ylim: plt.ylim((0, ylim));
    if title: plt.title(title)
    return fig, ax

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def create_nonlinear_features(df):
    # create some non-linear features on 'distance'
    df.loc[:,'distance_n'] = (df.loc[:,'distance'])/df['distance'].std()
    df.loc[:,'distance_n0p5'] = df.loc[:,'distance_n'] ** 0.5
    df.loc[:,'distance_n1p5'] = df.loc[:,'distance_n'] ** 1.5
    df.loc[:,'distance_n2'] = df.loc[:,'distance_n'] ** 2
    df.loc[:,'distance_n3'] = df.loc[:,'distance_n'] ** 3
    df.loc[:,'distance_n4'] = df.loc[:,'distance_n'] ** 4
    
    # Now, normalize them 
    df.loc[:,'distance_n0p5'] = df.loc[:,'distance_n0p5']/df.loc[:,'distance_n0p5'].std()
    df.loc[:,'distance_n1p5'] = df.loc[:,'distance_n1p5']/df.loc[:,'distance_n1p5'].std()
    df.loc[:,'distance_n2'] = df.loc[:,'distance_n2']/df.loc[:,'distance_n2'].std()
    df.loc[:,'distance_n3'] = df.loc[:,'distance_n3']/df.loc[:,'distance_n3'].std()
    df.loc[:,'distance_n4'] = df.loc[:,'distance_n4']/df.loc[:,'distance_n4'].std()

# # # # # # # # # 
# # # # # # # # # 

# DEFINE HELPER FUNCTIONS FOR THE CLASSIFIER
def threshold_rounder(y_pred_test, threshold=0.5):
    y_pred_test_01 = np.zeros_like(y_pred_test)
    y_pred_test_01[y_pred_test > threshold] = 1
    y_pred_test_01[y_pred_test < threshold] = 0

    return y_pred_test_01

def confusion_matrix(y_true, y_pred):
    a = y_pred[y_true == 1]
    b = y_pred[y_true == 0]
    
    tp = sum(a)
    fp = sum(b)
    tn = len(b) - sum(b)
    fn = len(a) - sum(a)
    
    accuracy  = (tp+tn)/ float(tp+tn+fp+fn)
    recall    = tp/float(tp+fn)
    precision = tp/float(tp+fp)
    f1_score  = 2*precision*recall/float(precision + recall)
    fpr = fp/(fp+tn)
    
    return accuracy, precision, recall, f1_score, fpr
#........................................................


# HELPER FUNCTIONS FOR THE NEURAL NETWORK MODEL
class Callback_Func(Callback):
    def __init__(self, train_data, test_data, start, wanna_plot=True):
        self.train_data = train_data
        self.test_data = test_data
        self.loss_train = []
        self.loss_test = []
        self.acc = []
        self.start = start
        self.wanna_plot = wanna_plot
        
        
    def plotter(self, title='validation mse'):
        ax = plt.subplot(121)
        plt.xlabel('epochs')
        plt.grid('on')
        plt.title('loss')
        x_plot = range(1, len(self.loss_train)+1)
        plt.xlim((1,max(max(x_plot),2)))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.plot(x_plot, self.loss_test, 'r--^', alpha=.7, label="validation")
        ax.plot(x_plot, self.loss_train, 'k--^', alpha=.7, label="train")
        plt.legend()

        ax = plt.subplot(122)
        plt.xlabel('epochs')
        plt.grid('on')
        ax.plot(x_plot, self.acc, 'b--^', alpha=.5)
        plt.xlim((1,max(max(x_plot),2)))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.title(title)
        plt.ion()
        plt.show()


    def on_epoch_end(self, epoch, logs={}):
        X_train, y_train = self.train_data
        X_test, y_test = self.test_data
        _loss_train, _ = self.model.evaluate(X_train, y_train, verbose=0)
        _loss_test, _acc = self.model.evaluate(X_test, y_test, verbose=0)
        self.loss_train.append(_loss_train)
        self.loss_test.append(_loss_test)
        self.acc.append(_acc)
        
        end = time.time()
        print('\nRuntime: %d min %d sec' % ((end-self.start)//60, (end-self.start)%60))
        
        if self.wanna_plot: self.plotter()
        
#........................................................


def return_ROC_statistics(model, X_train, X_cv, X_test, y_test, threshold=np.linspace(0,1,51)):
    y_pred_train = model.predict(X_train, batch_size=1024)[:,0]
    y_pred_cv    = model.predict(X_cv,    batch_size=1024)[:,0]
    y_pred_test  = model.predict(X_test,  batch_size=1024)[:,0]

    rec, fpr, acc, pre = [], [], [], [] 
    if not hasattr(threshold, '__iter__'):
        threshold = [threshold]
    for t in threshold:
        y_pred_train_01 = threshold_rounder(y_pred_train, threshold=t)
        y_pred_cv_01    = threshold_rounder(y_pred_cv,    threshold=t)
        y_pred_test_01  = threshold_rounder(y_pred_test,  threshold=t)
        acc_, pre_, rec_, f1_, fpr_ = confusion_matrix(y_test, y_pred_test_01)
        acc.append(acc_)
        pre.append(pre_)
        rec.append(rec_)
        fpr.append(fpr_)
    return y_pred_train, y_pred_cv, y_pred_test, y_pred_train_01, y_pred_cv_01, y_pred_test_01, acc, pre, rec, fpr
#........................................................