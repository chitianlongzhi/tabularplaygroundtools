# !(if [ -d tabularplaygroundtools ]; then rm -rf tabularplaygroundtools ; fi)
# !git clone https://github.com/chitianlongzhi/tabularplaygroundtools.git
# !cat tabularplaygroundtools/tabularplaygroundtools.py
# import tabularplaygroundtools.tabularplaygroundtools
# self.param = {
#     'train.file': '/kaggle/input/playground-series-s5e3/train.csv',
#     'test.file': '/kaggle/input/playground-series-s5e3/test.csv',
#     'CorrelationAnalysis': False,
#     'SimpleImputer': False,
#     'RobustScaler': False,
#     'KFold': True,
#     'type': 'LSTM' # Tensorflow, PyTorch, XGBoost
# }
# tabularplaygroundtools.tabularplaygroundtools.tabularplaygroundtools(self.param)
import pandas as pd
import glob
class tabularplaygroundtools:
  def __init__(self, param):
    self.param = param
    if not 'type' in self.param:
      print('ERROR: type')
      return
    if not 'train.file' in self.param:
      aa = glob.glob('/kaggle/input/*/train.csv')
      if len(aa)==1:
        self.param['train.file'] = aa[0]
        print('# train: {}'.format(self.param['train.file']))
      else:
        print('ERROR: train.file')
        return
    if not 'test.file' in self.param:
      aa = glob.glob('/kaggle/input/*/test.csv')
      if len(aa)==1:
        self.param['test.file'] = aa[0]
        print('# test: {}'.format(self.param['test.file']))
      else:
        print('ERROR: test.file')
        return
    print('############################################################')
    print('# Loading Data')
    print('import pandas as pd')
    print('train = pd.read_csv(\'{}\')'.format(self.param['train.file']))
    train = pd.read_csv(self.param['train.file'])
    print('test = pd.read_csv(\'{}\')'.format(self.param['test.file']))
    test = pd.read_csv(self.param['test.file'])
    if not 'target' in self.param:
      tt = set(train.columns) - set(test.columns)
      if len(tt)==1:
        self.param['target'] = tt.pop()
        print('# target: {}'.format(self.param['target']))
      else:
        print('ERROR: target')
        return
    print('############################################################')
    print('df = pd.concat([train, test], axis=0).reset_index(drop=True).drop([\'id\', \'{}\'], axis=1)'.format(self.param['target']))
    print('############################################################')
    print('# Missing Values')
    if 'SimpleImputer' in self.param and self.param['SimpleImputer']:
      print('from sklearn.impute import SimpleImputer')
      print('imputer = SimpleImputer(strategy=\'mean\')')
      print('print(train.isnull().sum())')
      for c, v in train.isnull().sum().items():
        if v>0:
          print('train[[\'{}\']] = imputer.fit_transform(df[[\'{}\']])'.format(c, c))
      print('print(test.isnull().sum())')
      for c, v in test.isnull().sum().items():
        if v>0:
          print('test[[\'{}\']] = imputer.fit_transform(df[[\'{}\']])'.format(c, c))
    else:
      print('print(train.isnull().sum())')
      for c, v in train.isnull().sum().items():
        if v>0:
          print('train[\'{}\'] = train[\'{}\'].fillna(df[\'{}\'].median())'.format(c, c, c))
      print('print(test.isnull().sum())')
      for c, v in test.isnull().sum().items():
        if v>0:
          print('test[\'{}\'] = test[\'{}\'].fillna(df[\'{}\'].median())'.format(c, c, c))
    if 'CorrelationAnalysis' in self.param and self.param['CorrelationAnalysis']:
      print('############################################################')
      print('# Correlation Analysis')
      print('import matplotlib.pyplot as plt')
      print('import seaborn as sns')
      print('numerical_features = train.select_dtypes(include=[\'int64\', \'float64\']).columns.tolist()')
      print('corr = train[numerical_features].corr()')
      print('plt.figure(figsize=(12, 10))')
      print('sns.heatmap(corr, annot=True, cmap=\'coolwarm\')')
      print('plt.title(\'Correlation Heatmap\')')
      print('plt.show()')
    print('############################################################')
    if 'KFold' in self.param and self.param['KFold']:
      self.prepare_scaling()
      pass
    else:
      print('# Split data')
      print('X = train.drop([\'id\', \'{}\'], axis=1)'.format(self.param['target']))
      print('y = train[\'{}\']'.format(self.param['target']))
      print('from sklearn.model_selection import train_test_split')
      print('X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=1)')
      print('X_test = test.drop([\'id\',], axis=1)')
      self.prepare_scaling()
      print('X_train = scaler.transform(X_train.values)')
      print('X_valid = scaler.transform(X_valid.values)')
      print('X_test = scaler.transform(X_test.values)')
    print('############################################################')
    if self.param['type'] == 'LSTM':
      self.model_LSTM()
    elif self.param['type'] == 'Tensorflow':
      self.model_Tensorflow()
    elif self.param['type'] == 'PyTorch':
      self.model_PyTorch()
    elif self.param['type'] == 'XGBoost':
      self.model_XGBoost()
    else:
      print('unknown type')
    print('############################################################')
    print('%%time')
    tab = ''
    if 'KFold' in self.param and self.param['KFold']:
      print('import numpy as np')
      print('from sklearn.model_selection import KFold')
      print('FOLDS = 5')
      print('kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)')
      print('oof = np.zeros(len(train))')
      print('pred = np.zeros(len(test))')
      print('for i, (train_index, valid_index) in enumerate(kf.split(train)):')
      print('    X_train = X.loc[train_index,:].copy()')
      print('    y_train = y.loc[train_index]')
      print('    X_valid = X.loc[valid_index,:].copy()')
      print('    y_valid = y.loc[valid_index]')
      print('    X_train = scaler.transform(X_train)')
      print('    X_valid = scaler.transform(X_valid)')
      if self.param['type'] == 'LSTM':
        self.fit_LSTM(tab='    ', pred='pred +=')
      elif self.param['type'] == 'Tensorflow':
        self.fit_Tensorflow(tab='    ', pred='pred +=')
      elif self.param['type'] == 'PyTorch':
        self.fit_PyTorch(tab='    ', pred='pred +=')
      elif self.param['type'] == 'XGBoost':
        self.fit_XGBoost(tab='    ', pred='pred +=')
      print('test[\'{}\'] = pred / FOLDS'.format(self.param['target']))
    else:
      if self.param['type'] == 'LSTM':
        self.fit_LSTM(tab='', pred='test[\'{}\'] ='.format(self.param['target']))
      elif self.param['type'] == 'Tensorflow':
        self.fit_Tensorflow(tab='', pred='test[\'{}\'] ='.format(self.param['target']))
      elif self.param['type'] == 'PyTorch':
        self.fit_PyTorch(tab='', pred='test[\'{}\'] ='.format(self.param['target']))
      elif self.param['type'] == 'XGBoost':
        self.fit_XGBoost(tab='', pred='test[\'{}\'] ='.format(self.param['target']))
    print('############################################################')
    print('submission = test[[\'id\', \'{}\']]'.format(self.param['target']))
    print('submission.to_csv(\'submission.csv\', index=False)')
  def prepare_scaling(self):
    print('############################################################')
    print('# Scaling')
    if 'RobustScaler' in self.param and self.param['RobustScaler']:
      print('from sklearn.preprocessing import RobustScaler')
      print('scaler = RobustScaler()')
    else:
      print('from sklearn.preprocessing import StandardScaler')
      print('scaler = StandardScaler()')
    print('scaler.fit(df.values)')
  def model_LSTM(self):
    print('# LSTM model')
    print('from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, LeakyReLU, GRU')
    print('inputs = Input(shape=(1, df.shape[1]))')
    print('x = LSTM(256, activation=\'tanh\', return_sequences=True)(inputs)')
    print('x = Dropout(0.3)(x)')
    print('x = LSTM(128, activation=\'tanh\', return_sequences=True)(x)')
    print('x = Dropout(0.3)(x)')
    print('x = LSTM(64, activation=\'tanh\', return_sequences=True)(x)')
    print('x = Dropout(0.3)(x)')
    print('x = LSTM(32, activation=\'tanh\')(x)')
    print('x = Dense(16, activation=\'tanh\')(x)')
    print('outputs = Dense(1, activation=\'sigmoid\')(x)')
    print('import tensorflow as tf')
    print('learning_rate = 0.001')
    print('model = tf.keras.Model(inputs=inputs, outputs=outputs)')
    print('model.compile(')
    print('    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),')
    print('    loss=\'binary_crossentropy\',')
    print('    metrics=[')
    print('        \'accuracy\',')
    print('        tf.keras.metrics.AUC(name=\'auc\'),')
    print('        tf.keras.metrics.Precision(name=\'precision\'),')
    print('        tf.keras.metrics.Recall(name=\'recall\')')
    print('    ]')
    print(')')
    print('model.summary()')
    print('# Define callbacks')
    print('patience = 20')
    print('early_stopping = tf.keras.callbacks.EarlyStopping(')
    print('    monitor=\'val_loss\',')
    print('    patience=patience,')
    print('    restore_best_weights=True,')
    print('    verbose=1')
    print(')')
    print('reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(')
    print('    monitor=\'val_loss\',')
    print('    factor=0.2,')
    print('    patience=patience//2,')
    print('    min_lr=1e-6,')
    print('      verbose=1')
    print(')')
    print('epochs = 100')
    print('batch_size = 32')
    print('X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))')
  def fit_LSTM(self, tab, pred):
    print(tab+'# reshape for LSTM')
    print(tab+'X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))')
    print(tab+'X_valid = X_valid.reshape((X_valid.shape[0], 1, X_valid.shape[1]))')
    print(tab+'history = model.fit(')
    print(tab+'    X_train, y_train,')
    print(tab+'    validation_data=(X_valid, y_valid),')
    print(tab+'    epochs=epochs,')
    print(tab+'    batch_size=batch_size,')
    print(tab+'    callbacks=[early_stopping, reduce_lr],')
    print(tab+'    verbose=1')
    print(tab+')')
    print(tab+pred+' model.predict(X_test)')
  def model_Tensorflow(self):
    print('# Tensorflow model')
    print('import warnings')
    print('warnings.filterwarnings(\'ignore\')')
    print('import tensorflow as tf')
    print('print(tf.test.is_gpu_available())')
    print('print(tf.test.is_built_with_cuda())')
    print('model = tf.keras.models.Sequential()')
    print('model.add(tf.keras.layers.Dense(128, activation=\'relu\', use_bias=True, input_shape=(X_train.shape[1],)))')
    print('model.add(tf.keras.layers.Flatten())')
    print('model.add(tf.keras.layers.Dropout(0.25))')
    print('model.add(tf.keras.layers.BatchNormalization())')
    print('model.add(tf.keras.layers.Dense(units=1, activation=\'sigmoid\'))')
    print('auc = tf.keras.metrics.AUC(name=\'aucroc\')')
    print('optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00001, rho=0.9, epsilon=1e-08, decay=0.0)')
    print('model.compile(loss=\'binary_crossentropy\', optimizer=optimizer, metrics=[\'accuracy\', auc])')
    print('model.summary()')
    print('tf.keras.utils.plot_model(model, show_shapes=True)')
    print('# Define callbacks')
    print('earlystopping = tf.keras.callbacks.EarlyStopping(monitor=\'val_loss\', min_delta=0, patience=5, verbose=1, restore_best_weights=True)')
    print('reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor=\'val_loss\', factor=0.3, patience=3, verbose=1, min_delta=1e-4)')
    print('callbacks = [earlystopping, reduce_lr]')
  def fit_Tensorflow(self, tab, pred):
    print(tab+'history = model.fit(')
    print(tab+'    x=X_train, y=y_train,')
    print(tab+'    batch_size=128,')
    print(tab+'    shuffle=True,')
    print(tab+'    epochs=10,')
    print(tab+'    validation_data=(X_valid, y_valid),')
    print(tab+'    callbacks=callbacks')
    print(tab+')')
    print(tab+pred+' model.predict(X_test)')
  def model_PyTorch(self):
    print('# PyTorch model')
    print('import torch')
    print('device=\'cuda\' if torch.cuda.is_available() else \'cpu\'')
    print('class LogisticRegressionModel(torch.nn.Module):')
    print('    def __init__(self):')
    print('        super(LogisticRegressionModel, self).__init__()')
    print('        self.linear1 = torch.nn.Linear(X_train_tensor.shape[1], 128)')
    print('        self.dropout = torch.nn.Dropout(0.25)')
    print('        self.linear2 = torch.nn.Linear(128, 1)')
    print('    def forward(self, x):')
    print('        x = self.linear1(x)')
    print('        x = torch.relu(x)')
    print('        x = self.dropout(x)')
    print('        x = self.linear2(x)')
    print('        x = torch.sigmoid(x)')
    print('        return x')
    print('model = LogisticRegressionModel().to(device)')
    print('criterion = torch.nn.BCELoss()')
    print('optimizer = torch.optim.Adam(model.self.parameters(), lr=0.001, eps=1e-08)')
  def fit_PyTorch(self, tab, pred):
    print(tab+'# reshape for PyTorch')
    print(tab+'X_train_tensor = torch.Tensor(np.float32(X_train)).to(device)')
    print(tab+'X_valid_tensor = torch.Tensor(np.float32(X_valid)).to(device)')
    print(tab+'X_test_tensor  = torch.Tensor(np.float32(X_test )).to(device)')
    print(tab+'y_train_tensor = torch.Tensor(np.float32(y_train)).to(device).reshape(-1, 1)')
    print(tab+'y_valid_tensor = torch.Tensor(np.float32(y_valid)).to(device).reshape(-1, 1)')
    print(tab+'train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)')
    print(tab+'train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)')
    print(tab+'valid_dataset = torch.utils.data.TensorDataset(X_valid_tensor, y_valid_tensor)')
    print(tab+'valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=512, shuffle=True)')
    print(tab+'for epoch in range(1000):')
    print(tab+'    model.train()')
    print(tab+'    loss_train_list=[]')
    print(tab+'    for X_train_tensor, y_train_tensor in train_loader:')
    print(tab+'        optimizer.zero_grad()')
    print(tab+'        o_train_tensor = model(X_train_tensor)')
    print(tab+'        loss_train = criterion(o_train_tensor, y_train_tensor)')
    print(tab+'        loss_train.backward()')
    print(tab+'        optimizer.step()')
    print(tab+'        loss_train_list.append(loss_train.item())')
    print(tab+'    model.eval()')
    print(tab+'    loss_valid_list=[]')
    print(tab+'    for X_valid_tensor, y_valid_tensor in train_loader:')
    print(tab+'        o_valid_tensor = model(X_valid_tensor)')
    print(tab+'        loss_valid = criterion(o_valid_tensor, y_valid_tensor)')
    print(tab+'        loss_valid_list.append(loss_valid.item())')
    print(tab+'    if epoch%100==0:')
    print(tab+'        print(\'{} train loss {}, valid loss {}\'.format(epoch, np.mean(loss_train_list), np.mean(loss_valid_list)))')
    print(tab+pred+' model(X_test_tensor).detach().numpy()')
  def model_XGBoost(self):
    print('# XGBoost model')
    print('from xgboost import XGBClassifier')
    #print('import xgboost')
    print('model = XGBClassifier(')
    print('    device=\'cpu\',')
    print('    max_depth=6,')
    print('    colsample_bytree=0.9,')
    print('    subsample=0.9,')
    print('    n_estimators=10_000,')
    print('    learning_rate=0.1,')
    print('    eval_metric=\'auc\',')
    print('    early_stopping_rounds=100,')
    print('    alpha=1,')
    print(')')
  def fit_XGBoost(self, tab, pred):
    print(tab+'model.fit(')
    print(tab+'    X_train, y_train,')
    print(tab+'    eval_set=[(X_valid, y_valid)],')
    print(tab+'    verbose=100')
    print(tab+')')
    #print(tab+'oof[valid_index] = model.predict_proba(X_valid)[:,1]')
    print(tab+pred+' model.predict_proba(X_test)[:,1]')
    #print('m = roc_auc_score(train[\'{}\'].values, oof)'.format(self.param['target']))
    #print('print(f\'XGBoost CV Score AUC = {m:.3f}\')')
