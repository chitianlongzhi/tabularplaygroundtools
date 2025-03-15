class tabularplaygroundtools:
  def __init__(self, param):
    print('# Loading Data')
    print('import pandas as pd')
    if not 'train.file' in param:
      print('ERROR: train.file')
    print('train = pd.read_csv(\'{}\')'.format(param['train.file']))
    train = pd.read_csv(param['train.file'])
    if not 'test.file' in param:
      print('ERROR: test.file')
    print('test = pd.read_csv(\'{}\')'.format(param['test.file']))
    test = pd.read_csv(param['test.file'])
    print('# Identifying Missing Values')
    print('train.isnull().sum()')
    for c, v in train.isnull().sum().items():
      if v>0:
        print('train[\'{}\'] = train[\'{}\'].fillna(train[\'{}\'].median())'.format(c, c, c))
    print('test.isnull().sum()')
    for c, v in test.isnull().sum().items():
      if v>0:
        print('test[\'{}\'] = test[\'{}\'].fillna(test[\'{}\'].median())'.format(c, c, c))
    if 'CorrelationAnalysis' in param:
      print('# Correlation Analysis')
      print('import matplotlib.pyplot as plt')
      print('import seaborn as sns')
      print('numerical_features = train.select_dtypes(include=[\'int64\', \'float64\']).columns.tolist()')
      print('corr = train[numerical_features].corr()')
      print('plt.figure(figsize=(12, 10))')
      print('sns.heatmap(corr, annot=True, cmap=\'coolwarm\')')
      print('plt.title(\'Correlation Heatmap\')')
      print('plt.show()')
    if not 'target' in param:
      print('ERROR: target')
    print('# Split data')
    print('X = train.drop([\'id\', \'{}\'], axis=1)'.format(param['target']))
    print('y = train[\'{}\']'.format(param['target']))
    print('from sklearn.model_selection import train_test_split')
    print('X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=1)')
    print('# Scaling')
    print('from sklearn.preprocessing import StandardScaler')
    print('scaler = StandardScaler()')
    print('X_train = scaler.fit_transform(X_train)')
    print('X_val = scaler.transform(X_val)')
    if 'type' in param and param['type'] == 'LSTM':
      print('# reshape for LSTM')
      print('X_train3 = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))')
      print('X_val3 = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))')
