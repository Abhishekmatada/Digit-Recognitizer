
importimport  numpynumpy  asas  npnp
 importimport  pandaspandas  asas  pdpd 
  fromfrom  matplotlibmatplot  import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
%matplotlib inline

data=pd.read_csv('mnist.csv')
data.head()

a=data.iloc[3,1:].values

a=a.reshape(28,28).astype('uint8')

plt.imshow(a)
df_x=data.iloc[:,1:]
df_y=data.iloc[:,0]

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)

x_train.head()
y_train.head()

rf=RandomForestClassifier(n_estimators=100)
rf.fit(x_train,y_train)


pred=rf.predict(x_test)
pred

s=y_test.values
count=0



forfor  ii  inin  rangerange((lenlen((predpred)):)):
         ifif  predpred[[ii]]====ss[[ii]:]:
                 count=count+1

count
len(pred)