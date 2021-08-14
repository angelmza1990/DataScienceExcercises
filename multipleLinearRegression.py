"""
Continuamos con la base de datos de los alquileres de Boston, pero para mejorar el modelo
de regresión lineal simple utilizaremos regresión lineal múltiple
"""


from sklearn import linear_model
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#utilizamos una base de datos de alquileres de Boston que está en la librería de sklearn
#visitar scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html
import sklearn.datasets


#traemos la base de datos
datosBoston = sklearn.datasets.load_boston()

#creamos nuestro dataframe para poder manejar mejor los datos y como columna solo utilizamos los datos en 'feature_names'
dataFrameBoston = pd.DataFrame(datosBoston.data, columns = datosBoston.feature_names)
print(dataFrameBoston.head())

#agregamos a nuestra tabla una columna con los precios medios de las viviendas que se encuentran en 'target'
dataFrameBoston['Precio'] = datosBoston.target

#matriz de correlación
matrizCorrelacion = dataFrameBoston.corr().round(3)
print(matrizCorrelacion)
# ya sabíamos que el 'Precio' tiene relación lineal negativa (-0,74) con LSTAT y positiva (0,7) con RM
#vemos  de nuevo el mapa de calor con estas relaciones
sns.heatmap(data = matrizCorrelacion, annot =True)
plt.show()

#realizamos también los diagramas de dispersión de todos los datos
#con estos diagramas podemos ver qué datos guardan alguna relación lineal
sns.pairplot(dataFrameBoston, height = 2)
plt.tight_layout()
plt.show()

#para nuestro modelo de regresión lineal múltiple utilizaremos RM y LSTAT con 'Precio'

dfModelo = dataFrameBoston[['RM', 'LSTAT']]
dfModeloPrecios = dataFrameBoston['Precio']

#ahora separamos nuestros datos de entrenamientoy de testeo
X_train, X_test, y_train, y_test = train_test_split(dfModelo, dfModeloPrecios, test_size=0.2, random_state=42) #dfModelo son los datos de X y dfModeloPrecios los de y

#creamos nuestro modelo
from sklearn.linear_model import LinearRegression
modeloLineal = LinearRegression()
modeloLineal.fit(X_train, y_train)

#vemos cuáles son los coeficientes (a,b) de nuestro modelo lineal múltiple  "y = aX_1 + bX_2 + d" y nuestra término independiente d.
print('Coeficientes: \n', modeloLineal.coef_)
print('Término independiente: \n', modeloLineal.intercept_)

#para medir qué tan bueno es el modelo, utilizamos R^2
from sklearn.metrics import r2_score
y_train_predict = modeloLineal.predict(X_train)
r2 = r2_score(y_train, y_train_predict)
print('El valor de R^2 es {}'.format(r2))

#utilizamos R^2 con los datos de test
y_test_predict = modeloLineal.predict(X_test)
r2 = r2_score(y_test, y_test_predict)
print('El valor de R^2  para lso datos de test es {}'.format(r2))

#finalmente, comparamos los datos reales con los que predice nuestro modelo

prediccion = modeloLineal.predict(dfModelo)

dfActualPrediccion = pd.DataFrame({'Actual': dataFrameBoston['Precio'], 'Predicción': prediccion})
#vemos en la tabla los datos actuales con los predecidos por el modelo
print(dfActualPrediccion.head(10))

#vemos estos datos de la tabla en un gráfico de barras
dfActualPrediccion.head(15).plot(kind ='bar')
plt.show()