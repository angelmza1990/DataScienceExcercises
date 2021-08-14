"""
Este es un ejercicio de regresión lineal simple de un curso de Udemy.
Importamos las librerías para hacer un modelo de regresión lineal con una base de datos precargada en la librería sklearn


"""


from sklearn import linear_model
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#utilizamos una base de datos de alquileres de Boston que está en la librería de sklearn
#visitar scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html
import sklearn.datasets

#traemos la base de datos
datosBoston = sklearn.datasets.load_boston()

#para ver de qué está compuesta la tabla utilizamos:
print(datosBoston.keys())

#creamos nuestro dataframe para poder manejar mejor los datos y como columna solo utilizamos los datos en 'feature_names'
dataFrameBoston = pd.DataFrame(datosBoston.data, columns = datosBoston.feature_names)
print(dataFrameBoston.head())

#averiguamos la cantidad de filas y columas de la tabla
print(dataFrameBoston.shape)

#obtenemos descripción de qué significa cada columna
print(datosBoston.DESCR)

#agregamos a nuestra tabla una columna con los precios medios de las viviendas que se encuentran en 'target'
dataFrameBoston['Precio'] = datosBoston.target

#vemos si alguna columna tiene valores nulos
print(dataFrameBoston.isnull().sum())
#vemos que no hay valores nulos en la tabla

#hacemos una tabla para analizar los valores de 'Precio'
sns.displot(dataFrameBoston['Precio'], bins=50)
plt.show()

#vemos que hay una acumulación de valores de los 'Precios' en torno al 20 y algunos valores aislados en 50

"""
Para aplicar nuestro modelo de regresión lineal simple utilizamos la matriz de correlación.
 Los datos que estén relacionados linealmente son aquellos más próximos a 1 o -1.
"""
matriz_correlacion = dataFrameBoston.corr().round(3)
print(matriz_correlacion)

#vemos que RM (precio de habitaciones) tiene relación con 'Precio' de 0.70
#también LSTAT con 'PrecioMedio' con -0.74

#podemos ver esta correlación en un mapa de calor
sns.heatmap(data=matriz_correlacion, annot=True)
plt.rcParams['figure.figsize'] = (10,10)
plt.show()

#utilizaremos sólo la relación RM - 'Precio' a fines de hacer simple el ejemplo

#veamos cómo se relaciona el 'Precio' de un departamento con la cantidad de habitaciones 'RM'

plt.scatter(dataFrameBoston['RM'], dataFrameBoston['Precio'])
plt.title('Nro. Habitaciones - Precio')
plt.xlabel('Nro Habitaciones')
plt.ylabel('Precio Hab')
plt.show()
#se puede apreciar que a medida que hay más habitaciones aumenta el precio del departamento


#también podemos ver que hay una relación lineal inversa de 'Precio' y 'LASTAT
#veamos cómo se relaciona el 'Precio' de un departamento con la cantidad de habitaciones 'RM'

plt.scatter(dataFrameBoston['LSTAT'], dataFrameBoston['Precio'])
plt.title('LSTAT - Precio')
plt.xlabel('LSTAT')
plt.ylabel('Precio Hab')
plt.show()


#obtenemos algunos datos estadísticos sobre los precios de las habitaciones
print('Máximo valor {}'.format(dataFrameBoston['Precio'].max()))
print('Mínimo valor {}'.format(dataFrameBoston['Precio'].min()))
print('Promedio {}'.format(dataFrameBoston['Precio'].mean()))
print('Mediana {}'.format(dataFrameBoston['Precio'].median()))
print('Desviación estándar {}'.format(dataFrameBoston['Precio'].std()))

#datos de test y entrenamiento para el modelo

X_featureRM = dataFrameBoston['RM']
y_price = dataFrameBoston['Precio']
# utilizamos la librería de sklearn para dividir aleatoriamente los datos de train y test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_featureRM, y_price, test_size = 0.2, random_state=42)

#vemos el tamaño de X_train, X_test, y_train, y_test
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
#notamos que las variables de entrenamiento son 404 (el 80% de los datos) y son 102 (el 20% de los datos como especificamos en train_test_split) con los que testearemos el modelo

#ahora creamos el modelo, para ello importamos
from sklearn.linear_model import LinearRegression
X_train = X_train.values.reshape(-1,1)
modeloLinealBoston = LinearRegression()
modeloLinealBoston.fit(X_train, y_train)

#queremos ver la ordenada al origen de la recta que calcula el modelo y la pendiente
print('Pendiente: \n', modeloLinealBoston.coef_)
print('Ordenada al origen: \n', modeloLinealBoston.intercept_)

#ahora estudiamos qué tan bueno es nuestro modelo, utilizamos el coeficiente de correlación R^2 de estadística
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
y_train_predict = modeloLinealBoston.predict(X_train)
r2 = r2_score(y_train, y_train_predict)
print('El valor de R^2 para las variables de entrenamiento del modelo es {}'.format(r2))

#ahora hacemos lo mismo pero con los datos de test
X_test = X_test.values.reshape(-1,1)
y_test_predict = modeloLinealBoston.predict(X_test)
r2 = r2_score(y_test, y_test_predict)
print('El valor de R^2 para las variables de testeo es {}'.format(r2))
#nos da un valor más pequeño que con las de train porque con las X_train construimos nuestro modelo

#graficamos nuestro modelo lineal con los puntos de datos de la tabla
datosPrediccion = modeloLinealBoston.predict(dataFrameBoston[['RM']])
#la recta la graficaremos de color rojo
plt.plot(dataFrameBoston['RM'], datosPrediccion, color = 'red')
plt.scatter(dataFrameBoston['RM'], dataFrameBoston['Precio'])
plt.title('Nro. Habitaciones - Precio')
plt.xlabel('Nro Habitaciones')
plt.ylabel('Precio Hab')
plt.show()


#ahora vamos a ver qué sucede gráficamente con los datos de test
plt.plot(dataFrameBoston['RM'], datosPrediccion, color = 'red')
plt.scatter(X_test, y_test) #a diferencia del gráfico anterior que era con dataFrameBoston['RM'], dataFrameBoston['Precio']
plt.title('Nro. Habitaciones - Precio')
plt.xlabel('Nro Habitaciones')
plt.ylabel('Precio Hab')
plt.show()

#finalmente comparamos los valores reales con los de la predicción
dfActualPrediccion = pd.DataFrame({'Actual': dataFrameBoston['Precio'], 'Predicción': datosPrediccion})
print(dfActualPrediccion.head(10))

#vemos la diferencia entre los datos reales y los que predice nuestro modelo, lo vemos en un gráfico de barras:
dfActualPrediccion.head(15).plot(kind='bar')
plt.show()