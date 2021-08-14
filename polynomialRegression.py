from sklearn import linear_model
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

"""
Utilizaremos para este ejercicio una base de datos de Kaggle: https://www.kaggle.com/akram24/position-salaries?select=Position_Salaries.csv
Que tiene 10 puestos laborales y sus respectivos salarios.
"""

#estos son los datos de la tabla
level = pd.Series([1,2,3,4,5,6,7,8,9,10])
salario = pd.Series([45,50,60,80, 111, 150,200,300,500,1000])

#vemos en el gráfico que un polinomio será mejor aproximación que una recta

X_level = level.values.reshape(-1,1)
y_salario = salario
plt.title('Nivel vs Salario')
plt.xlabel('Level')
plt.ylabel('Salario')
plt.scatter(X_level, y_salario)
plt.show()


#importamos las librerías que utilizaremos para regresión polinómica
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
#la documentación sobre la librería PolynomialFeatures se puede encontrar en https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
from sklearn.preprocessing import PolynomialFeatures

lin_model_salary = LinearRegression()
lin_model_salary.fit(X_level,y_salario)

y_predict_salary = lin_model_salary.predict(X_level)
plt.plot(X_level, y_predict_salary, color='red')

plt.title('Nivel vs Salario')
plt.xlabel('Level')
plt.ylabel('Salario')
plt.scatter(X_level, y_salario)
plt.show()

r2=r2_score(y_salario, y_predict_salary)
print('El error R^2 es {}'.format(r2))

#vimos que la regresión lineal no es lo que mejor se aproxima
#creamos nuestro polinomio

polinomio=PolynomialFeatures(degree=2,include_bias= False)
print(polinomio)
X_polinomio_level = polinomio.fit_transform(X_level)
print(X_polinomio_level)

lin_model_pol = LinearRegression()
lin_model_pol.fit(X_polinomio_level, y_salario)
y_predict_pol = lin_model_pol.predict(X_polinomio_level)

plt.plot(X_level, y_predict_pol, color = 'green')

plt.title('Nivel vs Salario')
plt.xlabel('Level')
plt.ylabel('Salario')
plt.scatter(X_level, y_salario)
plt.show()

#vemos con el error R^2 qué tan bueno es nuestro polinomio

r2_pol = r2_score(y_salario,y_predict_pol)
print("R^2 es {}".format(r2_pol))

"""
Si aumentamos el grado del polinomio podemos mejorar la aproximación a los puntos,
incluso sabemos que al tener 10 puntos sobre los que trabajar, hay un polinomio de grado 9 
que pasa por estos 10 puntos, pero no sería un buen modelo predictivo sino que sería
un descriptor de los que ya tenemos.
"""

polinomio5 = PolynomialFeatures(degree=5, include_bias=False)
X_polinomio5_level = polinomio5.fit_transform(X_level)
lin_model_pol5 = LinearRegression()
lin_model_pol5.fit(X_polinomio5_level, y_salario)
y_predict_pol5 = lin_model_pol5.predict(X_polinomio5_level)

plt.plot(X_level, y_predict_pol5, color='red')

plt.title('Nivel vs Salario')
plt.xlabel('Level')
plt.ylabel('Salario')
plt.scatter(X_level, y_salario)
plt.show()

#vemos con el error R^2 qué tan bueno es nuestro polinomio

r2_pol5 = r2_score(y_salario,y_predict_pol5)
print("R^2 es {}".format(r2_pol5))
#como decíamos, vemos que más que predecir ya casi que se ajusta totalmente a los datos proporcionados.


#finalmente, vemos cómo podemos encontrar un polinomio de grado 9 que pasa por todos los puntos

polinomio9 = PolynomialFeatures(degree=9, include_bias=False)
X_polinomio9_level = polinomio9.fit_transform(X_level)
lin_model_pol9 = LinearRegression()
lin_model_pol9.fit(X_polinomio9_level, y_salario)
y_predict_pol9 = lin_model_pol9.predict(X_polinomio9_level)

plt.plot(X_level, y_predict_pol9, color='red')

plt.title('Nivel vs Salario')
plt.xlabel('Level')
plt.ylabel('Salario')
plt.scatter(X_level, y_salario)
plt.show()

#vemos que R^2 es prácticamente 1

r2_pol9 = r2_score(y_salario,y_predict_pol9)
print("R^2 es {}".format(r2_pol9))