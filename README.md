# Curva-de-ROC-con-RStudio

title: "Método de Clasificación para modelos de Machine Learning"
author: "Naren Castellon"
date: "01/22/2021"

## 2. **Generación de gráficos ROC**

Cuando utilizamos técnicas de clasificación, podemos confiar en la técnica para clasificar los casos automáticamente. Alternativamente, podemos confiar en la técnica para generar solo las probabilidades de los casos que pertenecen a varias clases y luego determinar las probabilidades de corte nosotros mismos. Los gráficos de características operativas del receptor (ROC) ayudan con el último enfoque al brindar una representación visual de los positivos verdaderos y falsos en varios niveles de corte. Usaremos el paquete ROCR para generar gráficos ROC.

Otra opción para evaluar el desempeño de los clasificadores es utilizar gráficos en lugar de estadísticas cuantitativas. Los gráficos suelen ser más completos que las estadísticas individuales.


Si aún no ha instalado el paquete ROCR, instálelo ahora. Cargue los archivos de datos para este capítulo desde el sitio web del libro y asegúrese de que los archivos rocr-example-1.csv y rocr-example-2.csv estén en su directorio de trabajo de R.

**¿Cómo lo hacemos?**

**Paso 1.**

Para generar gráficos ROC, siga estos pasos:
1. Cargue el paquete ROCR:
```{r}
library(ROCR)
```

**Paso 2.**
2. Lea el archivo de datos y eche un vistazo:
```{r}
dat <- read.csv("data/roc-example-1.csv")
head(dat)
```

**Paso 3.**  Cree el objeto de predicción:

```{r}
pred <- prediction(dat$prob, dat$class)# 0=Fracaso
                                       # 1=éxito
```

El paso 3 crea un objeto de predicción basado en las probabilidades y las etiquetas de clase pasadas como argumentos. En los ejemplos actuales, nuestras etiquetas de clase son 0 y 1, y por defecto 0 se convierte en la clase "fracaso" y 1 se convierte en la clase "éxito". Veremos en la sección Hay más ... a continuación cómo manejar el caso de etiquetas de clase arbitrarias.

**Paso 4.** Cree el objeto de rendimiento:

```{r}
perf <- performance(pred, "tpr", "fpr")
perf
```

El paso 4 crea un objeto de rendimiento basado en los datos del objeto de predicción. Indicamos que queremos la "tasa de verdaderos positivos" y la "tasa de falsos positivos".

**Paso 5.** Trace el gráfico:
```{r}
plot(perf, main="CURVA ROC")
lines( par()$usr[1:2], par()$usr[3:4] )

```

El paso 5 grafica el objeto de rendimiento. La función de trazado no traza la línea diagonal que indica el umbral ROC, y agregamos una segunda línea de código para obtener eso.

Las curvas ROC (Receiver Operating Characteristic) se utilizan a menudo para examinar el compromiso entre detectar verdaderos positivos y evitar falsos positivos.


**Paso 6.** Encuentre los valores de corte para varias tasas positivas verdaderas. Extraiga los datos relevantes del objeto perf en un marco de datos prob.cuts:

```{r}
prob.cuts <- data.frame(cut=perf@alpha.values[[1]], fpr=perf@x.values[[1]], tpr=perf@y.values[[1]])

head(prob.cuts)
tail(prob.cuts)
```

A partir de los cortes de prob. Del marco de datos, podemos elegir el límite correspondiente a nuestra tasa positiva verdadera deseada.


Generalmente usamos gráficos ROC para determinar un buen valor de corte para la clasificación dadas las probabilidades. El paso 6 muestra cómo extraer del objeto de rendimiento el valor de corte correspondiente a cada punto del gráfico. Armados con esto, podemos determinar el límite que produce cada una de las tasas positivas verdaderas y, dada una tasa positiva verdadera deseada, podemos encontrar la probabilidad de corte apropiada.

A continuación, discutimos algunas más de las características importantes de ROCR.

**Usar etiquetas de clase arbitrarias**

A diferencia del ejemplo anterior, podríamos tener etiquetas de clase arbitrarias para el éxito y el fracaso. El archivo rocr-example-2.csv tiene el comprador y el no comprador como etiquetas de clase, y el comprador representa el caso de éxito.

En este caso, necesitamos indicar explícitamente las etiquetas de falla y éxito pasando un vector con el caso de falla como primer elemento:
```{r}
dat <- read.csv("data/roc-example-2.csv")
pred <- prediction(dat$prob, dat$class, label.ordering = c("non-buyer", "buyer"))
perf <- performance(pred, "tpr", "fpr")
plot(perf, main="CURVA de ROC")
lines( par()$usr[1:2], par()$usr[3:4] )
```

El puntaje de probabilidad de la predicción de corte por defecto es de  0.5 o la proporción de 1 y 0 en los datos de entrenamiento. Pero a veces, afinar el corte de probabilidad puede mejorar la precisión tanto en las muestras de desarrollo como en las de validación. 

En el siguiente gráfico de ROC se muestran diferentes puntos de cortes

```{r}
plot(perf, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.7))
```

Las curvas ROC (Receiver Operating Characteristic) se utilizan a menudo para examinar el compromiso entre detectar verdaderos positivos y evitar falsos positivos.

```{r}
curve(log(x), from=0, to=100, xlab="Tasa de falsos positivos", ylab="Tasa de verdaderos positivos",
      main="CURVA ROC", col="green", lwd=3, axes=F)
Axis(side=1, at=c(0, 20, 40, 60, 80, 100), labels = c("0%", "20%", "40%", "6
0%", "80%", "100%"))
Axis(side=2, at=0:5, labels = c("0%", "20%", "40%", "60%", "80%", "100%"))
segments(0, 0, 110, 5, lty=2, lwd=3)
segments(0, 0, 0, 4.7, lty=2, lwd=3, col="blue")
segments(0, 4.7, 107, 4.7, lty=2, lwd=3, col="blue")
text(20, 4, col="blue", labels = "Clasificador perfecto")
text(40, 3, col="green", labels = "Clasificador de prueba")
text(70, 2, col="black", labels= "Clasificador sin valor predictivo")
```


La línea azul en el gráfico anterior representa el clasificador perfecto donde tenemos 0% de falsos positivos y 100% de verdaderos positivos. La línea verde del medio es el clasificador de prueba. La mayoría de nuestros clasificadores entrenados con datos reales se verán así. La línea diagonal negra ilustra un clasificador sin predicciones de valor predictivo. Podemos ver que tiene la misma tasa de verdaderos positivos y la misma tasa de falsos positivos. Por tanto, no puede distinguir entre los dos.

En términos de identificar el valor positivo, queremos que nuestra curva ROC esté lo más cerca posible de la línea perfecta. Por lo tanto, medimos el área bajo la curva ROC (abreviada como AUC) para mostrar qué tan cerca está nuestra curva del clasificador perfecto. Para hacer esto, tenemos que cambiar la escala del gráfico de arriba. Mapeando 100% a 1, tenemos un cuadrado de 1 x 1. El área bajo el clasificador perfecto sería uno, y el área bajo el clasificador sin valor predictivo sería 0.5. Entonces, 1 y 0.5 serán los límites superior e inferior para nuestro modelo de curva ROC. Tenemos el siguiente sistema de puntuación (los números indican el área bajo la curva) para las curvas ROC del modelo predictivo:

* Sobresaliente: 0.9-1.0
* Excelente / bueno: 0,8-0,9
* Aceptable / regular: 0,7–0,8
* Deficiente: 0,6-0,7
* No discriminación: 0,5-0,6.

Tenga en cuenta que este sistema de clasificación es algo subjetivo. Usemos el paquete ROCR para dibujar una curva ROC.

```{r}
roc<-performance(pred, measure="tpr", x.measure="fpr")
```

Al especificar "tpr" (tasa de verdaderos positivos) y "fpr" (tasa de falsos positivos), creamos un objeto de "rendimiento"

```{r}
#plot(roc, main="Curva de ROC ", col="blue", lwd=3)
plot(perf, main="Curva de ROC", col="blue", lwd=3)
segments(0, 0, 1, 1, lty=2)

```


El comando de segmentos dibuja la línea punteada que representa el clasificador sin valor predictivo.

Para medir esto cuantitativamente, necesitamos crear un nuevo objeto de rendimiento con medida = "auc" o área bajo la curva.

```{r}
roc_auc<-performance(pred, measure="auc")

```

Ahora el roc_auc se almacena como un objeto S4. Esto es bastante diferente al marco de datos y las matrices. Primero, podemos usar la función `str ()` para ver su estructura.

```{r}
str(roc_auc)
```
El objeto ROC tiene seis miembros. El valor de AUC se almacena en valores y. Para extraer eso, usamos el símbolo @ de acuerdo con la salida de la función str ().

```{r}
roc_auc@y.values # El área bajo curva

```
Así, el AUC obtenido = 0.8401 lo que sugiere un clasificador justo, de acuerdo con el esquema de puntuación anterior.
