# Proyecto AI Dev Senpai 2020

### Conformación del equipo
---
- Anthony Cabrera
- Gonzalo Gutiérrez

### Descripción de la problemática a solucionar
---

En el marco del proyecto se entrenará una red *Generative Adversarial Network* (GAN) [1] con el fin de generar imágenes a color de frentes de automóviles con una resolución de 32 por 32 píxeles. Con el fin de disponibilizar el generador, se proveerá un endpoint por medio de la implementación de una API utilizando el *framework web* Flask [2].

### Descripción de la solución inicial planteada
---

Las redes GAN son una clase de algoritmo de deep learning el cual consiste de dos redes neuronales, el generador y el discriminador las cuales compiten entre sí buscando mejorar su desempeño de generación y detección respectivamente.
Por un lado el generador a partir de una entrada de ruido aleatorio genera una muestra falsa. Mientras que el discriminador por otra parte intenta distinguir las muestras reales (del conjunto de entrenamiento), de las falsas que son generadas por la red generadora. Esta competencia lleva a que el discriminador aprenda a clasificar correctamente los ejemplos como reales o falsos y simultáneamente el generador sea capaz de generar muestras más cercanas a la realidad y lograr así engañar a la red discriminadora.

Para este proyecto se diseña y entrena una red basada en la arquitectura Deep Convolutional Generative Adversarial Network (DCGAN)[3] ya que las mismas tienen un mejor desempeño para las imágenes por el aprovechamiento de la información espacial. En la misma el generador se encargará de generar imágenes de automóviles de un tamaño de 32x32x3 en un comienzo, y el discriminador de clasificar si las muestras tanto del dataset como las producidas por el generador son reales o falsas.

En cuanto al Dataset para entrenar a la red neuronal llamado “The Comprehensive Cars (CompCars) dataset” [4] que tiene un total de 136.726 imágenes de autos tomadas desde diferentes ángulos. El origen de las mismas fue realizado a partir de un scraping de la web (web-nature), así como también, tomadas de cámaras en la vía pública (surveillance-nature). En nuestro caso, se utilizarán las 44.481 imágenes de frentes de autos que forman parte del sub-set de surveillance-nature.
Es importante resaltar que se hace un procesado de las imágenes ya que las mismas tienen diversas resoluciones. Por efectos prácticos se llevan las mismas a un tamaño de 32x32x3 con el fin de reducir los tiempos de entrenamiento. Por otra parte, se realiza también el escalado de los valores de los pixeles al rango [-1, 1], ya que es recomendado usar la activación hyperbolic tangent (tanh) como salida del modelo generador [5].

### Descripción inicial del algoritmo de machine learning o modelo de deep learning a utilizar
---

*TODO*

### Análisis de soluciones existentes y detalle de la alternativa seleccionada
---

*TODO*

### Referencias y Bibliografía
--- 
[1] Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems. 2014.<br />
[2] https://palletsprojects.com/p/flask/ 
