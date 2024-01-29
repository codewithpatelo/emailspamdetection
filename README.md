# Detección de e-mails spam mediante la técnica de Clasificación por Naive Bayes
## Propuesta de Investigación

Gerpe Patricio, Simonelli Agostina, Toytoyndjian Eugenia

https://www.youtube.com/watch?v=9--tHWt6osU&list=PLN2e9R_DoC0SaPxIXJcw5ow4HK7CLwTgW

### Resumen:
El análisis de lenguaje natural se corresponde a un dominio de aplicación del aprendizaje estadístico que ha crecido en el último tiempo. Entre las aplicaciones más típicas, podemos encontrar el desarrollo de modelos de clasificación binaria entrenados a partir de corpuses de correos electrónicos para poder distinguir al correo electrónico deseado del no deseado. Los métodos de aprendizaje automático supervisado son considerados en la literatura científica como óptimos para esta tarea. El propósito de esta investigación es (A) analizar en profundidad el algoritmo supervisado de clasificación de Naive Bayes en tal aplicación, (B) utilizarlo para etiquetar nuevos correos como deseado o no deseado (spam) comparando el método de Naive Bayes contra otro método supervisado como el Radom Forest en base a la métrica de performance recall y (C ) contribuir didácticamente para comprender la complejidad del análisis de textos, dada entre otras cosas por la ambigüedad del lenguaje .
Problemática:

El uso de los e-mails está mundialmente expandido debido a su facilidad y practicidad. Su uso cobra especial importancia a nivel laboral y también personal. Por otro lado, la lectura y respuesta de mails es una actividad que demanda mucho tiempo. Pues como se explica en el trabajo de Henkel et. al [2], se requieren 10 empleados a tiempo completo para responder los mails que recibe la Agencia de Pensiones en Suecia, ya que responder cada mail demanda, en promedio, 10 minutos. Sumado a lo anterior, los e-mails spam pueden presentar contenido malicioso, y también termina causando una pérdida de tiempo considerable en el borrado de los mismos [3].  Asimismo, esta acumulación de e-mails innecesarios no solo representa inconvenientes a nivel individual, sino también un impacto ambiental cada vez más relevante dado la escala contemporánea de la comunicación digital y las emisiones de carbono involucradas en el almacenamiento de los mismos [4,5].
Dada esta problemática, el objetivo general de la investigación es analizar en profundidad el funcionamiento del algoritmo de Naive Bayes a partir de una revisión de literatura científica en tal campo de aplicación y utilizarlo para la clasificación de e-mails, en “spam” y “no spam”. Esto define entonces el tipo de problema en una clasificación binaria. Para comparar la performance del modelo trabajaremos con el algoritmo de Random Forest (RF) y tomaremos como métrica de performance a la sensibilidad o recall.
Para desarrollar el trabajo se utilizará una base de datos (BD) de uso libre, descargada de la plataforma Kaggle [6]. (e-mail Spam Classification Dataset e-mail Spam Classification Dataset CSV (kaggle.com) con la cuál ajustaremos los dos modelos a comparar en nuestra investigación.

### Objetivos de investigación:

En primer lugar, uno de los objetivos específicos del trabajo será implementar el algoritmo de NB en la clasificación de e-mails en las etiquetas de no spam y spam. Luego, comparar el algoritmo de NB vs RF, en su forma de clasificar los e-mails. Posteriormente se calcularán las métricas de clasificación, dándole mayor importancia al recall. Se prioriza esta métrica, ya que para este problema en particular consideramos importante minimizar las ocasiones en las que el e-mail sea considerado spam y no lo sea, pues puede resultar muy grave para las personas y organizaciones que e-mails importantes terminen en la casilla de spam. 
Otro objetivo es el didáctico, tanto para nuestro grupo creador del contenido, como para las personas usuarias del video. Esperamos que una persona que haya visto el video, pueda comprender la complejidad del análisis de textos, dada entre otras cosas por la ambigüedad del lenguaje (en el contexto de los e-mails). También que se comprendan los conceptos básicos de las técnicas de Naive Bayes y Random Forest, y el uso de algunos hiper parámetros de relevancia, como por ejemplo, el  Laplace Smoothing en el caso de Naive Bayes. 

### Revisión de literatura

El método de Naive Bayes es una técnica probabilística ideal para clasificación, su característica es que asume que las variables explicativas son todas independientes entre sí, dada una clase. Utiliza la probabilidad condicional de las palabras para determinar a qué categoría pertenece cada texto.  Este algoritmo es fácil de implementar, rápido y preciso. (Raschka, 2014) [7]
Al ser un algoritmo muy utilizado en el ámbito de Machine Learning, su explicación y relevancia se puede encontrar en distintos libros de hace ya decadas atras tales como “Machine Learning” de Tom M. Mitchell, donde, en el capítulo 6, llamado Bayesian Learning, explica el funcionamiento de Naive Bayes (apartado 6.9).
En este libro, Michel menciona que el algoritmo de Naive Bayes está entre los más prácticos para cierto tipo de problemas de aprendizaje; “Por ejemplo, Michel et al. (1994) proporcionan un estudio detallado que compara el clasificador ingenuo de Bayes con otros algoritmos de aprendizaje, incluidos árboles de decisión y algoritmos de redes neuronales. Estos investigadores muestran que el clasificador ingenuo de Bayes es competitivo con estos otros algoritmos de aprendizaje en muchos casos y que en algunos casos supera estos otros métodos” (Mitchell, 1997 , p. 154). [8]
Por otro lado, al revisar literatura científica, podemos dar cuenta que la problemática particular de la clasificación de e-mails entre spam / no spam utilizando el método de Naive Bayes ya ha sido previamente trabajada desde los 90, en distintas investigaciones como por ejemplo en el paper “A Bayesian Approach to Filtering Junk E-Mail”. [9] No obstante, podemos notar que el algoritmo por sencillo que pueda resultar, sigue siendo relevante en la actualidad. Pues, aún en el día de hoy existen numerosas investigaciones analizando dicho algoritmo para la tarea de clasificación de e-mails no deseados [13, 14, 15].
En el mismo sentido, existen también otros trabajos donde se realizan comparaciones entre los métodos de clasificación [16,17]. En ellos, se concluye la superioridad de las técnicas de aprendizaje automático supervisado, siempre y cuando antes exista un dataset de datos previamente etiquetados, lo cuál, también se reflexiona que es algo que puede consumir mucho tiempo y dinero. Ahora bien, en lo que respecta a la comparativa particular ente el método de Bayes y el Random Forest, este estudio se observó que el algoritmo Random Forest fue el que obtuvo la mejor precisión, siendo esta de un 95.5 % mientras que el algoritmo Naïve Bayes fue el más veloz en la construcción del clasificador.[10] Como previamente enunciamos, esta investigación buscará identificar si el método de Bayes puede resultar óptimo en términos de sensibilidad.

### Metodología de investigación

La base de datos utilizada fue “e-mail Spam Classification Dataset CSV” de la plataforma Kaggle. [4]. La misma presenta la estructura que se describe a continuación:

#### Estructura

El dataset cuenta con 5172 observaciones y 3002 columnas. La 1er columna corresponde al nombre del e-mail. Por otro lado, la última columna, “Prediction”,  hace referencia a si el correo electrónico es spam (1) o no spam (0). Las restantes 3000 columnas corresponden a las palabras más comunes utilizadas en los e-mails, luego de excluir aquellos caracteres no-alfabéticos. 
Los datos que figuran en las celdas corresponden a la cantidad de veces que aparece cada una de las palabras en cada uno de los correos electrónicos. 

Se trabajará con el lenguaje R, en la interfaz de programación R-studio y se utilizará el modelo de la librería NaiveBayes. Siendo que la probabilidad cero es un problema en este algoritmo (Vaibhav Jayaswal, 2020) [11], vamos a trabajar con el parámetro Laplace Smoothing = 1 para evitar tener probabilidad cero en aquellos casos donde una palabra esté en el dataset de test pero no en el dataset de entrenamiento. De esta manera, cuando se intente calcular la probabilidad de observar una palabra en un e-mail ya sea spam o no spam, la probabilidad nunca será cero. 
Para comparar la performance del modelo de Naive Bayes, utilizaremos el algoritmo de Random Forest de la librería RandomForest. Ambos modelos se compararán con las métricas recall, accuracy y precisión, con foco en el recall. De manera adicional, observaremos performance también en términos de eficiencia computacional ya que consideramos que este factor podría ser crítico a la hora de implementar productivamente un algoritmo de clasificación.




### Bibliografía:

[1] Fernández, J. M. (2023). Clasificación automática de correos electrónicos (Doctoral dissertation, Universidad Nacional de La Plata).

[2]  Henkel, M., Perjons, E., and Sneiders,  (2017) 1507–1516 E. Examining the potential of language technologies in public organizations by means of a business and its architecture model. International Journal of Information Management 37, 1.

[3] Analysis of Naive Bayes Algorithm for e-mail Spam Filtering across Multiple Datasets. En https://iopscience.iop.org/article/10.1088/1757- 899X/226/1/012091/pdf

[4] Berners-Lee, M. (2022). The carbon footprint of everything. Greystone Books Ltd.

[5] Batmunkh, A. (2022). Carbon Footprint of The Most Popular Social Media Platforms. Sustainability, 14(4), 2195. MDPI AG. Obtenido desde http://dx.doi.org/10.3390/su14042195

[6] e-mail Spam Classification Dataset CSV. (2020, 10 marzo). Kaggle. https://www.kaggle.com/datasets/balaka18/e-mail-spam-classification-dataset-csv

[7] Raschka, S. (2014). Naive bayes and text classification i-introduction and theory. arXiv preprint arXiv:1410.5329.

[8] Mitchell, T. (1997). Machine Learning-McGraw-Hill Science. Engineering.

[9]  Sahami, M., Dumais, S., Heckerman, D., & Horvitz, E. (1998, July). A Bayesian approach to filtering junk e-mail. In Learning for Text Categorization: Papers from the 1998 workshop (Vol. 62, pp. 98-105).

[10]. Soumyabrata Saha, Suparna DasGupta y Suman Kumar Das. (2021) «Spam mail detection using data mining: A comparative analysis». En: Smart Intelligent Computing and Applications. Springer, 2021, págs. 571-580

[11] Vaibhav Jayaswa (2020). Laplace smoothing in Naïve Bayes algorithm
Link: Laplace smoothing in Naïve Bayes algorithm | by Vaibhav Jayaswal | Towards Data Science

[12] Loukas, S., PhD. (2023, 18 septiembre). Text classification using Naive Bayes: Theory & a Working example. Medium. https://towardsdatascience.com/text-classification-using-naive-bayes-theory-a-working-example-2ef4b7eb7d5a

[13] Wibisono, A. (2023). Filtering Spam e-mail Menggunakan Metode Naive Bayes. Jurnal Teknologi Pintar, 3(4).

[14] Sumithra, A., Ashifa, A., Harini, S., & Kumaresan, N. (2022, January). Probability-based Naïve Bayes Algorithm for e-mail Spam Classification. In 2022 International Conference on Computer Communication and Informatics (ICCCI) (pp. 1-5). IEEE.

[15] Nagaraj, P., Muneeswaran, V., Reddy, G. S. S., Kumar, V. B., Mohan, B. M., & Kumar, S. (2023, January). Automatic e-mail Spam Classification Using Naïve Bayes. In 2023 International Conference on Computer Communication and Informatics (ICCCI) (pp. 1-5). IEEE.

[16] Ahmed, N., Amin, R., Aldabbas, H., Koundal, D., Alouffi, B., & Shah, T. (2022). Machine learning techniques for spam detection in e-mail and IoT platforms: Analysis and research challenges. Security and Communication Networks, 2022, 1-19.

[17] Zhang, C., Jia, D., Wang, L., Wang, W., Liu, F., & Yang, A. (2022). Comparative research on network intrusion detection methods based on machine learning. Computers & Security, 102861.




