Notice:
This code corresponds with a lecture at “Universidad Nacional del Centro de la Provincia de Buenos Aires”. Slides and further information can be found at my home page: https://sites.google.com/site/rodriguezjuanmanuel/talks/pre-enei-20. This information can be use freely, but please let me know if you use it. I also be really thankful for any correction or comment.

Para las librerías requeridas:
Python: https://www.python.org/
Numpy: http://www.numpy.org/
Theano: http://deeplearning.net/software/Theano/
Keras: http://Keras.io/
matplotlib: http://matplotlib.org/
scipy: https://www.scipy.org

slides.pfd: slides presentadas.
Descripción de los scripts de ejemplo:
linealregresion.py: genera el ejemplo simple, y gráfica los datos y la curva 3x.
linealregresion_exp_param.py: genera el ejemplo simple, y prueba de forma exploratoria los multiplicadore para la regresión lineal.
linealregresion_sgd.py: implamentación del gradiete de MSE y la estrategia de gradient checking.
linealregresion_sgd_2.py: implementación de descenso por el gradiente (gradient descent) utilizando solo numpy.
linealregresion_sgd_Theano.py:  implementación de descenso por el gradiente (gradient descent) utilizando Theano.
digist.py: salva los primeros 10 dígitos de entrenamiento de MNIST a archivos jpeg.                
digist_comp.py: imprime la matriz que representa al primer dígito en el conjunto de entrenamiento de MNIST.
digist_lr_Theano_reg.py: intento de utilizar regresión lineal para predecir dígitos de MNIST (muy malos resultados) basado en Theano.
digist_lr_Theano.py: utilización de regresión lineal para realizar 10 clasifica binarias (una por cada dígito posible) basado en Theano. One-vs-the-rest.
digist_lr.py: utilización de regresión lineal para realizar 10 clasifica binarias (una por cada dígito posible) basado en Keras. One-vs-the-rest.             
digist_logr_Theano.py: utilización de regresión logística para realizar 10 clasifica binarias (una por cada dígito posible) basado en Theano. One-vs-the-rest. No se encuentra en los slides, pero por completitud se agrega. Diferencias: función de error y función de ajuste ahora es sigmoide.
digist_logr.py: utilización de regresión logística para realizar 10 clasifica binarias (una por cada dígito posible) basado en Keras. One-vs-the-rest. No se encuentra en los slides, pero por completitud se agrega. Diferencias: función de error y función de ajuste ahora es sigmoide.
digist_ann.py: red neuronal densa para clasificar MNIST, función de error MSE y SGD (stocastic gradiant descent).
digist_ann_2.py: red neuronal densa para clasificar MNIST, función de error categorical crossentropy y rmsprop            
digist_conv.py: red neuronal convolucional para clasificar MNIST, función de error categorical crossentropy y rmsprop.
digist_conv_2.py: red neuronal convolucional y max pooling para clasificar MNIST, función de error categorical crossentropy y rmsprop.
