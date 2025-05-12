<p align="left">
  <img src="static/images/unison.png" alt="Logo UNISON" width="40"/>
  <img src="static/images/mcd.png" alt="Logo UNISON" width="40"/>
</p>

# Sistema de Reconocimiento Facial 

Este sistema permite la **identificación automática de personas** a través de una cámara en vivo utilizando técnicas de **visión por computadora** y **aprendizaje profundo**. Está diseñado como una herramienta educativa para la **Maestría en Ciencia de Datos**, y puede ejecutarse en tiempo real en un navegador web.

---

## ¿Cómo Funciona?

El sistema realiza tres tareas principales:

1. **Captura de rostro en vivo** desde una cámara web.
2. **Detección de rostro** mediante un modelo YOLOv8 especializado.
3. **Generación y comparación de embeddings faciales** usando el modelo FaceNet.

Si el rostro es reconocido, se muestra el nombre, el nivel de confianza y cualquier información adicional registrada. En caso contrario, se ofrece la opción de **registrar al nuevo individuo**.

---

##  Interfaz Web

La aplicación cuenta con una interfaz web construida con **Flask** y **Bootstrap 5**, que incluye:

- Transmisión en vivo desde la cámara
- Botones para **capturar y analizar** un rostro
- Visualización de **imagen capturada**
- Resultados de reconocimiento con **barra de confianza**
- Sección para **registrar nuevas personas**

---

## Estructura del Proyecto
```
├── app.py # Servidor Flask
├── templates/
│ └── index.html # Interfaz principal
├── static/
│ └── images/ # Logos e imágenes auxiliares
├── known_faces/ # Archivos .pkl con embeddings de personas conocidas
├── yolov8n-face.pt # Modelo YOLOv8 entrenado para detección de rostros
├── facenet_model.h5 # Modelo FaceNet para extracción de embeddings
└── data/
└── dataset_face_rec/ # Dataset de entrenamiento
```

---

## Tecnologías Utilizadas

- Python 3.12
- Flask
- OpenCV
- TensorFlow / Keras (FaceNet)
- Ultralytics YOLOv8 (detección de rostros)
- Bootstrap 5 (interfaz web)

---

##  Instrucciones de Uso

1. Ejecuta el servidor Flask:
    ```bash
    python app.py
    ```
2. Abre el navegador en `http://localhost:8000`
3. Presiona **"Capturar y Analizar Rostro"** para identificar a alguien.
4. Si el rostro no es reconocido, puedes registrarlo usando el botón **"Registrar Nueva Persona"**.

---

##  Formato de Registro

Cada persona registrada se guarda en un archivo `.pkl` en el directorio `known_faces/` con el siguiente contenido:

```python
{
  "name": "Nombre de la persona",
  "encodings": [embedding1, embedding2, ...],
  "info": {
    "from_dataset": True,
    "additional_info": "Información opcional"
  }
}
