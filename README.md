# Detector (captura_imagenes)

Resumen
--------
Notebook para capturar imágenes desde una cámara, ejecutar inferencia con un modelo YOLO (.pt), agregar resultados (10 imágenes por defecto), guardar resumen y enviar un estado ("sana", "danada", "defectosa") a un ESP32 por serial. También guarda imágenes anotadas (subcarpeta `anotadas`) dentro de la carpeta de captura.

Archivos principales
--------------------
- `captura_imagenes (1).ipynb` — Notebook principal con celdas:
  - configuración (MODEL_PATH, CAPTURE_COUNT, INTERVAL)
  - carga de modelo (ultralytics / torch.hub / torch.load fallback)
  - captura de imágenes
  - inferencia y agregación
  - guardado de `results.csv` y `results_summary.json` dentro de la carpeta de captura
  - funciones serial (listar_puertos, seleccionar_puerto, send_state, enviar_estado_auto)
  - UI (botones): "Ejecutar pipeline (capturar → predecir → enviar)" y "Ejecutar celdas ANTERIORES (modo kernel-only)"
  - guarda imágenes anotadas en `/<timestamp>_ui/anotadas/annotated_*.jpg` cuando sea posible

Requisitos (Python)
--------------------
Recomiendo crear un entorno virtual y luego instalar las dependencias mínimas:

```bash
# en zsh
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio  # si no lo tienes; usar la versión adecuada para tu CUDA/CPU
python -m pip install opencv-python pandas matplotlib ipywidgets pyserial
# opcional (recomendado si usas ultralytics models):
python -m pip install ultralytics
```

Si no quieres instalar `ultralytics`, el notebook intentará cargar con `torch.hub` o con `torch.load`.

Permisos y puerto serie
-----------------------
- Asegúrate de que tu usuario tenga permisos para acceder al puerto serial (habitualmente grupo `dialout` en Linux):

```bash
sudo usermod -aG dialout $USER
# luego cierra sesión y vuelve a entrar o reinicia la terminal para que el grupo se aplique
```

- Cierra el Monitor Serial del IDE (p. ej. Arduino IDE) antes de ejecutar el envío desde el notebook.
- El puerto por defecto configurado en el notebook es `/dev/ttyUSB0`. Puedes cambiar `PREFERRED_PORT` en la celda correspondiente.

Uso (rápido)
------------
1. Coloca tu archivo de modelo `.pt` en la misma carpeta que el notebook o ajusta `MODEL_FILE_NAME` / `MODEL_PATH` en la celda de configuración.
2. Abre `captura_imagenes (1).ipynb` en VS Code o Jupyter.
3. Si no cargaste el modelo manualmente, pulsa el botón:
   - "Ejecutar celdas ANTERIORES (modo kernel-only)" — esto intentará cargar el modelo y definirá helpers seriales si faltan, y luego ejecutará el pipeline.
4. Alternativamente, si ya cargaste el modelo, pulsa:
   - "Ejecutar pipeline (capturar → predecir → enviar)" — captura `CAPTURE_COUNT` imágenes, ejecuta inferencia y envía el estado al ESP32 (si hay puerto disponible).
5. Revisa el panel de salida del widget para ver resultados por imagen, guardado de anotadas y la respuesta del ESP32.

Qué se genera
-------------
Dentro del directorio de trabajo (donde ejecutas el notebook) se crearán carpetas con timestamp, por ejemplo:

- `20251123_153012_ui/` — carpeta de captura creada por la UI
  - `imagen_1.jpg`, ..., `imagen_10.jpg` — imágenes originales
  - `anotadas/annotated_imagen_1.jpg`, ... — imágenes anotadas (si el backend lo soporta)
  - `results.csv` — CSV con columnas `image`, `class`, `conf`, (y `image_name` añadido)
  - `results_summary.json` — resumen con `final_class`, `final_confidence`, medias, etc.

Variables útiles en kernel
--------------------------
Al ejecutar con la UI, el pipeline guardará en el kernel (globals) las siguientes variables para inspección posterior:
- `results_list` — lista de diccionarios por imagen: `{'image':..., 'class':..., 'conf':...}`
- `results_df` — DataFrame con los resultados
- `final_class`, `final_conf` — decisión agregada
- `carpeta` — ruta de la carpeta de captura

Serial / ESP32
--------------
- El notebook envía una línea con el estado seguida de `\n` (por ejemplo `sana\n`).
- Se incluye previamente un sketch de ejemplo para ESP32. Resumen del sketch:
  - Escucha por Serial (115200), lee líneas y en función de `sana`/`danada`/`defectosa` enciende pines (por ejemplo `GREEN_PIN`, `YELLOW_PIN`, `RED_PIN`) y responde con `ACK` o con alguna confirmación. 

Consejos y notas
----------------
- Anotaciones: si usas `ultralytics` el notebook llama `res.plot()` y guarda la imagen resultante; si usas `torch.hub` (yolov5) se usa `res.render()` y luego guarda la imagen. Si tu backend no soporta anotaciones, el notebook copia la imagen original en la carpeta `anotadas` como fallback.
- Rendimiento: la generación de anotadas actualmente ejecuta la inferencia otra vez por imagen (es simple y robusto). Si prefieres eficiencia, puede reusar los objetos de resultado de la primera pasada.
- Cámara: el índice de cámara por defecto en el notebook es `2` (ajústalo con la función `listar_camaras()` si tu cámara está en otro índice). Usa `probar_camara(index)` para testear.


Contacto / problemas
--------------------
Si encuentras errores al cargar el modelo (por ejemplo se detecta un .zip o un checkpoint con formato inesperado), asegúrate de que el archivo es un `.pt` compatible. Para problemas seriales, revisa permisos y que no haya otro proceso abriendo el puerto.


