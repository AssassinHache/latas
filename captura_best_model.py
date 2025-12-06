#!/usr/bin/env python3
"""Pipeline completo para capturar imágenes, ejecutar un modelo YOLO y enviar el
resultado al ESP32, replicando la lógica del notebook `captura_imagenes (1).ipynb`
pero usando el modelo `best_model.pth`.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch

try:
    import serial
    import serial.tools.list_ports as list_ports
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "pyserial es requerido para controlar el motor/ESP32. Instálalo con `pip install pyserial`."
    ) from exc

# ------------------------------------------------------------
# Configuración general
# ------------------------------------------------------------
CAPTURE_COUNT = 20
INTERVAL = 1.0  # segundos entre capturas
CAMERA_INDEX = 0

MODEL_FOLDER = Path('.')
MODEL_FILE_NAME = 'best_model.pth'
MODEL_PATH = (MODEL_FOLDER / MODEL_FILE_NAME).resolve()
MODEL_LOAD_PATH = MODEL_PATH
if MODEL_PATH.exists() and MODEL_PATH.suffix == '.pth':
    MODEL_LOAD_PATH = MODEL_PATH.with_suffix('.pt')
    if not MODEL_LOAD_PATH.exists() or MODEL_LOAD_PATH.stat().st_mtime < MODEL_PATH.stat().st_mtime:
        shutil.copyfile(MODEL_PATH, MODEL_LOAD_PATH)
        print(f"Copiando {MODEL_PATH.name} -> {MODEL_LOAD_PATH.name} para cumplir con Ultralytics.")
elif MODEL_PATH.suffix == '.pt':
    MODEL_LOAD_PATH = MODEL_PATH

PREFERRED_PORT = '/dev/ttyUSB0'
BAUDRATE = 115200

CLASS_TO_STATE = {
    'person': 'danada',
    'none': 'sana',
    'sana': 'sana',
    'defect': 'defectosa',
    'defectosa': 'defectosa',
    'defectuosa': 'defectosa'
}
VALID_STATES = ("sana", "danada", "defectosa")

# ------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------

def ensure_dir(path: Path | str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def timestamp_folder(suffix: str = '') -> str:
    base = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{base}{suffix}" if suffix else base

# ------------------------------------------------------------
# Control del motor / ESP32
# ------------------------------------------------------------

def ejecutar_motor(port: str = PREFERRED_PORT, baudrate: int = BAUDRATE, command: str = 'motor') -> None:
    """Envía el comando `motor` al ESP32 y espera a que responda."""
    try:
        esp = serial.Serial(port, baudrate, timeout=1)
    except Exception as exc:
        print(f"Error al conectar con {port}: {exc}")
        return

    time.sleep(2)
    print(f"Conectado al ESP32 en {port}. Enviando '{command}'.")
    esp.write((command + '\n').encode())

    try:
        while True:
            if esp.in_waiting:
                respuesta = esp.readline().decode('utf-8', errors='ignore').strip()
                if respuesta:
                    print('ESP32 ->', respuesta)
                if 'Motor terminado' in respuesta:
                    break
            else:
                time.sleep(0.1)
    finally:
        esp.close()
        print('Puerto serial cerrado.')

# ------------------------------------------------------------
# Captura de imágenes
# ------------------------------------------------------------

def capturar_imagenes(count: int = CAPTURE_COUNT, interval: float = INTERVAL, camera_index: int = CAMERA_INDEX) -> str:
    carpeta = timestamp_folder()
    ensure_dir(carpeta)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f'No se pudo abrir la cámara index={camera_index}.')

    try:
        for i in range(count):
            ret, frame = cap.read()
            if not ret:
                print(f'Frame {i + 1} no leído.')
                continue
            nombre_archivo = os.path.join(carpeta, f'imagen_{i + 1}.jpg')
            cv2.imwrite(nombre_archivo, frame)
            print(f'Imagen {i + 1}/{count} guardada -> {nombre_archivo}')
            time.sleep(interval)
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f'Captura finalizada. Carpeta: {carpeta}')
    return carpeta


def motor_y_captura_concurrente() -> str:
    result_holder: Dict[str, str] = {}

    def _wrap_capture():
        result_holder['carpeta'] = capturar_imagenes()

    hilo_motor = threading.Thread(target=ejecutar_motor)
    hilo_camara = threading.Thread(target=_wrap_capture)

    hilo_motor.start()
    hilo_camara.start()

    hilo_motor.join()
    hilo_camara.join()

    carpeta = result_holder.get('carpeta')
    if carpeta is None:
        raise RuntimeError('La captura no produjo una carpeta de salida.')
    return carpeta

# ------------------------------------------------------------
# Carga del modelo
# ------------------------------------------------------------

def load_model(model_path: Path) -> Tuple[torch.nn.Module, str]:
    if not model_path.exists():
        raise FileNotFoundError(f'No se encontró el archivo del modelo en {model_path}')

    backend = None
    model = None
    print('Intentando cargar el modelo desde', model_path)

    try:
        from ultralytics import YOLO
        print('Cargando con ultralytics.YOLO...')
        model = YOLO(str(model_path))
        backend = 'ultralytics'
    except Exception as exc:
        print('ultralytics no disponible o carga falló:', exc)

    if model is None:
        try:
            print('Intentando torch.hub (ultralytics/yolov5 custom)...')
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(model_path), trust_repo=True, source='github', force_reload=False)
            backend = 'yolov5_hub'
        except Exception as exc:
            print('torch.hub tampoco pudo cargar el modelo:', exc)

    if model is None:
        try:
            print('Intentando torch.load...')
            checkpoint = torch.load(str(model_path), map_location='cpu')
            if isinstance(checkpoint, torch.nn.Module):
                model = checkpoint
                backend = 'torch_module'
            elif isinstance(checkpoint, dict) and 'model' in checkpoint:
                model = checkpoint['model']
                backend = 'torch_state_dict'
            else:
                raise RuntimeError('Formato del checkpoint no reconocido.')
        except Exception as exc:
            raise RuntimeError('No se pudo cargar el modelo local.') from exc

    print('Modelo cargado. Backend:', backend)
    return model, backend  # type: ignore[return-value]

# ------------------------------------------------------------
# Inferencia
# ------------------------------------------------------------

def run_inference(model, backend: str, carpeta: str) -> Tuple[List[Dict], List[str]]:
    image_paths = sorted([str(p) for p in Path(carpeta).glob('*.jpg')])
    if not image_paths:
        raise RuntimeError('No hay imágenes para procesar.')

    results_list: List[Dict] = []
    for img_path in image_paths:
        print('Procesando', img_path)
        try:
            if backend == 'ultralytics':
                res = model(img_path)[0]
                boxes = getattr(res, 'boxes', None)
                if boxes is None or len(boxes) == 0:
                    results_list.append({'image': img_path, 'class': None, 'conf': 0.0})
                    continue
                confidences = boxes.conf.tolist()
                classes = boxes.cls.tolist()
                top_idx = int(np.argmax(confidences))
                top_conf = float(confidences[top_idx])
                top_cls = int(classes[top_idx])
                names = getattr(model, 'names', {}) if hasattr(model, 'names') else {}
                top_name = names.get(top_cls, str(top_cls))
            elif backend in {'torch_module', 'torch_state_dict'}:
                model.eval()
                img = cv2.imread(img_path)
                if img is None:
                    raise RuntimeError('No se pudo leer la imagen para inferencia.')
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                with torch.no_grad():
                    raw = model(tensor)
                if isinstance(raw, (list, tuple)) and len(raw) > 0:
                    raw = raw[0]
                if raw.dim() == 4:
                    raw = raw[0]
                raw = raw.cpu().numpy().reshape(-1, raw.shape[-1])
                boxes = raw[:, :4]
                obj_conf = raw[:, 4]
                class_scores = raw[:, 5:]
                scores = obj_conf[:, None] * class_scores
                top_idx = int(np.argmax(scores))
                top_conf = float(np.max(scores))
                top_cls = int(np.argmax(class_scores[top_idx]))
                names = getattr(model, 'names', {}) if hasattr(model, 'names') else {}
                top_name = names.get(top_cls, str(top_cls))
            else:
                raise RuntimeError('Backend no soportado para predicción.')
            results_list.append({'image': img_path, 'class': top_name, 'conf': top_conf})
            print(f' -> {top_name} ({top_conf:.3f})')
        except Exception as exc:
            print('Error procesando', img_path, exc)
            results_list.append({'image': img_path, 'class': None, 'conf': 0.0})
    return results_list, image_paths

# ------------------------------------------------------------
# Agregación de resultados
# ------------------------------------------------------------

def aggregate_results(results_list: List[Dict], carpeta: str) -> Dict:
    confs = [r['conf'] for r in results_list]
    arith_mean = float(np.mean(confs)) if confs else 0.0
    eps = 1e-9
    geom_mean = float(np.exp(np.mean(np.log(np.array(confs) + eps)))) if confs else 0.0

    class_scores: Dict[str, List[float]] = {}
    for r in results_list:
        cls = r['class'] or 'None'
        class_scores.setdefault(cls, []).append(r['conf'])
    avg_class_scores = {k: (sum(v) / len(v) if v else 0.0) for k, v in class_scores.items()}
    final_class = max(avg_class_scores.items(), key=lambda x: x[1])[0] if avg_class_scores else None
    final_conf = avg_class_scores.get(final_class, 0.0) if final_class else 0.0

    results_df = pd.DataFrame(results_list)
    results_df['image_name'] = results_df['image'].apply(lambda p: Path(p).name)

    csv_path = Path(carpeta) / 'results.csv'
    results_df.to_csv(csv_path, index=False)
    print('Resultados guardados en', csv_path)

    summary_payload = {
        'capture_folder': carpeta,
        'image_count': len(results_list),
        'final_class': final_class,
        'final_confidence': final_conf,
        'arithmetic_mean': arith_mean,
        'geometric_mean': geom_mean,
        'class_scores': avg_class_scores,
    }
    json_path = Path(carpeta) / 'results_summary.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)
    print('Resumen JSON guardado en', json_path)

    print('\nResumen por imagen:')
    for r in results_list:
        print(f"{Path(r['image']).name}: {r['class']} ({r['conf']:.3f})")

    print('\nPromedios:')
    print(f'Arithmetic mean confidence: {arith_mean:.4f}')
    print(f'Geometric mean confidence: {geom_mean:.4f}')
    print(f'Final class: {final_class} ({final_conf:.4f})')

    return {
        'results_df': results_df,
        'final_class': final_class,
        'final_conf': final_conf,
        'summary': summary_payload,
    }

# ------------------------------------------------------------
# Anotaciones
# ------------------------------------------------------------

def generar_anotaciones(model, backend: str, image_paths: List[str], destino: Optional[Path] = None) -> Optional[Path]:
    if not image_paths:
        return None
    destino = destino or (Path(image_paths[0]).parent / 'anotadas')
    ensure_dir(destino)
    for img_path in image_paths:
        try:
            if backend == 'ultralytics':
                res = model(img_path)[0]
                annotated = res.plot()
                salida = destino / f"annotated_{Path(img_path).name}"
                cv2.imwrite(str(salida), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            elif backend == 'yolov5_hub':
                res = model(img_path)
                res.render()
                annotated = res.ims[0]
                salida = destino / f"annotated_{Path(img_path).name}"
                cv2.imwrite(str(salida), annotated)
            else:
                salida = destino / f"annotated_{Path(img_path).name}"
                shutil.copy(img_path, salida)
            print('Guardada anotación en', salida)
        except Exception as exc:
            print('No fue posible anotar', img_path, '->', exc)
    return destino

# ------------------------------------------------------------
# Serial helpers
# ------------------------------------------------------------

def listar_puertos() -> List[str]:
    ports = list(list_ports.comports())
    devices = []
    if not ports:
        print('No se encontraron puertos serie disponibles.')
        return devices
    for idx, port in enumerate(ports):
        print(f"{idx}: {port.device} - {port.description}")
        devices.append(port.device)
    return devices


def seleccionar_puerto(preferido: Optional[str] = None) -> Optional[str]:
    puertos = listar_puertos()
    if not puertos:
        return None
    if preferido and preferido in puertos:
        print('Usando puerto preferido:', preferido)
        return preferido
    if len(puertos) == 1:
        print('Seleccionado puerto:', puertos[0])
        return puertos[0]
    print('Varios puertos disponibles, usando el primero. Ajusta PREFERRED_PORT si es necesario.')
    return puertos[0]


def send_state(port: str, state: str, baud: int = BAUDRATE, timeout: float = 1.0) -> Optional[str]:
    if state not in VALID_STATES:
        raise ValueError(f"Estado inválido: {state}")
    if port is None:
        raise RuntimeError('Puerto no especificado.')
    ser = serial.Serial(port, baudrate=baud, timeout=timeout)
    try:
        time.sleep(0.1)
        payload = (state + '\n').encode()
        ser.write(payload)
        try:
            resp = ser.readline().decode(errors='ignore').strip()
        except Exception:
            resp = None
        print(f"Enviado '{state}' -> {port}; Respuesta: '{resp}'")
        return resp
    finally:
        ser.close()


def enviar_estado_auto(state: str, preferido: Optional[str] = None) -> Optional[str]:
    port = seleccionar_puerto(preferido)
    if port is None:
        raise RuntimeError('No hay puertos serial disponibles.')
    return send_state(port, state)


def decide_and_send(results_list: List[Dict], final_class: Optional[str]) -> Tuple[str, Optional[str]]:
    key = str(final_class).lower() if final_class else 'none'
    state = CLASS_TO_STATE.get(key)
    if state is None:
        state = 'sana' if key in ('none', 'nan', '') else 'danada'
    print('Estado a enviar:', state)

    try:
        print('Intentando enviar con enviar_estado_auto...')
        resp = enviar_estado_auto(state, preferido=PREFERRED_PORT)
        return state, resp
    except Exception as exc:
        print('Error con enviar_estado_auto:', exc)
        try:
            port = seleccionar_puerto(PREFERRED_PORT)
            if port is None:
                raise RuntimeError('Sin puertos disponibles para fallback.')
            resp = send_state(port, state)
            return state, resp
        except Exception as fallback_exc:
            print('No fue posible enviar el estado:', fallback_exc)
            return state, None

# ------------------------------------------------------------
# Cámara utilities
# ------------------------------------------------------------

def listar_camaras(max_indices: int = 6) -> List[int]:
    disponibles = []
    for idx in range(max_indices):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                disponibles.append(idx)
                print(f'Cámara {idx}: DISPONIBLE')
            else:
                print(f'Cámara {idx}: abierta pero sin frames')
        else:
            print(f'Cámara {idx}: no disponible')
        cap.release()
    if disponibles:
        print('Índices utilizables:', disponibles)
    else:
        print('No se detectaron cámaras operativas en los primeros', max_indices, 'índices.')
    return disponibles

# ------------------------------------------------------------
# Programa principal
# ------------------------------------------------------------

def main():
    print('--- Pipeline captura + inferencia + envío (best_model.pth) ---')
    listar_camaras(max_indices=3)
    carpeta = motor_y_captura_concurrente()

    model, backend = load_model(MODEL_PATH)
    results_list, image_paths = run_inference(model, backend, carpeta)
    agg = aggregate_results(results_list, carpeta)
    generar_anotaciones(model, backend, image_paths)

    estado, respuesta = decide_and_send(results_list, agg['final_class'])
    print(f'Estado enviado: {estado}; Respuesta: {respuesta}')


if __name__ == '__main__':
    main()
