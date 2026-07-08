# Planificación completa — ImageCLEFmed Caption 2025

Documento de referencia del plan. El estado vivo se lleva en [`../CLAUDE.md`](../CLAUDE.md).

## 0. Objetivo y reality check
Replicar y superar el reto (3 tareas: Concept Detection, Caption Prediction, Explainability)
sobre ROCOv2, con una única RTX 3070 (8 GB) como máquina de ejecución.

| Tarea | Mejor de la tabla | ¿Alcanzable con 3070 8GB? | Dificultad |
|---|---|---|---|
| Concept Detection | F1 0.5888 | Sí, muy alcanzable | Media |
| Caption Prediction | Overall 0.3432 | Alcanzable, ambicioso | Alta |
| Explainability | Cualitativo (radiólogo) | Sí (no requiere entrenar) | Baja-media |

Los ganadores usaron **ensembles** y VLMs grandes (InstructBLIP, Qwen2-VL). Con una sola 3070
no hay ensembles enormes, pero sí se puede igualar Concept Detection y acercarse en Caption
con **concept-conditioning + LoRA/QLoRA**. Techo honesto: top en conceptos es realista;
top-1 en captions es difícil sin más cómputo.

## 1. Arquitectura del proyecto y flujo de dos máquinas
- **GitHub** como fuente única del código (`javiers2004/ImageCLEFmed-Caption`).
- **Datos y checkpoints** solo en la 3070 (nunca en git).
- **`configs/machines.yaml`**: rutas y `batch_size`/`num_workers` por hostname.
- **Modo smoke**: correr con pocas muestras para depurar en el portátil (4 GB) sin GPU real.
- **Entorno reproducible**: `environment.yml` + `requirements.txt` con versiones fijadas.

Estructura:
```
configs/     yaml base + config por-máquina
src/data/    datasets, dataloaders, transforms
src/concepts/  Fase 2
src/captions/  Fase 3
src/explain/   Fase 4
src/eval/    réplica del script oficial de evaluación
scripts/     CLIs de entrenamiento/inferencia/utilidades
tests/       tests unitarios
data/        (gitignored) ROCOv2 — solo en la 3070
checkpoints/ (gitignored) pesos entrenados
submissions/ ficheros con formato oficial
docs/        planificación y notas
```

## 2. Fases

### Fase 0 — Setup e infraestructura
- Repo, entorno, config por-máquina, `check_env.py`. **[hecho]**
- Métrica F1 de Concept Detection + tests. **[hecho]**
- Descargar ROCOv2; preparar la 3070; instalar métricas pesadas de captions; implementar `caption_metrics.py`.

### Fase 1 — EDA y pipeline de datos
- Analizar imágenes (modalidades, tamaños, canales), conceptos (nº total tras filtrado ≈ 1900–2000,
  distribución de frecuencia, nº medio por imagen) y captions (longitud, vocabulario).
- `Dataset`/`DataLoader` reutilizables, transforms, normalización, caché de features si hace falta.
- Usar el split oficial train/val/test (no re-splitear).

### Fase 2 — Concept Detection
De menor a mayor rendimiento:
1. Baseline: encoder congelado (BiomedCLIP/CLIP) + cabeza lineal, BCE, umbral global.
2. Fine-tuning: EfficientNet-B3 / ConvNeXt-Tiny / ViT / BiomedCLIP con cabeza multi-etiqueta (asymmetric loss).
3. Retrieval k-NN: encodear imágenes, agregar conceptos de vecinos (históricamente fuerte en ROCO).
4. Umbrales por-concepto + ensemble de 2–3 modelos.
- **Objetivo:** F1 > 0.55, meta 0.58+.

### Fase 3 — Caption Prediction (la más pesada)
1. Baseline generativo: fine-tune de BLIP-base o GIT-base (caben en 8 GB con fp16).
2. Concept-conditioning: inyectar los conceptos de la Fase 2 como prefijo/prompt → mejora factualidad (UMLS-F1, AlignScore).
3. VLM con LoRA/QLoRA: Qwen2-VL-2B o InstructBLIP (LLM congelado) en 4-bit.
4. Harness de evaluación local con las 6 métricas para iterar.
- **Objetivo:** Overall > 0.30, meta 0.34+.

### Fase 4 — Explainability
- Grad-CAM / attention-rollout: qué regiones justifican cada concepto/frase.
- Heatmap superpuesto + breve justificación textual. No requiere entrenamiento nuevo.

### Fase 5 — Tuning final y submissions
- Ensembles, tuning de umbrales, análisis de errores.
- Generar ficheros con el formato exacto de cada subtask.

### Fase 6 — Documentación
- Registro de experimentos (Weights & Biases o CSV), tablas comparativas, borrador de working notes.

## 3. Stack técnico
- Core: PyTorch, `timm`, HuggingFace `transformers`/`peft`/`accelerate`/`bitsandbytes`, `open_clip`/BiomedCLIP.
- Datos: `pandas`, `Pillow`/`opencv`, `albumentations`.
- Eval: `bert-score`, `rouge-score`, BLEURT, AlignScore, `medcat`/`quickumls`.
- Tracking: Weights & Biases.

## 4. Riesgos y mitigación
- 8 GB VRAM es el cuello de botella → QLoRA 4-bit, gradient checkpointing, batch pequeño + acumulación.
- Métricas de eval pesadas de instalar (BLEURT/AlignScore/MedCAT) → resolver en Fase 0, no al final.
- Sin ensembles grandes no se llega al top-1 en Caption → compensar con concept-conditioning (factualidad).
