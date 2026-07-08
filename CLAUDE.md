# CLAUDE.md — Contexto del proyecto (léeme primero)

> Este archivo es la memoria compartida del proyecto. Claude lo lee automáticamente
> al abrir el repo, así que sirve para **retomar la conversación en cualquier máquina**
> (por ejemplo, al pasar del portátil de desarrollo al ordenador de ejecución con la RTX 3070).
> **Mantenlo actualizado**: cuando cambie el estado del proyecto, edita la sección "Estado actual".

## Qué es este proyecto
Replicación (y objetivo de **superar**) del reto **ImageCLEFmedical Caption 2025** sobre el
dataset **ROCOv2** (imágenes de radiología de PubMed Central OpenAccess).

Se abordan **las tres tareas**:
1. **Concept Detection** — clasificación multi-etiqueta de conceptos UMLS (CUIs) a partir de la imagen.
2. **Caption Prediction** — generar el texto (caption) que describe la imagen.
3. **Explainability** — justificar/visualizar las captions (evaluado por un radiólogo).

**Objetivos de referencia (mejores runs oficiales, a batir):**
- Concept Detection: **F1 = 0.5888**
- Caption Prediction: **Overall = 0.3432**

Nota: los plazos de la competición ya pasaron (envíos cerraron en mayo 2025). Esto es una
replicación post-hoc con fines de aprendizaje y de superar la tabla. Los datos (ROCOv2) son públicos.

## Dataset ROCOv2
- 80 091 train / 17 277 val / 19 267 test imágenes de radiología.
- Descarga pública (sin registro): https://zenodo.org/records/10821435 (DOI 10.5281/zenodo.10821435).
- Los datos **viven solo en la máquina de ejecución** (no se suben a git). Ruta configurada en `configs/machines.yaml` → `data_dir`.
- Cita obligatoria: Rückert et al., *ROCOv2*, Scientific Data 11(1), 2024, doi:10.1038/s41597-024-03496-6.

## Montaje de dos máquinas
- **Portátil de desarrollo** (GTX 1650 Ti, 4 GB VRAM): se escribe el código y se hacen *smoke tests*. No entrena modelos reales. Hostname: `LAPTOP-VJR4VD4I`.
- **Sobremesa de ejecución** (RTX 3070, 8 GB VRAM + 32 GB RAM): **aquí se entrena e infiere todo lo real**.
- Sincronización del **código** vía GitHub: `https://github.com/javiers2004/ImageCLEFmed-Caption`.
  Flujo: escribir en el portátil → `git push` → en la 3070 `git pull`.
- **Todas las decisiones de tamaño de modelo/batch están limitadas por los 8 GB de la 3070**:
  usar fp16/bf16, gradient accumulation, LoRA/QLoRA para cualquier VLM > ~1B parámetros, batch pequeño.

> ⚠️ **OneDrive**: el repo está dentro de una carpeta OneDrive en el portátil. OneDrive puede
> corromper `.git`. En la 3070, **clonar el repo FUERA de OneDrive**.

## Cómo empezar en una máquina nueva (p. ej. la 3070)
```bash
git clone https://github.com/javiers2004/ImageCLEFmed-Caption.git   # fuera de OneDrive
cd ImageCLEFmed-Caption
conda env create -f environment.yml
conda activate imageclef-caption
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python scripts/check_env.py          # muestra hostname, GPU y config
```
Tras el primer `check_env.py` en la 3070: **copiar su hostname** en `configs/machines.yaml`
(sustituyendo la entrada `EXEC-3070`) y ajustar `data_dir` a la ruta real de ROCOv2.

## Estado actual (actualizar según se avance)
**Fase 0 — Setup (casi completa).**
- [x] Estructura del repo, entorno (`environment.yml`, `requirements.txt`).
- [x] Config por-máquina con detección de hostname (`src/utils/config.py`, `configs/machines.yaml`).
- [x] `scripts/check_env.py` validado en el portátil (detecta la 1650 Ti).
- [x] Métrica **F1 de Concept Detection** implementada y testeada (`src/eval/concept_f1.py`, 9/9 tests en `tests/`).
- [x] Repo en GitHub, primer push hecho.
- [ ] **Descargar ROCOv2** en la 3070 (en curso por el usuario).
- [ ] Preparar entorno en la 3070 + añadir su hostname a `machines.yaml`.
- [ ] Implementar métricas de captions (`src/eval/caption_metrics.py`): preprocesado oficial + BERTScore, ROUGE-1, BLEURT-20, similarity, UMLS-F1, AlignScore.
- [ ] Instalar dependencias pesadas de evaluación en la 3070 (ver `src/eval/README.md`).

**Siguiente paso concreto:** con ROCOv2 descargado, arrancar la **Fase 1 (EDA + pipeline de datos)**.

## Plan por fases (resumen)
Detalle completo en [`docs/PLAN.md`](docs/PLAN.md).
- **Fase 0** — Setup, entorno, datos, evaluación oficial funcionando. *(en curso)*
- **Fase 1** — EDA + pipeline de datos (`src/data/`).
- **Fase 2** — Concept Detection: baseline → fine-tuning (EfficientNet/ViT/BiomedCLIP) → retrieval k-NN → ensemble + umbrales. Objetivo F1 > 0.55.
- **Fase 3** — Caption Prediction: BLIP/GIT → concept-conditioning (usar salidas de Fase 2) → VLM con LoRA/QLoRA. Objetivo Overall > 0.30.
- **Fase 4** — Explainability: Grad-CAM / attention + justificación textual.
- **Fase 5** — Tuning final, ensembles, generación de submissions con formato oficial.
- **Fase 6** — Documentación / working notes.

## Convenciones del repo
- Config: `from src.utils.config import load_config` → fusiona `configs/base.yaml` + config de la máquina detectada por hostname.
- Datos y checkpoints **nunca** en git (ver `.gitignore`). Solo en la 3070.
- Tests con `pytest` (`python -m pytest tests/ -q`).
- Idioma del proyecto: español (el usuario, `javiers2004`, trabaja en español).

## Evaluación (crítico para iterar)
Repo oficial: https://github.com/taubsity/clef-caption-evaluation
- Concept Detection: F1 (sklearn, averaging `binary`), primario (todos los conceptos) y secundario (solo conceptos anotados a mano). **Ya implementado** en `src/eval/concept_f1.py`.
- Caption Prediction: media de 6 métricas (BERTScore recall+idf con `microsoft/deberta-xlarge-mnli`, ROUGE-1 F, BLEURT-20, similarity con modelo de embedding médico, UMLS-F1 con MedCAT, AlignScore). **Pendiente de implementar.** Ver `src/eval/README.md`.
