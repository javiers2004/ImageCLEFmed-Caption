# ImageCLEFmed Caption 2025 — Concept Detection + Caption Prediction + Explainability

Replicación (y objetivo de superar) del reto ImageCLEFmedical Caption 2025 sobre
el dataset **ROCOv2** (80 091 train / 17 277 val / 19 267 test imágenes de radiología).

**Objetivos de referencia (mejores runs oficiales):**
- Concept Detection: F1 **0.5888**
- Caption Prediction: Overall **0.3432**

## Flujo de dos máquinas
- **Portátil (GTX 1650 Ti, 4 GB):** desarrollo y *smoke tests*. No entrena modelos reales.
- **Sobremesa (RTX 3070, 8 GB + 32 GB RAM):** todo el entrenamiento/inferencia real.
- Sincronización de **código** vía GitHub. **Datos y checkpoints** viven solo en la 3070.

> ⚠️ **OneDrive:** este repo está dentro de una carpeta OneDrive en el portátil. OneDrive
> puede corromper `.git` al sincronizar. En la máquina 3070 clona el repo **fuera** de
> OneDrive. En el portátil, considera excluir la carpeta de la sincronización.

## Setup (en cada máquina)
```bash
# 1. Entorno
conda env create -f environment.yml
conda activate imageclef-caption

# 2. PyTorch con CUDA (Ampere/Turing -> cu121)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Resto de dependencias
pip install -r requirements.txt

# 4. Comprobar entorno (muestra hostname, GPU y config resuelta)
python scripts/check_env.py
```
Tras el primer `check_env.py` en la 3070, copia su **hostname** en `configs/machines.yaml`
(sustituyendo la entrada `EXEC-3070`) y ajusta `data_dir` a la ruta real de ROCOv2.

## Estructura
```
configs/     yaml base + config por-máquina (machines.yaml)
src/data/    datasets, dataloaders, transforms
src/concepts/  Fase 2 — Concept Detection
src/captions/  Fase 3 — Caption Prediction
src/explain/   Fase 4 — Explainability
src/eval/    réplica del script oficial de evaluación (ver src/eval/README.md)
scripts/     CLIs de entrenamiento/inferencia/utilidades
data/        (gitignored) ROCOv2 — solo en la 3070
checkpoints/ (gitignored) pesos entrenados
submissions/ ficheros con formato oficial
```

## Fases
- **Fase 0** — Setup, entorno, descarga ROCOv2, evaluación oficial funcionando. *(en curso)*
- **Fase 1** — EDA + pipeline de datos.
- **Fase 2** — Concept Detection (baseline → fine-tuning → retrieval → ensemble).
- **Fase 3** — Caption Prediction (BLIP/GIT → concept-conditioning → VLM con LoRA/QLoRA).
- **Fase 4** — Explainability (Grad-CAM / attention + justificación textual).
- **Fase 5** — Tuning final, ensembles y generación de submissions.
- **Fase 6** — Documentación / working notes.

## Datos: ROCOv2
Descargar desde la web de ImageCLEF / Zenodo (requiere registro). Colocar en la ruta
`data_dir` configurada en `machines.yaml`. No subir a git.
Cita obligatoria del dataset: Rückert et al., *ROCOv2*, Scientific Data 11(1), 2024,
doi:10.1038/s41597-024-03496-6.
