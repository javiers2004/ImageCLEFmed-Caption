# Harness de evaluación (réplica del script oficial)

Objetivo: reproducir localmente las métricas oficiales para poder iterar sin
depender del portal de la competición. **Montar esto en la Fase 0**, no al final.

Repo oficial: https://github.com/taubsity/clef-caption-evaluation

## Concept Detection
- **F1** (scikit-learn, averaging `binary`), por-muestra, promediado sobre el test.
- Score primario: todos los conceptos. Secundario: solo conceptos anotados a mano.
- Fácil de replicar en Python puro (sin modelos externos).

## Caption Prediction — 6 métricas (Overall = media)

Preprocesado común (BERTScore, ROUGE, BLEURT): minúsculas, números → token
`number`, quitar puntuación, tratar el caption como una sola frase.

| Métrica | Aspecto | Modelo / paquete | Notas de instalación |
|---|---|---|---|
| Image-Caption Similarity | Relevancia | modelo de embedding médico | requiere imagen + caption |
| BERTScore (Recall + idf) | Relevancia | `microsoft/deberta-xlarge-mnli` | pip `bert-score`; idf del corpus de test |
| ROUGE-1 (F) | Relevancia | `rouge-score` | trivial |
| BLEURT-20 | Relevancia | checkpoint `BLEURT-20` | instalar desde repo google-research/bleurt |
| UMLS Concept F1 | Factualidad | MedCAT | requiere modelos UMLS |
| AlignScore | Factualidad | RoBERTa (checkpoint HF) | instalar desde repo AlignScore |

### Instalaciones especiales (en la máquina 3070)
```bash
# BLEURT
pip install git+https://github.com/google-research/bleurt.git
# descargar checkpoint BLEURT-20 (~2 GB)

# AlignScore
pip install git+https://github.com/yuh-zha/AlignScore.git
python -m spacy download en_core_web_sm
# descargar checkpoint AlignScore-base/large

# MedCAT (UMLS) — requiere aceptar licencia UMLS para los modelos
pip install medcat
```

## Plan de implementación
- `src/eval/concept_f1.py`  — F1 primario y secundario.
- `src/eval/caption_metrics.py` — wrappers de las 6 métricas + preprocesado.
- `src/eval/run_eval.py` — CLI: recibe predicciones + ground truth, imprime tabla.

Validar contra el repo oficial con un puñado de ejemplos antes de confiar en los números.
