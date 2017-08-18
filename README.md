# Clasificador NER con Linear Support Vector Machine

### script para generar dataset
- /dataset/generador_dataset.py

### script para generar serializaciones
- /ner/ner.py

### script para clasificar oraciones
- /ner/intent_trainer.py

### En la carpeta ner se encuentran todos los pkl necesarios para la clasificacion de texto

- lin_svc.pkl
- selector.pkl
- selector_intent.pkl
- vectorizer.pkl
- vectorizer_intent.pkl

### ejecuta el script /ner/intent_trainer.py para clasificar texto dentro de algono de los codominios

```bash
$ python intent_trainer.py
``` 
