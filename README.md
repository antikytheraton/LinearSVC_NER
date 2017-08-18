# Clasificador NER con Linear Support Vector Machine

### script para generar dataset
- /dataset/generador_dataset.py
```bash
$ python generador_dataset.py
```

### script para generar serializaciones
- /ner/ner.py
```bash
$ python ner.py
```

### script para clasificar oraciones
- /ner/intent_trainer.py
```bash
$ python intent_ner.py
```

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
 ## Documentacion intent_trainer.py


- De /ner/intent_trainer.py
```python
obtener_respuesta(frase):
```

- recibe como parametro frase como un string
- devuelve un string '0' si la pregunta es "Quien eres?"
- devuelve un string '1' si la pregunte es "De donde vienes?"
- devielve un string '2' si la pregunta es "Hola bot"
- devielve un string '3' si la pregunta es "no endiendo tu pregunta"
``` 
