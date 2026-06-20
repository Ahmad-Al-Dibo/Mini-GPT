# Project Map

## 2026-06-10 - Gestandaardiseerde inference loading

### Aangepast

- `gpt_lib/inference.py` toegevoegd als centrale inference-module.
- `gpt_lib/__init__.py` exporteert nu `load_model`, `predict`, `LoadedModel` en `DEFAULT_MODEL_PATH`.
- `app.py` gebruikt nu de centrale loader in plaats van eigen checkpoint/modelbouw-code.
- `app.py` ondersteunt `MODEL_PATH` en `PORT` als environment variables voor lokale demo's en later deployment.
- `templates/index.html` opgeschoond naar een eenvoudige webdemo met promptveld, knop en output.
- `requirements.txt` aangevuld met de runtime-dependencies die de app en tokenizer-code al gebruiken.
- `tests/test_core.py` bevat nu een test die een checkpoint laadt via de nieuwe inference-interface.

### Waarom

- De loading-code stond eerder verspreid in de Flask-app en was daardoor lastig te hergebruiken.
- De nieuwe interface ondersteunt direct gebruik vanuit andere scripts:
  `load_model().predict("prompt")` of `predict("prompt")`.
- Flask blijft alleen verantwoordelijk voor request/response en UI.

### Architecturale beslissing

- Standaard modelpad: `output/mini_gpt.pth`.
- Checkpoint-formaat: bestaande `.pth` checkpoints blijven ondersteund.
- Inference gebruikt bij voorkeur `best_model_state_dict` als die in het checkpoint staat; anders valt de loader terug op `model_state_dict`.
- Tokenizer-data wordt voorlopig gereconstrueerd uit `stoi` en `itos` in het checkpoint. Voor toekomstige SentencePiece-modellen is het beter om ook het originele tokenizer-model/artifact op te slaan.

### Impact

- Hergebruik wordt eenvoudiger en consistenter.
- Flask start sneller omdat het model lazy-loaded blijft.
- Geen extra modelconversie nodig voor bestaande checkpoints.
- Als `output/mini_gpt.pth` nog niet bestaat, geeft de loader een expliciete foutmelding totdat training klaar is of een ander `model_path` wordt meegegeven.

## 2026-06-10 - Markdown output in webdemo

### Aangepast

- `templates/index.html` toont modeloutput nu standaard als Markdown-preview.
- Er is een toggle toegevoegd voor `Preview` en raw `Markdown`.
- Markdown-rendering gebeurt client-side zonder extra dependency.

### Waarom

- Modelantwoorden met headings, lijsten, quotes en codeblokken zijn beter leesbaar in de demo.
- De raw Markdown blijft beschikbaar om tekst makkelijk te kopieren of verder te verwerken.

### Impact

- Geen backend-wijziging en geen extra package nodig.
- Output wordt voor rendering HTML-escaped om scripts of ongewenste HTML niet uit modeltekst uit te voeren.

## 2026-06-10 - Tokenizer-consistentie tussen training en inference

### Aangepast

- `SentencePieceTokenizer` kan zichzelf nu opslaan als checkpoint metadata via `to_checkpoint()`.
- `SentencePieceTokenizer.from_checkpoint(...)` herbouwt de exacte tokenizer uit het opgeslagen SentencePiece model.
- `save_training_checkpoint(...)` schrijft `tokenizer_metadata` mee naar nieuwe checkpoints.
- `load_model(...)` gebruikt de opgeslagen SentencePiece-tokenizer wanneer die aanwezig is.
- Oude SentencePiece-checkpoints zonder `tokenizer_metadata.model_proto` geven nu een expliciete runtime warning en gebruiken de compatibiliteitsfallback.
- `train.py` geeft `data.tokenizer` nu ook door aan `Generator` voor de sample generatie na training.
- `load_compatible_checkpoint(...)` vergelijkt nu ook de tokenizer mapping bij resume-training, zodat weights niet per ongeluk met andere token IDs worden hergebruikt.
- Tests toegevoegd voor SentencePiece checkpoint round-trip en inference-loader tokenizergebruik.

### Waarom

- `train.py` gebruikt `tokenizer_type="sentencepiece"`.
- De vorige inference fallback gebruikte alleen `stoi`/`itos` en whitespace-splitting. Dat is niet exact hetzelfde als de getrainde SentencePiece-tokenizer.

### Impact

- Nieuwe checkpoints gebruiken in Flask en scripts dezelfde tokenizer als tijdens training.
- Oude checkpoints zonder `tokenizer_metadata` blijven laadbaar via fallback, maar zijn minder exact voor SentencePiece-inference.
- Resume-training weigert voortaan checkpoints waarvan de tokenizer mapping niet overeenkomt met de net gebouwde tokenizer.
