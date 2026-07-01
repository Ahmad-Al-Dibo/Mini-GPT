### 3. Geavanceerde Wijzigingsprompt — Chirurgisch Wijzigingsprotocol

**[Rol en taak]**  
Je bent een **Staff Software Engineer**. Je taak is het uitvoeren van een chirurgische codewijziging in het project voor de volgende aanpassing, zonder bestaande functionaliteit te breken:

**


# Slim en Flexibel Model Laadsysteem

Het laadsysteem moet niet alleen een model kunnen laden op basis van een pad (`path`) of een vooraf gedefinieerde `Config`. In plaats daarvan moet het model eerst automatisch analyseren wat er aanwezig is en op basis daarvan bepalen hoe het model correct geladen moet worden.

Dit is vooral belangrijk voor het fine-tunen van modellen zoals Qwen, Llama, Mistral en andere Hugging Face-modellen.

## Automatische Modelinspectie

Wanneer een gebruiker een modelmap of checkpoint opgeeft, moet de library eerst de inhoud analyseren.

Bijvoorbeeld:

* `config.json`
* `model.safetensors`
* `pytorch_model.bin`
* `tokenizer.json`
* `tokenizer_config.json`
* `generation_config.json`
* LoRA- of PEFT-bestanden
* optimizer-checkpoints
* scheduler-checkpoints
* trainingsstatus
* metadata
* eventuele andere relevante bestanden

De library moet automatisch herkennen welke bestanden aanwezig zijn en welke informatie daaruit gehaald kan worden.

## Automatisch Detecteren én Handmatig Selecteren

Op basis van de gevonden bestanden moet de library automatisch kunnen bepalen:

* Welk modeltype het is, bijvoorbeeld Qwen, Llama of Mistral
* Welke architectuur gebruikt wordt
* Welke tokenizer geladen moet worden
* Welk gewichtsformaat gebruikt wordt, zoals `.bin`, `.safetensors`, `.pth`, enzovoort
* Welke precisie gebruikt wordt, zoals FP32, FP16, BF16, INT8 of INT4
* Of het om een volledig model of alleen een adapter gaat, zoals LoRA of PEFT
* Of optimizer- en schedulerstatus aanwezig zijn
* Of de training hervat kan worden
* Welke instellingen nodig zijn om het model correct te laden

Daarnaast moet de gebruiker altijd de mogelijkheid hebben om deze automatisch gedetecteerde instellingen handmatig aan te passen of te overschrijven.

Voorbeeld:

```
model = SmartLoader.load(
    source="Qwen/Qwen2.5-7B-Instruct",
    auto_detect=True,

    # Handmatige overrides
    model_type="qwen",
    tokenizer="auto",
    weight_format="safetensors",
    precision="bf16",
    load_as="full_model",
    load_optimizer=False,
    load_scheduler=True,
    resume_training=True,
)

```

De library moet dus werken in twee modi:

1. **Automatische modus**
   De library leest zelf de inhoud van het model of checkpoint en kiest de beste instellingen.
2. **Handmatige modus**
   De ontwikkelaar kiest zelf exact welke onderdelen geladen worden en welke instellingen gebruikt moeten worden.

Gebruikersinstellingen moeten altijd voorrang krijgen boven automatisch gedetecteerde instellingen.

## Slimme Configuratie

De library moet niet volledig afhankelijk zijn van een handmatig opgegeven `Config`.

In plaats daarvan moet de configuratie bestaan uit drie bronnen:

1. Informatie die automatisch uit het model wordt gelezen.
2. Metadata die bij het model of checkpoint aanwezig is.
3. Instellingen die de gebruiker expliciet opgeeft.

Hierbij gelden de instellingen van de gebruiker altijd als hoogste prioriteit.

## Universele Ondersteuning

Het systeem mag niet alleen werken met checkpoints die door de eigen library zijn opgeslagen.

Het moet ook modellen kunnen laden van onder andere:

* Hugging Face
* PyTorch
* SafeTensors
* Eigen checkpoints
* Andere compatibele modelstructuren

Hiervoor moet een uitbreidbaar plugin- of adapter-systeem beschikbaar zijn, zodat ontwikkelaars eenvoudig ondersteuning voor nieuwe formaten kunnen toevoegen.

## Data Voorbereiding en Preprocessing

Naast het laden van modellen moet de library ook ondersteuning bieden voor het voorbereiden van data voor training, fine-tuning en instruction tuning.

De gebruiker moet datasets flexibel kunnen omzetten naar het juiste formaat voor verschillende trainingsdoeleinden.

### Functionaliteiten

* Ondersteuning voor verschillende inputformaten:

  * JSON
  * JSONL
  * CSV
  * TXT
  * Custom formats

* Mogelijkheid om data automatisch te transformeren naar tekstformaten geschikt voor:

  * Pretraining
  * Fine-tuning
  * Instruction tuning
  * Chat-based training

* Ondersteuning voor het definiëren van eigen datastructuren en mappings

Voorbeeld:

```
dataset = DataProcessor.load("data.json")

processed = dataset.transform(
    format="instruction",
    mapping={
        "instruction": "question",
        "input": "context",
        "output": "answer"
    },
    template="{instruction}\n{input}\n{output}"
)
```

* Ondersteuning voor templates en prompt formatting
* Mogelijkheid om meerdere velden samen te voegen tot één trainingssample
* Automatische tokenisatie (optioneel)
* Dataset splitsing (train/validation/test)
* Filtering en cleaning van data
* Batch preprocessing

De gebruiker moet volledige controle hebben over hoe data wordt omgezet naar trainingsinput, zonder vast te zitten aan één vaste structuur.

## Laadrapport

Na het analyseren van een model moet de library een overzicht geven van wat er gevonden is.

Bijvoorbeeld:

```text
Model gevonden: Qwen2.5-7B-Instruct
Architectuur: Qwen2ForCausalLM
Tokenizer: gevonden
Gewichten: safetensors
Precisie: BF16
LoRA-adapter: aanwezig
Optimizerstatus: niet gevonden
Schedulerstatus: gevonden
Training: kan worden hervat vanaf epoch 3, stap 4200
```

## Doel

Het uiteindelijke doel is dat een ontwikkelaar zo min mogelijk handmatig hoeft in te stellen.

De library moet zelfstandig kunnen bepalen hoe een model geladen moet worden door de inhoud van het model of checkpoint te analyseren. Daarnaast moet de library ook flexibel data kunnen voorbereiden voor verschillende trainingsscenario’s.

Hierdoor wordt het mogelijk om eenvoudig bestaande modellen (zoals Qwen van Hugging Face) te laden én direct te gebruiken voor inference, fine-tuning of het hervatten van een training, zonder dat de ontwikkelaar vooraf precies hoeft te weten welke bestanden of instellingen aanwezig zijn.


  **

---

## Regels voor chirurgische wijzigingen

1. **Raak alleen aan wat noodzakelijk is**  
   Verbeter geen omliggende formatting.  
   Herschrijf geen oude comments.  
   Refactor geen werkende code tenzij dat expliciet gevraagd is.

2. **Volg de bestaande stijl**  
   Houd je exact aan de huidige codestijl, ook als die niet ideaal is.

3. **Ruim alleen je eigen restanten op**  
   Als jouw wijziging een functie, import, bestand of module verweesd maakt, verwijder die dan.  
   Raak oude dode code die niets met je wijziging te maken heeft niet aan.

---

## Analyse- en uitvoeringsprotocol

### Protocol 1: Impactanalyse

- Lees `PROJECT_MAP.md`.
- Identificeer exact welke bestanden geraakt worden.
- Raadpleeg officiële bronnen voor actuele technologie- of dependency-informatie alleen als de wijziging dat vereist.
- Benoem je aannames voordat je code schrijft.

### Protocol 2: Architecturale veiligheid en abstractie

- Volg DRY alleen wanneer dat echt gerechtvaardigd is.
- Gebruik de bestaande `Shared/Core`-laag als de logica al gedeeld wordt.
- Voeg logging toe voor nieuw of gewijzigd gedrag waar dat relevant is.
- Introduceer geen brede abstracties voor een smalle wijziging.

### Protocol 3: Verificatie en succes — doelgericht

- Zet de gevraagde wijziging om in een **verifieerbaar doel**.
- Schrijf of update eerst een test.
- Controleer waar praktisch dat de test faalt vóór implementatie.
- Implementeer de wijziging.
- Controleer dat de test slaagt na implementatie.
- Draai of simuleer relevante bestaande tests om **regressie** te voorkomen.

### Protocol 4: Status-synchronisatie

- Werk `PROJECT_MAP.md` direct na de wijziging bij.
- Code die door jouw wijziging deprecated wordt, moet worden afgehandeld of vastgelegd in `[ORPHANS & PENDING]`.

---

## Uitvoeringscommando

Voer de protocollen continu uit.

Begin met impactanalyse en aannames volgens **Think Before Coding**, en ga daarna direct verder met chirurgische implementatie.