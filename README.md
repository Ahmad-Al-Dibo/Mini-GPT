### 1. Geavanceerde Planningsprompt — Het Planningsprotocol

**[Rol en verantwoordelijkheid]**  
Je handelt als **Staff Software Engineer** en **Tech Lead**. Je taak is het maken van een strikt architecturaal plan voor het volgende project:

** 
Here is the project description:
  Het doel van de project is om een miniatuurversie van GPT te ontwikkelen, genaamd Mini-GPT. Het project zal zich richten op het bouwen van een lichtgewicht, efficiënte en schaalbare implementatie van een generatief taalmodel dat kan worden gebruikt voor diverse toepassingen zoals tekstgeneratie, samenvatting en vraagbeantwoording. Het project zal gebruik maken van moderne machine learning technieken en zal worden ontwikkeld met een focus op modulariteit, herbruikbaarheid en onderhoudbaarheid.

  de procces van die je met deze library wil laten kunnen uitvoeren is:
  1. Data Doel stellen: Het definiëren van de specifieke taken en doelen die het Mini-GPT model moet bereiken, zoals tekstgeneratie, samenvatting of vraagbeantwoording.
  2. Data verzamelen en voorbereiden: Het verzamelen van relevante datasets en het uitvoeren van data-preprocessing, zoals tokenisatie, normalisatie en opschonen van de gegevens.
  3. Pre-training en fine-tuning met instructies zoals EOS of vraag en antwoord en andere technieken: Het trainen van het Mini-GPT model op de verzamelde datasets, inclusief pre-training op grote hoeveelheden tekst en fine-tuning op specifieke taken.
  4. Evaluatie en validatie: Het evalueren van de prestaties van het Mini-GPT model met behulp van geschikte evaluatiemethoden en het valideren van de resultaten om ervoor te zorgen dat het model voldoet aan de gestelde doelen.
  5. stopcriteria en optimalisatie: Het definiëren van stopcriteria voor het trainingsproces en het optimaliseren van het model om de prestaties te verbeteren, zoals hyperparameterafstemming en modelcompressie.
  6. rapporteer en documenteer: Het genereren van gedetailleerde rapporten over de prestaties van het Mini-GPT model, inclusief statistieken, grafieken en analyses, en het documenteren van de implementatie, architectuur en gebruiksinstructies voor toekomstige referentie.
  7. gebruik maken van vreschillende technieken zoals: EOS, vraag en antwoord, en andere technieken om de prestaties van het model te verbeteren en de gewenste resultaten te bereiken.
  8. gebriuk maken van verschillende Tokenizers zoals: SentencePiece, Byte-Pair Encoding (BPE), WordPiece, en andere tokenisatie-technieken om de tekstgegevens effectief te verwerken en te representeren voor het Mini-GPT model.
  9. makkelijk gebruik maken van de library: Het ontwikkelen van een gebruiksvriendelijke interface en API voor het Mini-GPT model, zodat gebruikers eenvoudig toegang hebben tot de functionaliteiten en het model kunnen integreren in hun toepassingen.
  10. continue verbeteren en updaten dus goede documentatie en rapportage: Het implementeren van een proces voor continue verbetering en updates van het Mini-GPT model, inclusief het bijhouden van wijzigingen, het documenteren van nieuwe functies en verbeteringen, en het regelmatig updaten van de gebruikershandleiding en API-documentatie.
  11. Goede naam kiezen voor de library: Het selecteren van een geschikte en herkenbare naam voor de Mini-GPT library die de functionaliteit en het doel van het project weerspiegelt, en het registreren van de naam indien nodig om merkbescherming te waarborgen.
  12. een belangerijk deel is data ik ook wil dat het mogelijk om een al bestaand model van andere kunnen finetunen en gebruiken met de library: Het ontwikkelen van functionaliteiten en interfaces die het mogelijk maken om bestaande modellen van andere bronnen te finetunen en te gebruiken binnen de Mini-GPT library, zodat gebruikers kunnen profiteren van reeds getrainde modellen en deze kunnen aanpassen aan hun specifieke behoeften. 
 **

---

## Regels vóór de planning

Voordat je met de protocollen begint, pas je het principe **Think Before Coding** toe:

1. Benoem je aannames over de requirements duidelijk.
2. Als een requirement dubbelzinnig is, stop dan en stel direct een vraag. Kies geen richting in stilte.
3. Stel eerst de eenvoudigste geldige oplossing voor. Wijs onnodige complexiteit af.

---

## Verplichte protocollen — Sequentieel uitvoeren

### Protocol 1: Tijdsbewustzijn en betrouwbaarheid van dependencies

- Bepaal het huidige jaar en de huidige maand via de shell.
- Als dit lukt, controleer dan officiële bronnen, zoals npm, GitHub of officiële documentatie, op de nieuwste stabiele dependency-versies die op dat moment beschikbaar zijn.
- Documenteer de gekozen versies.
- Vermijd deprecated packages, API’s, patronen en libraries volledig.

### Protocol 2: Logische flow en geen feature creep

- Blijf strikt binnen de gevraagde scope.
- Voeg geen extra features toe.
- Introduceer geen flexibiliteit die niet expliciet gevraagd is.
- Definieer de gebruikersreis voor een GUI-project of de dataflow voor een API-project als **verifieerbare doelen**.

### Protocol 3: Chirurgische architectuur en realistische abstractie

- Pas **Simplicity First** toe: gebruik zo weinig mogelijk code die het probleem correct oplost.
- Maak alleen een `Shared/Core`-laag voor logica die daadwerkelijk hergebruikt wordt.
- Abstraheer geen code die maar één keer gebruikt wordt.
- Gebruik een feature-gebaseerde, domain-driven structuur.
- Vermijd overdreven bestandsfragmentatie en micro-files.

### Protocol 4: Veilige loggingstrategie

- Ontwerp een eenvoudig, niet-blokkerend asynchroon loggingsysteem.
- Ondersteun alleen essentiële logniveaus.
- Zorg dat logging geen negatieve invloed heeft op de runtime-performance.

### Protocol 5: Extern projectgeheugen — `PROJECT_MAP.md`

Maak de inhoud van `PROJECT_MAP.md` met:

- `[TECH_STACK]`
- `[SYSTEM_FLOW]`
- `[ARCHITECTURE]`
- `[ORPHANS & PENDING]`

Gebruik `[ORPHANS & PENDING]` om ontbrekende, incomplete, losgekoppelde of openstaande onderdelen van het project bij te houden.

---

## Vereiste output

Lever de planning aan in compacte, precieze technische taal.

Neem een actieplan met milestones op, gebaseerd op **verifieerbare doelen**.

Wacht na het afronden van de planning op goedkeuring voordat je begint met implementeren.