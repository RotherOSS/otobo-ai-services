# OTOBO-AI

Ticket Answering Service

![Screenshot](/img/Screenshot.png)

## Beschreibung

Erzeugt eine Schnittstelle, die sich in das OTOBO-Ticketsystem integrieren lässt.\
Diese Schnittstelle ermöglicht es, Anfragen anhand bestehender Tickets beantworten zu lassen (`/llm-call/invoke/`).\
Dazu werden zuerst bestehende Tickets aufbereitet (embedded) und in einer Vektor-Datenbank gespeichert (`/embedding/insert/`).

Für die Speicherung der Embeddings wird zur Zeit Chroma-DB benutzt.

Die API verwendet einen API-Key zur Authentifizierung.

Die Anwendung kann grundsätzlich unterschiedliche Sprachmodelle verwenden, ist aber zur Zeit auf Llama2 optimiert.

Die Dokumentation der API lässt sich nach Installation leicht aufrufen:

1. [Swagger](http://127.0.0.1:8000/docs#/) (<http://127.0.0.1:8000/docs#/> , falls unter dieser Adresse und Port gestartet)
2. [ReDoc](http://127.0.0.1:8000/redoc), (<http://127.0.0.1:8000/redoc> , falls unter dieser Adresse und Port gestartet)

## Aufbau

### Das Sytem besteht aus 3 Elementen

1. API mit den unten beschriebenen Routes
1. Docker mit Chroma DB zur Speicherung des vektorisierten Inhalts
1. LLM

## Installation

### Installation Datenbank

#### Elasticsearch

Um Elasticsearch in einem Docker zu laden verwenden Sie

```bash
docker run -p 9200:9200 -e "discovery.type=single-node" -e "xpack.security.enabled=false"
-e "xpack.security.http.ssl.enabled=false" --name es_812 -d docker.elastic.co/elasticsearch/elasticsearch:8.12.1
```

### Installation API

Sie können diesen API-Server in einem Docker-Container betreiben.\
Stellen Sie sicher, dass alle Umgebungsvariablen korrekt gesetzt sind. \

#### Netzwerk

Datenbank und TAS-Service müssen sich in dem selben virtuellen Netzwerk befinden. Dazu muss ein Netzwerk erstellt werden

```bash
docker network create otobo_tas
```

Dies sollte nun aufgeführt werden

```bash
docker network ls
```

Der Datenbank-Container muss im selben Netzwerk laufen (sh. oben), Etwa:

```bash
docker run --network otobo_tas ...
```

Ein laufender Container kann nachträglich zum Netzwerk hinzugefügt werden

```bash
docker network connect otobo_tas <container-id (Anfang...)>
```

So finden Sie die Ip-Adresse, die der Container im virtuellen Netzwerk nutzt.

```bash
docker inspect <container-id (Anfang...)>
```

Unten im Output befindet sich die Ip-Adresse unter `Networks->otobo_tas->IPAdress`.

Das Docker-Image wird im Verzeichnis, in dem sich die Datei `Dockerimage` mit diesem Befehl erzeugt (Dauer ca. 5 Minuten):

```bash
docker build -t ki-werkstatt/otobo:latest .
docker run --env-file .\.env -p 8080:8080 --name OTOBO -d ki-werkstatt/otobo:latest
```

bei Problemen auch

```bash
lock-Datei neu erstellen: poetry lock --no-cache
sauber erstellen: docker build --no-cache -t ki-werkstatt/otobo:latest .
```

Nachdem die Datenbank gestartet wurde (Wichtig!), kann mit diesem Befehl nun der Service gestartet werden:

```bash
docker run -d -e AI_API_KEY=ServiceApiKey -e AI_VECTORDB_AUTH_TOKEN=databaseToken -e CHROMADB_HOST=ipDatenbankInDockerNetwork -e SERVER_PORT=8080 --network=otobo_tas -p 8080:8080 --name OTOBO <image-id (Anfang...)>
```

1. ServiceApiKey: Frei definierbarer Token, der später für den Aufruf
1. databaseToken: CHROMA_SERVER_AUTH_CREDENTIALS, sh oben
1. ipDatenbankInDockerNetwork: Ip des Datenbankservers im virtuellen Netzwerk oder `network alias`
1. <image-id (Anfang...)>: Docker-Image-Id

### Installation Sprachmodell Llama2

Die API wurde mit Llama2 13b getestet.\
[Ollama für Mac](https://ollama.ai/library/llama2)

## Inbetriebnahme

Nachdem alle Komponenten installiert wurden, können diese getestet werden.
Setzen Sie zuerst die Umgebungsvariablen.
Die Datei .env_example enthält alle notwendigen Umgebungsvariablen.\
Hier einige Beispielwerte:

```bash
AI_API_SERVER_HOST="0.0.0.0"
AI_API_SERVER_PORT="8000"

AI_VECTORDB_HOST="localhost"
AI_VECTORDB_PORT="8000"
AI_VECTORSTORE_INDEX="mytickets"

AI_API_KEY="<use a secret API key>"

LLM_OTOBO_API_KEY="secret-to-use-otobo-ai-server"
LLM_OLLAMA_URL="<url-from-model>"
LLM_OLLAMA_MODEL=llama2:13b-chat

```

Ist der Docker des API-Servers gestartet, können Sie auf den\
[Swagger](http://127.0.0.1:8000/docs#/) (<http://127.0.0.1:8000/docs#/)\>
falls unter dieser Adresse und Port gestartet - zugreifen.

## Entwicklung

1. Download von git
1. Virtuelle Umgebung einrichten
1. `.env` erzeugen
1. Umgebungsvariablen setzen
1. Server starten

## Virtuelle Umgebung einrichten in VS Code

\<strg> + \<shift> + p: Python:Create vitual enviroment wählen, dann venv, dann Interpreter, dann pyproject.toml eingeben. Installation aller Pakete startet parallel im Hintergrund. Abwarten, bis alles installiert ist, dann erst starten.

Aus der Entwicklungsumgebung startet der Server mit:

```bash
langchain serve
```

Beenden mit \<strg> + c

Beispiel eines Aufrufs durch einen Client mit Auth:

```bash
import requests

api_url = "http://127.0.0.1:8000/embedding/insert/"
data = [] # create a list of Dict of type Ticket instead

headers = {
  'Content-Type': 'application/json',
  'access_token': '<very_secret_token_here>'
}

response = requests.put(api_url, json=data, headers=headers)
```

Mit der Route `heartbeat` lässt sich prüfen, ob der API-Server grundsätzlich funktioniert und die DB erreichbar ist.\
Für den Aufruf von heartbeat ist keine Authentifizierung nötig.

Geben Sie nun unter `Authorize` den API-Key ein. Danach stehen auch alle geschützten Routen für Tests zur Verfügung.

Das System ist nun einsatzbereit.

Sollte es Probleme mit der Python-version geben:
Windows: pyenv installieren mit ```Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"```

dann ```pyenv install 3.11.9
poetry config virtualenvs.prefer-active-python true
pyenv local 3.11```

## Hinweis 2

Erhält man folgende Fehlermeldung, ist die Datenbank bzw. der Index leer.

```
"detail": "NotFoundError(404, 'index_not_found_exception', 'no such index [api-test]', api-test, index_or_alias)"
```

LangSmith ist auch nicht mehr in dieser Form nicht mehr kostenlos nutzbar. Bei der Fehlermeldung

```
Failed to multipart ingest runs: langsmith.utils.LangSmithAuthError
```

bitte folgende .env Variablen löschen:

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls_...
LANGCHAIN_PROJECT=OTOBO-AI
```

## Bugfix

Mega-fieser Bug in Langchain. Ums kurz zu fassen. In den Tiefen der Runnables überschreibt Langchain die Runnabe_Config, so dass wenn du mit LangGraph einen Agent baust und ihn über LangServe bereitstellst, wird plötzlich von LangFuse (ich weiß, viele Langs) nicht mehr mitgeloggt. \
Das waren ein paar stressige Tage...

Grund: In ```.../lib/python3.12/site-packages/langgraph/utils/config.py```  bzw. im Container
```/usr/local/lib/python3.12/site-packages/langgraph/utils/config.py``` überschreibt ensure_config die Callbackhandler.

### Workaround

Im Ordner bugfix liegt eine modifizierte config.py. Diese muss in den Container kopiert werden NACHDEM poetry gelaufen ist:
```COPY ./bugfix/config.py /usr/local/lib/python3.12/site-packages/langgraph/utils/config.py```

## Langfuse

Langfuse klinkt sich in in den Workflow zum Erzeugen von Antworten mit KI ein und bereitet die Daten so auf, dass man Input und Output besser nachverfolgen kann.

![Screenshot](/img/langfuse.png)

### Installation Langfuse

[Langfuse im Docker installieren](https://langfuse.com/self-hosting/docker)

Dies installiert einen Docker-Stack inkl. Datenbank und Web-Frontend.

### Erste Schritte

Ist Langfuse installiert, richtet man über die Web-Oberfläche ein Projekt ein. Dort werden auch die nötigen Secret- und Public-Keys erzeugt.

### Enviroment-Parameter

In der otobo-api werden dann diese Keys in Enviroment-Variablen hinterlegt.

LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_HOST=http://(ip-Adresse/Servername):3000

## Updating Dependencies

To update the dependencies of the Docker Compose project, follow this procedure to ensure compatibility and reproducibility:

### 1. Unpin Non-Critical Dependencies

In `requirements.txt`, remove version pins from most packages. Keep pins **only** for critical compatibility fixes.

**Before:**
```txt
uvicorn==0.25.0
fastapi==0.108.0

# critical compatibility fixes
numpy==1.26.4
```

**After:**
```txt
uvicorn
fastapi

# critical compatibility fixes
numpy==1.26.4
```

### 2. Rebuild the Containers

Run:
```bash
docker-compose build
```

### 3. Start the Project

Run:
```bash
docker-compose up
```

Ensure everything starts properly.

### 4. Test Functionality

Test the application as usual. If there are version-related issues, resolve them by pinning the necessary libraries in `requirements.txt`.

### 5. Get Installed Versions

After confirming the application works, extract the actual installed versions:
```bash
docker-compose run --rm otobo-ai pip list
```

### 6. Pin All Dependencies

Use the output of `pip list` to update `requirements.txt`, pinning each used library to its current version.

Example:
```txt
uvicorn==0.34.0
fastapi==0.115.0

# critical compatibility fixes
numpy==1.26.4
```

### 7. Commit the Updated Requirements

Once confirmed working, commit the updated `requirements.txt`:

```bash
git add requirements.txt
git commit -m "Update pinned dependencies"
```
