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

1. [Swagger](http://127.0.0.1:8000/docs#/) (http://127.0.0.1:8000/docs#/ , falls unter dieser Adresse und Port gestartet)
2. [ReDoc](http://127.0.0.1:8000/redoc), (http://127.0.0.1:8000/redoc , falls unter dieser Adresse und Port gestartet)

## Aufbau

### Das Sytem besteht aus 3 Elementen

1. API mit den unten beschriebenen Routes
1. Docker mit Chroma DB zur Speicherung des vektorisierten Inhalts
1. LLM

## Hinweis

Im KI-Bereich sind nahezu alle Tools und Bibliotheken noch mit Versionen 0.x bezeichnet. Da es zum Zeitpunkt der Fertigstellung keine Alternativen gab, werden diese Bibliotheken hier verwendet. U.A. LangChain und ChromaDB.

## Installation

### Installation Datenbank

#### Elasticsearch

Um Elasticsearch in einem Docker zu laden verwenden Sie

```bash
docker run -p 9200:9200 -e "discovery.type=single-node" -e "xpack.security.enabled=false" \
-e "xpack.security.http.ssl.enabled=false" docker.elastic.co/elasticsearch/elasticsearch:8.9.0
```

#### Chroma Server

Sie können einen Chroma-Server in einem Docker-Container betreiben.

```bash
docker pull chromadb/chroma
```

nur DB

```bash
docker run -p 8000:8000 chromadb/chroma
```

oder im Netzwerk mit `network alias` und `named volume`

```bash
docker run -p 8000:8000 --network otobo_tas --network-alias chromaserver -v chroma-data:/chroma/chroma --name chromaserver chromadb/chroma
```

oder mit API-Key Auth und Netzwerk ("test-token" durch entsprechenden Wert ersetzen)

```bash
docker run -p 8000:8000
--network otobo_tas
--network-alias chromaserver
-v chroma-data:/chroma/chroma
--env=CHROMA_SERVER_AUTH_CREDENTIALS=test-token
--env=CHROMA_SERVER_AUTH_CREDENTIALS_PROVIDER=chromadb.auth.token.TokenConfigServerAuthCredentialsProvider
--env=CHROMA_SERVER_AUTH_PROVIDER=chromadb.auth.token.TokenAuthServerProvider
chromadb/chroma
```

Sie können das Docker-Image auch selbst aus dem Dockerfile im Chroma GitHub Repository erstellen

```bash
git clone git@github.com:chroma-core/chroma.git
cd chroma
docker-compose up -d --build
```

Eine Anleitung finden Sie [hier](https://docs.trychroma.com/deployment).

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

#

So finden Sie die Ip-Adresse, die der Container im virtuellen Netzwerk nutzt.

```bash
docker inspect <container-id (Anfang...)>
```

Unten im Output befindet sich die Ip-Adresse unter `Networks->otobo_tas->IPAdress`.

#

Das Docker-Image wird im Verzeichnis, in dem sich die Datei `Dockerimage` mit diesem Befehl erzeugt (Dauer ca. 5 Minuten):

```bash
docker build -t ki-werkstatt/otobo:latest .
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

# Inbetriebnahme

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
[Swagger](http://127.0.0.1:8000/docs#/) (http://127.0.0.1:8000/docs#/)\
falls unter dieser Adresse und Port gestartet - zugreifen.

# Entwicklung

1. Download von git
1. Virtuelle Umgebung einrichten
1. `.env` erzeugen
1. Umgebungsvariablen setzen
1. Server starten

### Virtuelle Umgebung einrichten in VS Code

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
