# Bugfix

In den Tiefen der Runnables überschreibt Langchain die Runnabe_Config, so dass wenn du mit LangGraph einen Agent baust und ihn über LangServe bereitstellst, wird plötzlich von LangFuse (ich weiß, viele Langs) nicht mehr mitgeloggt. \

Grund: In ```.../lib/python3.12/site-packages/langgraph/utils/config.py``` überschreibt ensure_config die Callbackhandler

## Workaround

Im Ordner bugfix liegt eine modifizierte config.py. Diese muss in den Container kopiert werden NACHDEM poetry gelaufen ist:
```COPY ./bugfix/config.py /usr/local/lib/python3.12/site-packages/langgraph/utils/config.py```

## Wichtig

config.py ist eine Kopie der in diesem Monent aktuellen Version von langgraph und muss geändert werden, wenn eine neue Version installiert wird
