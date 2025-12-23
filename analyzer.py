try:
    # Prova prima il nuovo package
    from google import genai
    from google.genai import types
    USE_NEW_API = True
except ImportError:
    # Fallback al vecchio package
    import google.generativeai as genai
    USE_NEW_API = False

import pandas as pd
import json
import os
from typing import Dict, Any
import logging

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 1. Configurazione - Usa variabile d'ambiente per maggiore sicurezza
API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY_HERE")

if USE_NEW_API:
    client = genai.Client(api_key=API_KEY)
    logger.info("Usando il nuovo package google.genai")
else:
    genai.configure(api_key=API_KEY)
    logger.info("Usando il vecchio package google.generativeai (deprecato)")

def analizza_dati(file_csv: str, max_reviews: int = 50, column_name: str = 'body') -> Dict[str, Any]:
    """
    Analizza i dati delle recensioni usando Google Gemini AI.
    
    Args:
        file_csv: Percorso del file CSV contenente le recensioni
        max_reviews: Numero massimo di recensioni da analizzare (default: 50)
        column_name: Nome della colonna contenente il testo delle recensioni (default: 'body')
    
    Returns:
        Dizionario con l'analisi completa del sentiment e insights
    
    Raises:
        FileNotFoundError: Se il file CSV non esiste
        ValueError: Se la colonna specificata non esiste nel CSV
    """
    logger.info(f"Inizio analisi del file: {file_csv}")
    
    # Validazione file
    if not os.path.exists(file_csv):
        raise FileNotFoundError(f"File non trovato: {file_csv}")
    
    # Caricamento dati
    try:
        df = pd.read_csv(file_csv)
        logger.info(f"Caricati {len(df)} record dal file CSV")
        
        # Verifica esistenza colonna
        if column_name not in df.columns:
            raise ValueError(f"Colonna '{column_name}' non trovata. Colonne disponibili: {', '.join(df.columns)}")
        
        # Pulizia e preparazione dati
        df[column_name] = df[column_name].fillna('')
        # Estrarre il testo dalla colonna 'a-size-base 3' che contiene le recensioni complete
        if 'a-size-base 3' in df.columns:
            reviews_list = df['a-size-base 3'].astype(str).tolist()[:max_reviews]
        else:
            reviews_list = df[column_name].astype(str).tolist()[:max_reviews]
        reviews = "\n\n".join([f"Recensione {i+1}: {rev}" for i, rev in enumerate(reviews_list) if rev.strip() and rev != 'nan'])
        
        logger.info(f"Preparate {len(reviews_list)} recensioni per l'analisi")
        
    except pd.errors.EmptyDataError:
        raise ValueError("Il file CSV è vuoto")
    except Exception as e:
        logger.error(f"Errore nel caricamento file: {e}")
        raise

    # Configurazione Modello con parametri ottimizzati
    generation_config = {
        "temperature": 0.3,  # Bassa temperatura per output più consistente
        "top_p": 0.8,
        "top_k": 40,
        "max_output_tokens": 4096,  # Aumentato per evitare troncamenti
    }
    
    # Lista di modelli da provare in ordine di preferenza (modelli 2024-2025)
    model_names = [
        'models/gemini-2.5-flash',      # Più recente e veloce
        'models/gemini-2.0-flash',       # Alternativa veloce
        'models/gemini-flash-latest',    # Alias all'ultimo flash
        'models/gemini-2.5-pro',         # Pro più recente
        'models/gemini-pro-latest'       # Alias all'ultimo pro
    ]
    
    prompt = f"""
    Agisci come un esperto analista di dati, consulente strategico e specialista di sentiment analysis.
    Analizza attentamente i seguenti feedback dei clienti e restituisci un output ESCLUSIVAMENTE in formato JSON valido.
    
    Struttura JSON richiesta (rispetta esattamente questa struttura):
    {{
      "sentiment_score": 75,
      "sentiment_label": "Positivo/Neutrale/Negativo",
      "numero_recensioni_analizzate": {len(reviews_list)},
      "punti_critici": [
        {{"problema": "descrizione problema", "frequenza": "alta/media/bassa", "impatto": "alto/medio/basso"}},
        {{"problema": "descrizione problema", "frequenza": "alta/media/bassa", "impatto": "alto/medio/basso"}}
      ],
      "vantaggi_competitivi": [
        {{"vantaggio": "descrizione pregio", "menzioni": "numero stimato di menzioni"}},
        {{"vantaggio": "descrizione pregio", "menzioni": "numero stimato di menzioni"}}
      ],
      "temi_ricorrenti": ["tema 1", "tema 2", "tema 3"],
      "consiglio_ingegneristico": "descrizione tecnica dettagliata per migliorare il prodotto",
      "strategia_marketing": "strategia comunicativa basata sui dati per migliorare la percezione del cliente",
      "priorita_intervento": ["azione prioritaria 1", "azione prioritaria 2", "azione prioritaria 3"]
    }}

    FEEDBACK DA ANALIZZARE:
    {reviews}
    
    Rispondi SOLO con il JSON, senza testo aggiuntivo prima o dopo.
    """

    # Generazione con gestione errori e retry su diversi modelli
    last_error = None
    
    for model_name in model_names:
        try:
            logger.info(f"Tentativo con modello: {model_name}")
            
            if USE_NEW_API:
                # Usa la nuova API
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=generation_config['temperature'],
                        top_p=generation_config['top_p'],
                        top_k=generation_config['top_k'],
                        max_output_tokens=generation_config['max_output_tokens']
                    )
                )
                text_response = response.text.strip()
            else:
                # Usa la vecchia API
                model = genai.GenerativeModel(
                    model_name,
                    generation_config=generation_config
                )
                response = model.generate_content(prompt)
                text_response = response.text.strip()
            
            logger.info(f"✓ Risposta ricevuta da Gemini API usando {model_name}")
            
            # Pulizia dell'output (rimuove eventuali blocchi di codice Markdown)
            if text_response.startswith('```'):
                text_response = text_response.split('```')[1]
                if text_response.startswith('json'):
                    text_response = text_response[4:]
            text_response = text_response.replace('```', '').strip()
            
            # Parse JSON
            result = json.loads(text_response)
            logger.info("✓ Analisi completata con successo")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Errore nel parsing JSON con {model_name}: {e}")
            logger.error(f"Risposta ricevuta: {text_response[:500]}")
            last_error = ValueError(f"Risposta API non valida. Errore JSON: {e}")
            continue  # Prova il prossimo modello
            
        except Exception as e:
            logger.warning(f"Modello {model_name} non disponibile: {e}")
            last_error = e
            continue  # Prova il prossimo modello
    
    # Se siamo qui, tutti i modelli hanno fallito
    logger.error("Tutti i modelli hanno fallito")
    raise last_error if last_error else Exception("Impossibile generare contenuto con nessun modello disponibile")

def salva_risultati(risultato: Dict[str, Any], output_file: str = "analisi_risultati.json"):
    """Salva i risultati dell'analisi in un file JSON."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(risultato, f, indent=4, ensure_ascii=False)
        logger.info(f"Risultati salvati in: {output_file}")
    except Exception as e:
        logger.error(f"Errore nel salvataggio dei risultati: {e}")


def stampa_report(risultato: Dict[str, Any]):
    """Stampa un report formattato dei risultati."""
    print("\n" + "="*80)
    print("📊 REPORT ANALISI RECENSIONI")
    print("="*80)
    
    print(f"\n🎯 Sentiment Score: {risultato.get('sentiment_score', 'N/A')}/100")
    print(f"📈 Sentiment: {risultato.get('sentiment_label', 'N/A')}")
    print(f"📝 Recensioni analizzate: {risultato.get('numero_recensioni_analizzate', 'N/A')}")
    
    print("\n⚠️  PUNTI CRITICI:")
    for i, punto in enumerate(risultato.get('punti_critici', []), 1):
        if isinstance(punto, dict):
            print(f"   {i}. {punto.get('problema', punto)}")
            print(f"      Frequenza: {punto.get('frequenza', 'N/A')} | Impatto: {punto.get('impatto', 'N/A')}")
        else:
            print(f"   {i}. {punto}")
    
    print("\n✅ VANTAGGI COMPETITIVI:")
    for i, vantaggio in enumerate(risultato.get('vantaggi_competitivi', []), 1):
        if isinstance(vantaggio, dict):
            print(f"   {i}. {vantaggio.get('vantaggio', vantaggio)}")
        else:
            print(f"   {i}. {vantaggio}")
    
    if 'temi_ricorrenti' in risultato:
        print("\n🔍 TEMI RICORRENTI:")
        for tema in risultato['temi_ricorrenti']:
            print(f"   • {tema}")
    
    print("\n🔧 CONSIGLIO INGEGNERISTICO:")
    print(f"   {risultato.get('consiglio_ingegneristico', 'N/A')}")
    
    print("\n📢 STRATEGIA MARKETING:")
    print(f"   {risultato.get('strategia_marketing', 'N/A')}")
    
    if 'priorita_intervento' in risultato:
        print("\n🎯 PRIORITÀ DI INTERVENTO:")
        for i, priorita in enumerate(risultato['priorita_intervento'], 1):
            print(f"   {i}. {priorita}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    try:
        # Percorso del file CSV
        csv_file = "recensioni.csv"
        
        # Esegui l'analisi - la colonna con le recensioni è 'a-size-base 3'
        risultato = analizza_dati(csv_file, max_reviews=50, column_name='a-size-base 3')
        
        # Stampa report formattato
        stampa_report(risultato)
        
        # Salva anche in JSON per uso futuro
        salva_risultati(risultato)
        
        # Stampa anche JSON completo
        print("📄 Output JSON completo:")
        print(json.dumps(risultato, indent=4, ensure_ascii=False))
        
    except Exception as e:
        logger.error(f"Errore durante l'esecuzione: {e}")
        print(f"\n❌ Errore: {e}")
        exit(1)
