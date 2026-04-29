# -*- coding: utf-8 -*-
"""
Created on Thu May  8 16:20:08 2025

@author: Timm
"""

import requests
import time
import csv
from requests.exceptions import ChunkedEncodingError

# ===  API-Zugang
ACCESS_TOKEN = "A3v6phESDjmSWpZuF8yXYYdqvVMZgFr0FH5Gf8R3fGwzQzvbUCjf"
HEADERS = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Content-Type": "application/json"
}
MAX_RESULTS_PER_CALL = 100
TOTAL_RESULTS = 10000
CSV_DATEINAME = "vr_ar_patente.csv"

# === Schlagwörter für Titel/Abstract

KEYWORDS = [
    
    # ▶ Allgemeine Begriffe
    "virtual reality", "augmented reality", "mixed reality", "XR controller",

    # ▶ Headsets, Displays & visuelle Darstellung
    "head-mounted display", "wearable display", "stereoscopic display",
    "field of view", "wireless headset", "foveated rendering",
    "focus tunable lenses", "eye tracking", "pass-through display",
    "vergence-accommodation conflict", "dynamic light field", "adaptive optics",
    "pancake lenses", "microdisplay", "OLED display", "LCD display",
    "optical waveguide", "see-through display", "retinal projection display",
    "holocake lenses", "varifocal lenses",

    # ▶ Eingabegeräte & Interfaces
    "motion controller", "gesture recognition", "hand tracking",
    "controller input", "force feedback", "vibration motor",
    "haptic feedback", "wearable input device", "joystick tracking",
    "button input", "touchpad interface", "voice command recognition",
    "eye-gaze control", "brain-computer interface", "EMG sensor input",
    "neural interface band", "EMG neural interface",

    # ▶ Sensorik & Tracking
    "motion capture", "tracking system", "positional tracking",
    "low-latency communication", "spatial audio", "inertial measurement unit",
    "6DoF tracking", "3DoF tracking", "environmental sensor integration",
    "proximity sensor calibration", "IMU integration", "infrared tracking",
    "inside-out tracking", "outside-in tracking", "depth sensing camera",
    "LIDAR sensor", "SLAM algorithm", "marker-based tracking",
    "EMG sensor", "neural interface",

    # ▶ Systemintegration & Rechenleistung
    "edge computing", "cloud rendering", "real-time data processing",
    "network latency optimization", "wireless data transmission",
    "GPU acceleration", "CPU-GPU synchronization", "thermal management systems",
    "on-device AI processing", "hardware-software co-design", "modular hardware architecture",

    # ▶ Komfort & Benutzererfahrung
    "digital interpupillary distance adjustment", "immersive experience",
    "ergonomic design", "weight distribution optimization",
    "ventilation system", "adjustable head straps", "user comfort metrics",
    "anti-motion sickness technology", "eye strain reduction", "adaptive brightness control",
    "Elite Straps", "comfort-focused design",

    # ▶ Sonstiges / spezielle Hardwaredesigns
    "virtual hardware device", "modular hardware components",
    "customizable hardware interfaces", "hardware scalability",
    "hardware compatibility standards", "virtual object anchoring",
    "real-world object augmentation", "context-aware gaming hardware",
    "environment mapping technology", "virtual cockpit environment",
    "virtual store environment", "augmented indicia access rights",
    "virtual graphical user interface", "virtual hardware selection interface"
]

def clean(text):
    if isinstance(text, str):
        return text.replace("\n", " ").replace("\r", " ").strip()
    return ""

alle_patente = []

for start in range(0, TOTAL_RESULTS, MAX_RESULTS_PER_CALL):
    query = {
        "query": {
            "bool": {
                "must": [
                    {"match": {"legal_status.granted": True}},
                    {"terms": {"jurisdiction": ["US", "EP"]}},
                    {"range": {
                        "date_published": {
                            "gte": "2000-01-01", 
                            "lte": "2025-12-31"
                        }
                    }},
                    {
                        "bool": {
                            "should": [
                                {"match": {"title": kw}} for kw in KEYWORDS
                            ] + [
                                {"match": {"abstract": kw}} for kw in KEYWORDS
                            ],
                            "minimum_should_match": 1
                        }
                    }
                ]
            }
        },
        "size": MAX_RESULTS_PER_CALL,
        "from": start,
        "include": [
            "lens_id",
            "biblio.invention_title",
            "abstract",
            "description",
            "claims",
            "date_published",
            "biblio.classifications_cpc"
        ]
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post("https://api.lens.org/patent/search", headers=HEADERS, json=query)
            response.raise_for_status()
            break
        except ChunkedEncodingError:
            print(f"API-Verbindungsfehler bei Start {start}, Versuch {attempt + 1}/{max_retries}")
            time.sleep(5)
        except Exception as e:
            print(f"Schwerwiegender Fehler bei Start {start}: {e}")
            break
    else:
        continue

    daten = response.json().get("data", [])
    print(f"{len(daten)} Treffer bei Start {start}")

    for eintrag in daten:
        try:
            lens_id = eintrag.get("lens_id", "")

            # Titel
            title = ""
            title_data = eintrag.get("biblio", {}).get("invention_title", "")
            if isinstance(title_data, list):
                for t in title_data:
                    if isinstance(t, dict) and t.get("lang", "").lower() == "en":
                        title = t.get("text", "")
                        break
            elif isinstance(title_data, str):
                title = title_data

            # Abstract
            abstract = ""
            abstract_data = eintrag.get("abstract", "")
            if isinstance(abstract_data, list):
                for ab in abstract_data:
                    if isinstance(ab, dict) and ab.get("lang", "").lower() == "en":
                        abstract = ab.get("text", "")
                        break
            elif isinstance(abstract_data, str):
                abstract = abstract_data

            # Beschreibung
            description = ""
            desc_data = eintrag.get("description", "")
            if isinstance(desc_data, dict):
                if desc_data.get("lang", "").lower() == "en":
                    description = desc_data.get("text", "")
            elif isinstance(desc_data, str):
                description = desc_data

            # Claims
            claims = []
            claims_list = eintrag.get("claims", [])
            if isinstance(claims_list, list):
                for claim_block in claims_list:
                    claim_texts = claim_block.get("claims", [])
                    if isinstance(claim_texts, list):
                        for c in claim_texts:
                            if isinstance(c, dict):
                                ct = c.get("claim_text", [])
                                if isinstance(ct, list):
                                    claims.extend(ct)
                                elif isinstance(ct, str):
                                    claims.append(ct)
            claims_combined = " ".join(claims)

            # Datum
            publication_date = eintrag.get("date_published", "")

            # CPC-Codes
            cpc_codes = []
            cpc_raw = eintrag.get("biblio", {}).get("classifications_cpc", {})
            if isinstance(cpc_raw, dict):
                for cpc in cpc_raw.get("classifications", []):
                    if isinstance(cpc, dict):
                        symbol = cpc.get("symbol")
                        if symbol:
                            cpc_codes.append(symbol)

            alle_patente.append({
                "lens_id": clean(lens_id),
                "title": clean(title),
                "abstract": clean(abstract),
                "claims": clean(claims_combined),
                "description": clean(description),
                "publication_date": clean(publication_date),
                "cpc_codes": clean(", ".join(sorted(set(cpc_codes))))
            })

        except Exception as e:
            print(f" Fehler bei Patent {eintrag.get('lens_id')}: {e}")

    print(f" {len(daten)} Patente von Start {start} verarbeitet.")
    time.sleep(60)

# === CSV schreiben
if alle_patente:
    feldnamen = [
        "lens_id", "title", "abstract", "claims",
        "description", "publication_date", "cpc_codes"
    ]

    with open(CSV_DATEINAME, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=feldnamen, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for patent in alle_patente:
            writer.writerow(patent)

    print(f"Export abgeschlossen: {len(alle_patente)} Patente in '{CSV_DATEINAME}' gespeichert.")
else:
    print("Keine Patente exportiert – Datenliste leer.")
