
# CutlerySorter Project

Detta projekt handlar om att identifiera och sortera bestick med hjälp av YOLO-modellen och en Raspberry Pi. Skripten använder Picamera2 för att fånga bilder i realtid och YOLO för att identifiera bestick i dessa bilder.

## Skriptöversikt

### cutlery_training.py
Detta skript är huvudsakligt för att träna en YOLO-modell för bestickigenkänning. Den fångar bilder med Picamera2 och använder en förtränad YOLO-modell för att känna igen bestick i dessa bilder. Det skapar ett dataset och tränar en modell som kan identifiera vanliga bestick såväl som specifika bestick från olika flygbolag (t.ex. SAS och Emirates).

#### Funktioner:
1. **Kamerainställning:** Använder Picamera2 för att fånga bilder i 1280x1280 upplösning.
2. **YOLO Modell:** Laddar en förtränad YOLO-modell (kan anpassas för olika versioner av YOLO, som yolov8n).
3. **Datahantering:** Sparar bilder och etiketter i YOLO-format, och delar upp datasetet för träning och validering.
4. **Interaktivt val:** Användaren kan välja att kategorisera bilder som SAS, Emirates eller vanliga bestick.
5. **Modellträning:** När datasetet är klart, startar träningen av YOLO-modellen för att känna igen bestick.

### run_camera.py
Det här skriptet fångar bilder i realtid och kör YOLO-modellen för att identifiera bestick, och visar resultaten på skärmen.

#### Funktioner:
1. **Real-tidsbildbehandling:** Fångar och bearbetar bilder från en ansluten kamera.
2. **YOLO Detektion:** Identifierar och markerar bestick i realtid.
3. **FPS Uppmätning:** Beräknar och visar FPS (frames per second) för detektionssystemet.

### convert.py
Skriptet används för att konvertera modeller eller data till ett annat format, vilket kan vara användbart för att optimera prestanda eller förbereda modellen för användning på andra plattformar.

### test.py
Det här skriptet används för att testa modellen eller för att göra förhandsgranskningar på tränade modeller, inklusive att testa bilder för korrekt igenkänning.

### bestick_dataset
Denna mapp innehåller alla bilder och etiketter som används för att träna modellen. Den delas upp i tränings- och valideringsuppsättningar.

## Förutsättningar

Innan du kör skripten, se till att du har installerat följande programvara och bibliotek:

- **OpenCV** – För bildbehandling och visning av kameraflöde.
- **Picamera2** – För att interagera med Raspberry Pi:s kamera.
- **YOLOv8** – För att köra YOLO-modellen för objektigenkänning.
- **ultralytics** – För att ladda och använda YOLO-modellen.

## Installation

För att installera och konfigurera miljön på din Raspberry Pi, följ dessa steg:

1. Klona detta repository eller kopiera filerna till din Raspberry Pi.
2. Skapa en virtuell miljö:
   ```bash
   python3 -m venv yolo_object
   source yolo_object/bin/activate
   ```
3. Installera nödvändiga bibliotek:
   ```bash
   pip install opencv-python
   pip install picamera2
   pip install ultralytics
   ```
4. Säkerställ att du har den förtränade YOLO-modellen tillgänglig på rätt sökväg.

## Användning

För att köra träningen och bestickigenkänning, följ dessa steg:

1. **Navigera till mappen där skripten är lagrade.**
2. **Aktivera den virtuella miljön:**
   ```bash
   source yolo_object/bin/activate
   ```
3. **Kör `cutlery_training.py` för att träna modellen:**
   ```bash
   python3 cutlery_training.py
   ```

   Detta skript kommer att fånga bilder från kameran, identifiera bestick och spara bilder och etiketter för träning av modellen.

4. **För att köra realtidsdetektion, kör `run_camera.py`:**
   ```bash
   python3 run_camera.py
   ```

   Detta skript fångar bilder i realtid, kör YOLO-detektion på dessa bilder och visar resultaten på skärmen.

## Datasetstruktur

Skriptet kommer att skapa en mappstruktur under `bestick_dataset`:

```
bestick_dataset/
    images/
        train/
        val/
    labels/
        train/
        val/
    bestick_dataset.yaml
```

- `train/` och `val/` innehåller bilder och etiketter för träning och validering.
- `bestick_dataset.yaml` beskriver datasetets struktur och definierar klasserna för bestick.

## Modeller

- **YOLOv8** – Förtränade modeller kan laddas eller tränas vidare beroende på behov.
- **Modeller som används i ncnn-format** kan användas för snabbare inferens.

## Avslutning

Detta projekt gör det möjligt att automatiskt identifiera och sortera bestick med hjälp av datorseende och maskininlärning. Genom att träna YOLO-modellen på ett specifikt dataset kan bestick identifieras och kategoriseras efter flygbolag eller vanliga typer.
