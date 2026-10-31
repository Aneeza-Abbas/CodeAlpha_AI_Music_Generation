# AI Music Studio

An AI-powered music generation application using a trained LSTM (Long Short-Term Memory) deep learning model.

## Features

- AI-based music generation
- LSTM deep learning model
- MIDI music generation and download
- WAV audio conversion and download
- Browser-based audio playback
- Interactive Flask web interface

## Technologies Used

- Python 3.11
- TensorFlow and Keras
- NumPy
- PrettyMIDI
- SciPy
- Flask
- HTML and CSS

## Dataset

The model was trained using a sample of MIDI files from the Pop-K dataset.

## Project Structure

```text
CodeAlpha_AI_Music_Generation/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   ├── pop_music_lstm.keras
│   └── music_vocabulary.pkl
├── templates/
│   └── index.html
└── outputs/
```

## Installation

Install Python 3.11, then run:

```bash
py -3.11 -m venv .venv
```

Activate the environment in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
python app.py
```

Open your browser at:

http://127.0.0.1:5000

Click **Generate Music** to create a new musical sequence.

## Output

The application generates:

- `generated_music.mid`
- `generated_music.wav`

Both files are saved in the `outputs` folder.

## Model

The project uses an LSTM neural network trained on musical events extracted from MIDI files. Temperature sampling is used during music generation.

This is an educational AI music generation prototype.
