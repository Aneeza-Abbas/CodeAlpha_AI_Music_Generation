
import os
import pickle
import random

import numpy as np
import pretty_midi
from scipy.io import wavfile

from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "pop_music_lstm.keras")
VOCAB_PATH = os.path.join(MODEL_DIR, "music_vocabulary.pkl")

print("Loading AI music model...")

model = load_model(MODEL_PATH)

with open(VOCAB_PATH, "rb") as file:
    vocabulary = pickle.load(file)

note_to_int = {note: i for i, note in enumerate(vocabulary)}
int_to_note = {i: note for i, note in enumerate(vocabulary)}

print("AI MUSIC MODEL READY!")


def sample_with_temperature(predictions, temperature=0.8):
    predictions = np.asarray(predictions).astype("float64")
    predictions = np.log(predictions + 1e-8) / temperature

    probabilities = np.exp(predictions - np.max(predictions))
    probabilities /= np.sum(probabilities)

    return np.random.choice(len(probabilities), p=probabilities)


def generate_music(length=200):

    # Start with a random sequence from the vocabulary.
    seed = random.choices(range(len(vocabulary)), k=50)
    pattern = list(seed)

    generated_notes = []

    for _ in range(length):

        input_sequence = np.array([pattern[-50:]])

        prediction = model.predict(input_sequence, verbose=0)[0]

        next_index = sample_with_temperature(prediction)

        generated_notes.append(int_to_note[next_index])

        pattern.append(next_index)

    return generated_notes


def save_music(generated_notes):

    midi = pretty_midi.PrettyMIDI()

    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(
            "Acoustic Grand Piano"
        )
    )

    start_time = 0.0
    duration = 0.35

    for event in generated_notes:

        try:
            if "." in event and all(
                part.isdigit() for part in event.split(".")
            ):
                pitch_classes = [
                    int(part) for part in event.split(".")
                ]

                midi_pitches = [
                    60 + pc for pc in pitch_classes
                    if 0 <= pc <= 11
                ]

            else:
                midi_pitches = [
                    pretty_midi.note_name_to_number(event)
                ]

            for pitch_number in midi_pitches:

                instrument.notes.append(
                    pretty_midi.Note(
                        velocity=90,
                        pitch=pitch_number,
                        start=start_time,
                        end=start_time + duration
                    )
                )

            start_time += duration

        except (ValueError, TypeError):
            continue

    midi.instruments.append(instrument)

    # Save MIDI file.
    midi_path = os.path.join(
        OUTPUT_DIR, "generated_music.mid"
    )

    midi.write(midi_path)

    # Convert MIDI into WAV audio.
    print("Converting generated music into WAV...")

    audio = midi.synthesize(fs=22050)

    audio = np.clip(audio, -1.0, 1.0)

    audio = (audio * 32767).astype(np.int16)

    wav_path = os.path.join(
        OUTPUT_DIR, "generated_music.wav"
    )

    wavfile.write(wav_path, 22050, audio)

    print("MIDI AND WAV FILES CREATED!")

    return "generated_music.mid", "generated_music.wav"


@app.route("/", methods=["GET", "POST"])
def index():

    message = None
    midi_filename = None
    wav_filename = None

    if request.method == "POST":

        try:
            print("Generating new music...")

            generated_notes = generate_music(200)

            midi_filename, wav_filename = save_music(
                generated_notes
            )

            message = "Your original AI-generated music is ready!"

        except Exception as error:

            print("Generation error:", error)

            message = f"Generation failed: {error}"

    return render_template(
        "index.html",
        message=message,
        filename=midi_filename,
        wav_filename=wav_filename
    )


@app.route("/download/<filename>")
def download(filename):

    allowed_files = [
        "generated_music.mid",
        "generated_music.wav"
    ]

    if filename not in allowed_files:
        return "File not found", 404

    return send_from_directory(
        OUTPUT_DIR,
        filename,
        as_attachment=True
    )


@app.route("/play/<filename>")
def play(filename):

    if filename != "generated_music.wav":
        return "File not found", 404

    return send_from_directory(
        OUTPUT_DIR,
        filename,
        mimetype="audio/wav"
    )


if __name__ == "__main__":
    app.run(debug=True)
