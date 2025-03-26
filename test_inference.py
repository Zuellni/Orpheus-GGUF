from classes import Orpheus, Snac, Whisper
from utils import Timer

input = "../../voices/sonic.wav"
output = "output.wav"
text = (
    "She sells seashells by the seashore. "
    "The shells she sells are seashells, I'm sure. "
    "So if she sells seashells on the seashore, then I'm sure she sells seashore shells."
)

with Timer("Loaded orpheus"):
    orpheus = Orpheus("../models/orpheus-pt/model.q8_0.gguf")

with Timer("Loaded snac"):
    snac = Snac("../models/snac")

with Timer("Loaded whisper"):
    whisper = Whisper("../../models/turbo")

with Timer("Encoded audio"):
    audio = snac.load(input)
    codes = snac.encode(audio)

with Timer("Transcribed audio"):
    transcript = whisper.transcribe(audio)

with Timer("Generated audio"):
    codes = orpheus.generate(codes, transcript, text)

with Timer("Decoded audio"):
    audio = snac.decode(codes)

snac.save(audio, output)
orpheus.unload()
