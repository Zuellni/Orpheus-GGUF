import json
from pathlib import Path

import huggingface_hub as hf
import safetensors.torch as st
import torch
import torchaudio
import torchaudio.functional as tf
import transformers
from llama_cpp import Llama
from snac import SNAC

transformers.logging.set_verbosity_error()


class Orpheus:
    def __init__(
        self,
        path: str | Path = "zuellni/orpheus-3b-0.1-pt-gguf",
        model: str = "model.q8_0.gguf",
        context: int = 8192,
        flash_attn: bool = True,
    ) -> None:
        if not (path := Path(path)).is_file():
            path = Path(hf.hf_hub_download(path.as_posix(), model))

        self.model = Llama(
            model_path=str(path),
            n_gpu_layers=-1,
            n_ctx=context,
            flash_attn=flash_attn,
            verbose=False,
        )

    def encode(
        self, text: str, add_bos: bool = False, add_eos=False, special: bool = False
    ) -> list[int]:
        tokens = self.model.tokenize(text.encode(), add_bos, special)
        return tokens + [self.model.token_eos()] if add_eos else tokens

    def decode(self, tokens: list[int], special: bool = False) -> str:
        return self.model.detokenize(tokens, special=special).decode()

    def generate(
        self,
        codes: list[torch.LongTensor],
        transcript: str,
        text: str,
        top_k: int = 40,
        top_p: float = 0.9,
        min_p: float = 0.0,
        typical_p: float = 1.0,
        temp: float = 0.5,
        repeat_penalty: float = 1.1,
    ) -> list[torch.LongTensor]:
        speech = []

        for i in range(codes[0].shape[1]):
            speech.append(codes[0][0][i].item() + 128266)
            speech.append(codes[1][0][2 * i].item() + 128266 + 4096)
            speech.append(codes[2][0][4 * i].item() + 128266 + (2 * 4096))
            speech.append(codes[2][0][(4 * i) + 1].item() + 128266 + (3 * 4096))
            speech.append(codes[1][0][(2 * i) + 1].item() + 128266 + (4 * 4096))
            speech.append(codes[2][0][(4 * i) + 2].item() + 128266 + (5 * 4096))
            speech.append(codes[2][0][(4 * i) + 3].item() + 128266 + (6 * 4096))

        start_of_speech = [128257]
        end_of_speech = [128258]
        start_of_human = [128259]
        end_of_human = [128260]
        start_of_ai = [128261]
        end_of_ai = [128262]

        transcript = self.encode(transcript, add_bos=True, add_eos=True)
        text = self.encode(text, add_bos=True, add_eos=True)

        inputs = start_of_human + transcript + end_of_human
        inputs += start_of_ai + start_of_speech + speech + end_of_speech + end_of_ai
        inputs += start_of_human + text + end_of_human
        inputs += start_of_ai + start_of_speech

        max_tokens = self.model.n_ctx() - len(inputs)
        assert max_tokens > 0, "input too long, increase context"
        outputs = []

        for token in self.model.generate(
            tokens=inputs,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            typical_p=typical_p,
            temp=temp,
            repeat_penalty=repeat_penalty,
        ):
            if token in end_of_speech or len(outputs) >= max_tokens:
                break

            outputs.append(token)

        outputs = outputs[: len(outputs) // 7 * 7]
        outputs = [o - 128266 for o in outputs]

        layer_0 = []
        layer_1 = []
        layer_2 = []

        for i in range((len(outputs) + 1) // 7):
            layer_0.append(outputs[7 * i])
            layer_1.append(outputs[7 * i + 1] - 4096)
            layer_2.append(outputs[7 * i + 2] - (2 * 4096))
            layer_2.append(outputs[7 * i + 3] - (3 * 4096))
            layer_1.append(outputs[7 * i + 4] - (4 * 4096))
            layer_2.append(outputs[7 * i + 5] - (5 * 4096))
            layer_2.append(outputs[7 * i + 6] - (6 * 4096))

        return [
            torch.LongTensor([layer_0]),
            torch.LongTensor([layer_1]),
            torch.LongTensor([layer_2]),
        ]

    def unload(self) -> None:
        if self.model._sampler:
            self.model._sampler.close()

        self.model.close()
        self.model = None
        torch.cuda.empty_cache()


class Snac:
    def __init__(
        self,
        path: str | Path = "zuellni/snac-24khz-st",
        device: str = "cuda",
        dtype: str = "float32",
    ) -> None:
        if not (path := Path(path)).is_dir():
            path = Path(hf.snapshot_download(path.as_posix()))

        self.config = json.loads(next(path.glob("*.json")).read_text(encoding="utf-8"))
        self.device = device
        self.dtype = getattr(torch, dtype)

        self.model = SNAC(**self.config)
        st.load_model(self.model, next(path.glob("*.safetensors")), device=self.device)
        self.model.to(self.device, self.dtype).eval()

    def load(self, path: str | Path, max_len: int | None = None) -> torch.Tensor:
        audio, sample_rate = torchaudio.load(path)

        if len(audio) > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        if sample_rate != self.model.sampling_rate:
            audio = tf.resample(audio, sample_rate, self.model.sampling_rate)

        return audio[:, :max_len]

    def save(self, audio: torch.Tensor, path: str | Path) -> None:
        torchaudio.save(path, audio.cpu(), self.model.sampling_rate)

    def encode(self, audio: torch.Tensor) -> list[torch.LongTensor]:
        with torch.inference_mode():
            audio = audio.to(self.device, self.dtype).unsqueeze(0)
            return self.model.encode(audio)

    def decode(self, codes: list[torch.LongTensor]) -> torch.Tensor:
        with torch.inference_mode():
            codes = [c.to(self.device) for c in codes]
            return self.model.decode(codes).float().squeeze(0)


class Whisper:
    def __init__(
        self,
        path: str | Path = "openai/whisper-large-v3-turbo",
        device: str = "cuda",
        dtype: str = "float16",
    ) -> None:
        self.model = transformers.pipeline(
            task="automatic-speech-recognition",
            model=Path(path).as_posix(),
            device=device,
            torch_dtype=getattr(torch, dtype),
        )

    def transcribe(self, audio: torch.Tensor) -> str:
        return self.model(audio.squeeze().numpy())["text"].strip()

    def unload(self) -> None:
        self.model = None
        torch.cuda.empty_cache()
