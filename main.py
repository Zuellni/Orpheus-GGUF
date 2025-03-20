import json
from pathlib import Path
from warnings import simplefilter

simplefilter("ignore")

import huggingface_hub as hf
import safetensors.torch as st
import torch
import torchaudio
import torchaudio.functional as tf
import transformers
from llama_cpp import Llama
from snac import SNAC

transformers.logging.set_verbosity_error()


class Codec:
    def __init__(
        self,
        path: Path | str = "annuvin/snac_24khz-st",
        device: str = "cuda",
        dtype: str = "float16",
    ) -> None:
        if not (path := Path(path)).is_dir():
            path = Path(hf.snapshot_download(path.as_posix()))

        self.config = json.loads(next(path.glob("*.json")).read_text(encoding="utf-8"))
        self.device = device
        self.dtype = getattr(torch, dtype)

        self.model = SNAC(**self.config)
        st.load_model(self.model, next(path.glob("*.safetensors")), device=self.device)
        self.model.to(self.device, self.dtype).eval()

    def load(self, path: Path | str, max_len: int | None = None) -> torch.FloatTensor:
        audio, sample_rate = torchaudio.load(path)

        if len(audio) > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        if sample_rate != self.model.sampling_rate:
            audio = tf.resample(audio, sample_rate, self.model.sampling_rate)

        return audio[:, :max_len]

    def save(self, audio: torch.FloatTensor, path: Path | str) -> None:
        torchaudio.save(audio.float().cpu(), path, self.model.sampling_rate)

    def encode(self, audio: torch.FloatTensor | Path | str) -> list[torch.LongTensor]:
        if isinstance(audio, Path) or isinstance(audio, str):
            audio = self.load(audio)

        with torch.inference_mode():
            audio = audio.to(self.device, self.dtype).unsqueeze(0)
            return self.model.encode(audio)

    def decode(self, codes: list[torch.LongTensor]) -> torch.FloatTensor:
        with torch.inference_mode():
            return self.model.decode(codes)


class Model:
    def __init__(
        self,
        path: Path | str = "annuvin/orpheus-3b-0.1-ft-gguf",
        file: str = "model.q8_0.gguf",
        context: int = 4096,
        flash_attn: bool = True,
    ) -> None:
        if not (path := Path(path)).is_file():
            path = Path(hf.hf_hub_download(path.as_posix(), file))

        self.model = Llama(
            model_path=str(path),
            n_gpu_layers=-1,
            n_ctx=context,
            n_batch=context,
            n_ubatch=context,
            flash_attn=flash_attn,
            verbose=False,
        )

    def encode(self, text: str, bos: bool = False, special: bool = False) -> list[int]:
        return self.model.tokenize(text.encode(), bos, special)

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
        temp: float = 0.6,
        repeat_penalty: float = 1.1,
    ) -> str:
        reference = []

        for i in range(codes[0].shape[1]):
            reference.append(codes[0][0][i].item() + 128266)
            reference.append(codes[1][0][2 * i].item() + 128266 + 4096)
            reference.append(codes[2][0][4 * i].item() + 128266 + (2 * 4096))
            reference.append(codes[2][0][(4 * i) + 1].item() + 128266 + (3 * 4096))
            reference.append(codes[1][0][(2 * i) + 1].item() + 128266 + (4 * 4096))
            reference.append(codes[2][0][(4 * i) + 2].item() + 128266 + (5 * 4096))
            reference.append(codes[2][0][(4 * i) + 3].item() + 128266 + (6 * 4096))

        # start_ids = [128259]
        # start_tokens = "<custom_token_3>"

        # end_ids = [128009, 128260, 128261, 128257]
        # end_tokens = "<|eot_id|><custom_token_4><custom_token_5><custom_token_1>"

        # final_ids = [128258, 128262]
        # final_tokens = "<custom_token_2><custom_token_6>"

        # start = f"<custom_token_3>{transcript}<|eot_id|><custom_token_4><custom_token_5>"
        # end = f"{codes}<custom_token_2><custom_token_6>"
        # final = f"<custom_token_3>{text}<|eot_id|><custom_token_4><custom_token_5><custom_token_1>

        inputs = [128259] + self.encode(transcript) + [128009, 128260, 128261, 128257]
        inputs += reference + [128258, 128262]
        inputs += [128259] + self.encode(text) + [128009, 128260, 128261, 128257]

        max_tokens = self.model.n_ctx() - len(inputs)
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
            # <|eot_id|> = 128009, 128258 = <custom_token_2>
            # self.model.token_eos()
            if token in [128009, 128258] or len(outputs) >= max_tokens:
                break

            outputs.append(token)

        print(self.decode(outputs))
        return outputs

        # code_lists = []
        # for row in processed_rows:
        #     row_length = row.size(0)
        #     new_length = (row_length // 7) * 7
        #     trimmed_row = row[:new_length]
        #     trimmed_row = [t - 128266 for t in trimmed_row]
        #     code_lists.append(trimmed_row)

        # def redistribute_codes(code_list):
        # layer_1 = []
        # layer_2 = []
        # layer_3 = []
        # for i in range((len(code_list)+1)//7):
        #     layer_1.append(code_list[7*i])
        #     layer_2.append(code_list[7*i+1]-4096)
        #     layer_3.append(code_list[7*i+2]-(2*4096))
        #     layer_3.append(code_list[7*i+3]-(3*4096))
        #     layer_2.append(code_list[7*i+4]-(4*4096))
        #     layer_3.append(code_list[7*i+5]-(5*4096))
        #     layer_3.append(code_list[7*i+6]-(6*4096))
        # codes = [torch.tensor(layer_1).unsqueeze(0),
        #         torch.tensor(layer_2).unsqueeze(0),
        #         torch.tensor(layer_3).unsqueeze(0)]
        # audio_hat = snac_model.decode(codes)

        # my_samples = []
        # for code_list in code_lists:
        # samples = redistribute_codes(code_list)
        # my_samples.append(samples)

    def unload(self):
        if self.model._sampler:
            self.model._sampler.close()

        self.model.close()


class Whisper:
    def __init__(
        self,
        model: str = "openai/whisper-large-v3-turbo",
        device: str = "cuda",
        dtype: str = "float16",
    ) -> None:
        self.model = transformers.pipeline(
            task="automatic-speech-recognition",
            model=model,
            device=device,
            torch_dtype=getattr(torch, dtype),
            model_kwargs={"attn_implementation": "flash_attention_2"},
        )

    def transcribe(self, audio: torch.FloatTensor) -> str:
        return self.model(audio.squeeze().numpy())["text"]


if __name__ == "__main__":
    speaker = "D:/AI/TTS/Voices/Alice.wav"
    text = "The quick brown fox jumped over the lazy dog."

    codec = Codec("D:/AI/TTS/Orpheus/Models/Snake")
    model = Model("D:/AI/TTS/Orpheus/Models/Orpheus/model.q8_0.gguf")
    whisper = Whisper("D:/AI/TTS/Models/Turbo")

    audio = codec.load(speaker)
    codes = codec.encode(audio)

    print(audio.device, audio.dtype, audio.shape)
    print(codes[0].device, codes[0].dtype, codes[0].shape)
    print(codes[1].device, codes[1].dtype, codes[1].shape)
    print(codes[2].device, codes[2].dtype, codes[2].shape)

    transcript = whisper.transcribe(audio)
    print(transcript)

    codes = model.generate(codes, transcript, text)
    print(codes)

    # audio = codec.decode(codes)
    # print(audio.device, audio.dtype, audio.shape)

    # codec.save(audio, "output.wav")
    model.unload()
