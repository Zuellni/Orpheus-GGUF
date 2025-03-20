import json
from pathlib import Path

import huggingface_hub as hf
import numpy as np
import safetensors.torch as st
import torch
import torchaudio
import torchaudio.functional as tf
import transformers
from llama_cpp import Llama
from snac import SNAC


class Snake:
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
        torchaudio.save(audio, path, self.model.sampling_rate)

    @torch.inference_mode
    def encode(self, audio: torch.FloatTensor | Path | str) -> list[torch.LongTensor]:
        if isinstance(audio, Path) or isinstance(audio, str):
            audio = self.load(audio)

        audio = audio.to(self.device, self.dtype).unsqueeze(0)
        return self.model.encode(audio)

    @torch.inference_mode
    def decode(self, codes: list[torch.LongTensor]) -> torch.FloatTensor:
        audio = self.model.decode(codes)
        return audio


class Orpheus:
    def __init__(
        self,
        path: Path | str = "annuvin/orpheus-3b-0.1-ft-gguf",
        file: str = "model.q8_0.gguf",
        context: int = 4096,
        flash_attn: bool = False,
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

    # todo: make sense of this
    def encode(self, text: str, bos: bool = False, special: bool = False) -> list[int]:
        # all_codes = []

        # for i in range(codes[0].shape[1]):
        #     all_codes.append(codes[0][0][i].item() + 128266)
        #     all_codes.append(codes[1][0][2 * i].item() + 128266 + 4096)
        #     all_codes.append(codes[2][0][4 * i].item() + 128266 + (2 * 4096))
        #     all_codes.append(codes[2][0][(4 * i) + 1].item() + 128266 + (3 * 4096))
        #     all_codes.append(codes[1][0][(2 * i) + 1].item() + 128266 + (4 * 4096))
        #     all_codes.append(codes[2][0][(4 * i) + 2].item() + 128266 + (5 * 4096))
        #     all_codes.append(codes[2][0][(4 * i) + 3].item() + 128266 + (6 * 4096))

        # start_tokens = torch.tensor([[ 128259]], dtype=torch.int64)
        # end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
        # final_tokens = torch.tensor([[128258, 128262]], dtype=torch.int64)
        # voice_prompt = and_the_transcript_is
        # prompt_tokked = tokenizer(voice_prompt, return_tensors="pt")

        # input_ids = prompt_tokked["input_ids"]

        # zeroprompt_input_ids = torch.cat([start_tokens, input_ids, end_tokens, torch.tensor([myts]), final_tokens], dim=1) # SOH SOT Text EOT EOH

        # prompts = the_model_should_say

        # all_modified_input_ids = []

        # for prompt in prompts:
        # input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        # second_input_ids = torch.cat([zeroprompt_input_ids, start_tokens, input_ids, end_tokens], dim=1)
        # all_modified_input_ids.append(second_input_ids)

        # all_padded_tensors = []
        # all_attention_masks = []

        # max_length = max([modified_input_ids.shape[1] for modified_input_ids in all_modified_input_ids])

        # for modified_input_ids in all_modified_input_ids:
        # padding = max_length - modified_input_ids.shape[1]
        # padded_tensor = torch.cat([torch.full((1, padding), 128263, dtype=torch.int64), modified_input_ids], dim=1)
        # attention_mask = torch.cat([torch.zeros((1, padding), dtype=torch.int64), torch.ones((1, modified_input_ids.shape[1]), dtype=torch.int64)], dim=1)
        # all_padded_tensors.append(padded_tensor)
        # all_attention_masks.append(attention_mask)

        # all_padded_tensors = torch.cat(all_padded_tensors, dim=0)
        # all_attention_masks = torch.cat(all_attention_masks, dim=0)

        # input_ids = all_padded_tensors.to("cuda")
        # attention_mask = all_attention_masks.to("cuda")
        return self.model.tokenize(text.encode(), bos, special)

    # and make sense of this too
    def decode(self, tokens: list[int], special: bool = False) -> str:
        # token_to_find = 128257
        # token_to_remove = 128258

        # # Check if the token exists in the tensor
        # token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)

        # if len(token_indices[1]) > 0:
        #     last_occurrence_idx = token_indices[1][-1].item()
        #     cropped_tensor = generated_ids[:, last_occurrence_idx+1:]
        # else:
        #     cropped_tensor = generated_ids

        # mask = cropped_tensor != token_to_remove
        # processed_rows = []
        # for row in cropped_tensor:
        #     masked_row = row[row != token_to_remove]
        #     processed_rows.append(masked_row)

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
        return self.model.detokenize(tokens, special=special).decode()

    # and fix this
    def generate(
        self,
        text: str,
        codes: str,
        transcript: str,
        top_k: int = 50,
        top_p: float = 0.9,
        min_p: float = 0.0,
        typical_p: float = 1.0,
        temp: float = 0.6,
        repeat_penalty: float = 1.0,
    ) -> str:
        # prompt = f"{codes}{transcript}{text}" # not like this
        inputs = self.encode(prompt, special=True)
        max_tokens = self.model.n_ctx() - len(inputs)
        outputs = []

        # generated_ids = model.generate(
        #     input_ids=input_ids,
        #     # attention_mask=attention_mask,
        #     max_new_tokens=990,
        #     do_sample=True,
        #     temperature=0.5,
        #     # top_k=40,
        #     top_p=0.9,
        #     repetition_penalty=1.1,
        #     num_return_sequences=1,
        #     eos_token_id=128258,
        #     # end_token_id=128009
        # )

        for token in self.model.generate(
            tokens=inputs,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            typical_p=typical_p,
            temp=temp,
            repeat_penalty=repeat_penalty,
        ):
            if token == self.model.token_eos() or len(outputs) >= max_tokens:
                break

            outputs.append(token)

        return self.decode(outputs, special=True)

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
        )

    def transcribe(self, audio: torch.FloatTensor | np.ndarray) -> str:
        if isinstance(audio, torch.FloatTensor):
            audio = audio.numpy()

        return self.model(audio)["text"].strip()


if __name__ == "__main__":
    speaker = "D:/AI/TTS/Voices/Alice.wav"
    text = "The quick brown fox jumped over the lazy dog."

    snake = Snake("D:/AI/TTS/Orpheus/Models/Snake")
    # orpheus = Orpheus("D:/AI/TTS/Orpheus/Models/Orpheus/model.q8_0.gguf")
    # whisper = Whisper("D:/AI/TTS/Models/Turbo")

    audio = snake.load(speaker)
    codes = snake.encode(audio)
    print(audio.device, audio.dtype, audio.shape)
    print(codes[0].device, codes[0].dtype, codes[0].shape)
    print(codes[1].device, codes[1].dtype, codes[1].shape)
    print(codes[2].device, codes[2].dtype, codes[2].shape)
    # transcript = whisper.transcribe(audio)
    # codes = orpheus.generate(text, codes, transcript)
    audio = snake.decode(codes)
    # print(audio.device, audio.dtype, audio.shape)
    # snac.save(audio, "output.wav")
    # orpheus.unload()
