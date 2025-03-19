# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import argparse
import torch
import soundfile as sf

from cli.SparkTTS import SparkTTS

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run TTS inference.")
    home_dir = os.path.join(os.path.expanduser("~"), '.str2speech', 'models', "SparkTTS")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=home_dir,
        help="Path to the model directory",
    )
    parser.add_argument(
        "--save_file",
        type=str,
        default="output.wav",
        help="Output file",
    )
    parser.add_argument(
        "--text", type=str, required=True, help="Text for TTS generation",
        default="Hello from Spark TTS"
    )
    parser.add_argument("--prompt_text", 
        type=str, help="Transcript of prompt audio",
        default="This is some nice text for training. The quick brown fox jumped over the lazy dog in a forest full of mushrooms"
    )
    parser.add_argument(
        "--prompt_speech_path",
        default=os.path.join(home_dir, "voices", "generic_female.wav"),
        type=str,
        help="Path to the prompt audio file",
    )
    parser.add_argument("--gender", choices=["male", "female"])
    parser.add_argument(
        "--pitch", choices=["very_low", "low", "moderate", "high", "very_high"]
    )
    parser.add_argument(
        "--speed", choices=["very_low", "low", "moderate", "high", "very_high"]
    )
    return parser.parse_args()


def run_tts(args):
    """Perform TTS inference and save the generated audio."""
    print(f"Using model from: {args.model_dir}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = SparkTTS(args.model_dir, device)
    with torch.no_grad():
        wav = model.inference(
            args.text,
            args.prompt_speech_path,
            prompt_text=args.prompt_text,
            gender=args.gender,
            pitch=args.pitch,
            speed=args.speed,
        )
        sf.write(args.save_file, wav, samplerate=16000)
    print(f"Audio saved.")


def main():
    args = parse_args()
    run_tts(args)

if __name__ == "__main__":
    main()
