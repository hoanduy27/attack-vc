import os 
import random 
from random import Random as Rng
import pandas as pd 
import yaml
import sys

def inference_manifest_generator(
    audio_dir: str,
    attack_manifest: str,
    replace: bool = False,
    seed: int = None,
):
    RNG = random.Random(seed)
    sample_func = RNG.choices if replace else RNG.sample
    attack_df = pd.read_csv(attack_manifest)

    speakers = [] 
    

    for name in os.listdir(audio_dir):
        if os.path.isdir(os.path.join(audio_dir, name)):
            speakers.append(name)

    def_speakers_ = list(attack_df.spk.unique())
    content_speakers = [
        spk for spk in speakers if spk not in def_speakers_
    ]
    RNG.shuffle(content_speakers)

    def sample_utterance(df):
        utts = []
        chosen_content_spks = sample_func(content_speakers, k=len(df))
        for spk in chosen_content_spks:
            while True:
                chosen_utt = RNG.choice(
                    os.listdir(os.path.join(audio_dir, spk))
                )

                if chosen_utt not in utts:
                    utts.append(os.path.join(spk, chosen_utt))
                    break

        df['content_spk'] = chosen_content_spks
        df['content_utt'] = utts

        return df

    df = attack_df.groupby('spk', group_keys=True).apply(sample_utterance)

    return df


if __name__ == "__main__":
    audio_dir = sys.argv[1]
    attack_manifest = sys.argv[2]
    config_path = sys.argv[3]
 
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    print(attack_manifest)

    print(config_path)

    df = inference_manifest_generator(
        audio_dir,
        attack_manifest,
        **config
    )

    savename = os.path.basename(os.path.splitext(config_path)[0])

    df.to_csv(os.path.join('data', f'{savename}.csv'), index=0)
    
