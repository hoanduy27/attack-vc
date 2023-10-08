import os 
import random 
from random import Random as Rng
import pandas as pd 
import yaml
import sys

def attack_manifest_generator(
    audio_dir: str,
    n_speakers: int,
    utt_per_spk: int, 
    adv_spk_per_spk: int, 
    disjoint_adv: bool = True, 
    replace: bool = False,
    seed: int = None,
):
    RNG = random.Random(seed)
    sample_func = RNG.choices if replace else RNG.sample

    speakers = [] 
    

    for name in os.listdir(audio_dir):
        if os.path.isdir(os.path.join(audio_dir, name)):
            speakers.append(name)

    RNG.shuffle(speakers)

    def_speakers_ = speakers[: n_speakers]

    if disjoint_adv:
        adv_speakers_pool = speakers[n_speakers:]
    else:
        adv_speakers_pool = speakers[:]

    spk2utt = {}

    def_speakers = []
    def_utterances = []
    adv_speakers = []
    adv_utterances = []


    # Get random utt
    for speaker in def_speakers_:
        # Choose random utterance
        utts = []
        for utt in os.listdir(os.path.join(audio_dir, speaker)):
            if utt.endswith('.wav'):
                utts.append(os.path.join(speaker, utt))

        RNG.shuffle(utts)
        utts = utts[:utt_per_spk]
        def_utterances.extend(utts)
        def_speakers.extend([speaker] * len(utts))
        spk2utt[speaker] = utts
    
    for speaker, utts in spk2utt.items():
        # Choose random utterance from another speaker for each def_spk
        if adv_spk_per_spk:
            # Single adversarial
            if adv_spk_per_spk == 1:
                chosen_adv_spk = RNG.choice(adv_speakers_pool)
                
                chosen_adv_utts = sample_func(
                    os.listdir(os.path.join(audio_dir, chosen_adv_spk)),
                    k = len(utts),
                )
                chosen_adv_utts = list(map(lambda x: os.path.join(chosen_adv_spk, x), chosen_adv_utts))

                chosen_adv_spks = [chosen_adv_spk] * len(utts)


            # One-to-One adversarial
            elif adv_spk_per_spk == -1:
                chosen_adv_spks = sample_func(adv_speakers_pool, k=len(utts))
                chosen_adv_utts = [] 
                for spk in chosen_adv_spks:
                    chosen_utt = RNG.choice(
                        os.listdir(os.path.join(audio_dir, spk))
                    )

                    chosen_adv_utts.append(os.path.join(spk, chosen_utt))

            adv_speakers.extend(chosen_adv_spks)
            adv_utterances.extend(chosen_adv_utts)


    # def_speakers = list(map(lambda x: [x[0]] * len(x[1]),spk2utt.items()))
    
    
    items = zip(def_speakers, def_utterances, adv_speakers, adv_utterances) if adv_spk_per_spk else zip(def_speakers, def_utterances)

    df = pd.DataFrame(items, columns=['spk', 'utt', 'adv_spk', 'adv_utt'] if adv_spk_per_spk else ['spk', 'utt'])
    
    return df


if __name__ == "__main__":
    audio_dir = sys.argv[1]
    config_path = sys.argv[2]
 
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    df = attack_manifest_generator(
        "/olli_data/personal_data/duy/VCTK-Corpus/wav48",
        **config
    )

    savename = os.path.basename(os.path.splitext(config_path)[0])

    df.to_csv(os.path.join('data', f'{savename}.csv'), index=0)
    





