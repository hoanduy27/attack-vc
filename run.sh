audio_dir=/olli_data/personal_data/duy/VCTK-Corpus/wav48/
egs="egs/adain-vc"

# python embed_origin.py ${audio_dir} ${egs} sample_single_target
python attacker.py ${audio_dir} ${egs} sample_per_speaker attack_target
python attacker.py ${audio_dir} ${egs} sample_per_utterance attack_target
python attacker.py ${audio_dir} ${egs} sample_untarget attack_untarget