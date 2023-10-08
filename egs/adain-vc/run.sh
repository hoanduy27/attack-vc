audio_dir=/olli_data/personal_data/duy/VCTK-Corpus/wav48
python attack_manifest_generator.py ${audio_dir} config/sample_per_speaker.yaml
python attack_manifest_generator.py ${audio_dir} config/sample_per_utterance.yaml
python attack_manifest_generator.py ${audio_dir} config/sample_untarget.yaml