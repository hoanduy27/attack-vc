content_dir=/olli_data/personal_data/duy/VCTK-Corpus/wav48/
egs=egs/adain-vc/
inference_data=inference

python attack_vc/inferencer.py egs/adain-vc/exp/sample_per_speaker-origin ${content_dir} ${egs} ${inference_data}
python attack_vc/inferencer.py egs/adain-vc/exp/sample_per_speaker-attack_target ${content_dir} ${egs} ${inference_data}
python attack_vc/inferencer.py egs/adain-vc/exp/sample_per_utterance-attack_target ${content_dir} ${egs} ${inference_data}
python attack_vc/inferencer.py egs/adain-vc/exp/sample_untarget-attack_untarget ${content_dir} ${egs} ${inference_data}