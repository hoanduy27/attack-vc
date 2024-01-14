import torch
import torchaudio
import time

# from watchdog.observers import Observer
# from watchdog.events import FileSystemEventHandler

from attack_vc.models.vc.adain_vc import AdainVC

# def hotload_debug(path='.'):
#     def decorator(func, *args, **kwargs):
#         class HotloadDebugEventHandler(FileSystemEventHandler):
#             def on_modified(self, event):
#                 if not event.is_directory and event.src_path.endswith('.py'):
#                     print(f'Reloading: {event.src_path}')
#                     func(*args, **kwargs)

#         event_handler = HotloadDebugEventHandler()
#         observer = Observer()
#         observer.schedule(event_handler, path=path, recursive=True)
#         observer.start()

#         try:
#             while True:
#                 time.sleep(1)
#         except KeyboardInterrupt:
#             observer.stop()
#         observer.join()

#     return decorator

# @hotload_debug()
def model_test():
    config_path = '/home/duy/github/attack-vc/model/config_adv.yaml'
    model = AdainVC.from_config(config_path)

    speech, sr = torchaudio.load("/home/duy/github/attack-vc/egs/adain-vc/exp_eps0.1/sample_per_utterance-attack_target/wav/p225/p225_008.wav")

    speech = speech.unsqueeze(0)

    speech_lengths = torch.tensor([speech.size(-1)])

    model(speech, speech_lengths, sr)



model_test()
# if __name__ == "__main__":
    

