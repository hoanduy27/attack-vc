import torch 
from torch import nn
from tqdm import tqdm

from attack_vc import TrainableAttacker

MAXLEN = 288000
MAXLEN_TIME = 20000

def linf_clamp(tensor, eps):
    """Clamp tensor to eps in Linf norm"""
    if isinstance(eps, torch.Tensor) and eps.dim() == 1:
        eps = eps.unsqueeze(1)
    return torch.clamp(tensor, min=-eps, max=eps)


class UniversalAttacker(TrainableAttacker):
    def __init__(
        self,
        model, 
        eps=0.3,
        eps_item=0.1,
        nb_epochs=10,
        nb_iter=40,
        rand_init=True,
        clip_min=None,
        clip_max=None,
        order=np.inf,
        targeted=False,
        train_mode_for_backward=True,
        lr=0.001,
        time_universal=False,
        univ_perturb=None
    ):
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.eps = eps
        self.eps_item = eps_item
        self.lr = lr
        self.nb_iter = nb_iter
        self.nb_epochs = nb_epochs
        self.rand_init = rand_init
        self.order = order
        self.targeted = targeted
        self.model = model
        self.train_mode_for_backward = train_mode_for_backward
        self.time_universal = time_universal

        self.univ_perturb = univ_perturb
        if self.univ_perturb is None:
            len_delta = MAXLEN_TIME if time_universal else MAXLEN

            self.univ_perturb = nn.parameter.Parameter(torch.zeros(size=len_delta))

        assert isinstance(self.eps, torch.Tensor) or isinstance(
            self.eps, float)

    def fit(self, loader):

        if self.train_mode_for_backward:
            self.model.module_train()
        else:
            self.model.module_eval()

        delta = self.univ_perturb.tensor.data
        success_rate = 0

        best_success_rate = -100
        epoch = 0

        #####HYPERPARAM for fixed delta#####
        use_time_universal = self.time_universal
        ####################################

        while epoch < self.nb_epochs:
            print(f'{epoch}s epoch')
            epoch += 1
            # GENERATE CANDIDATE FOR UNIVERSAL PERTURBATION
            for idx, batch in enumerate(tqdm(loader, dynamic_ncols=True)):
                batch = batch.to(self.asr_brain.device)
                wav_init, wav_lens = batch.sig

                if use_time_universal:
                    base_delta = delta[:MAXLEN_TIME]
                    delta_x = base_delta.repeat(torch.ceil(
                        wav_init.shape[1]/base_delta.shape[0]))
                    delta_x = delta_x[:wav_init.shape[1]]
                else:
                    # Slice or Pad to match the shape with data point x
                    delta_x = torch.zeros_like(wav_init[0])
                    if wav_init.shape[1] <= delta.shape[0]:
                        delta_x[:wav_init.shape[1]] = delta[: wav_init.shape[1]].detach()
                    else:
                        delta_x[: delta.shape[0]] = delta.detach()

                delta_batch = delta_x.unsqueeze(0).expand(wav_init.size())
                
                _, _, predicted_tokens_origin = self.asr_brain.compute_forward(
                    batch, rs.Stage.ADVTRUTH)
                predicted_words_origin = [
                    decode(utt_seq).split(" ")
                    for utt_seq in predicted_tokens_origin
                ]

                if use_time_universal:
                    r = torch.rand_like(base_delta) / 1e+4
                else:
                    r = torch.rand_like(delta_x) / 1e+4
                r.requires_grad_()

                batch.sig = wav_init + delta_batch, wav_lens
                _, _, predicted_tokens_adv = self.asr_brain.compute_forward(
                    batch, rs.Stage.ADVTARGET)
                predicted_words_adv = [
                    decode(utt_seq).split(" ")
                    for utt_seq in predicted_tokens_adv
                ]

                # self.asr_brain.cer_metric.append(batch.id, predicted_words_adv, predicted_words_origin)
                def cer_metric(id, ref, hyp):
                    computer = self.asr_brain.hparams.cer_computer()
                    computer.append(id, ref, hyp)
                    return computer.summarize("error_rate")
                CER = 0
                # print(CER)

                for i in range(self.nb_iter):
                    if use_time_universal:
                        r_batch = r.repeat(torch.ceil(wav_init.shape[1]/r.shape[0]))[
                            :delta_x.size()].unsqueeze(0).expand(delta_batch.size())
                    else:
                        r_batch = r.unsqueeze(0).expand(delta_batch.size())

                    batch.sig = wav_init + delta_batch + r_batch, wav_lens
                    predictions = self.asr_brain.compute_forward(
                        batch, rs.Stage.ATTACK)
                    # loss = 0.5 * r.norm(dim=1, p=2) - self.asr_brain.compute_objectives(predictions, batch, rs.Stage.ATTACK)
                    ctc = - \
                        self.asr_brain.compute_objectives(
                            predictions, batch, rs.Stage.ATTACK)
                    l2_norm = r.norm(p=2).to(
                        self.asr_brain.device)
                    loss = 0.5 * l2_norm + ctc
                    # loss = ctc
                    loss.backward()
                    # print(l2_norm,ctc,CER)
                    grad_sign = r.grad.data.sign()
                    r.data = r.data - self.lr * grad_sign
                    # r.data = r.data - 0.1 * r.grad.data
                    r.data = linf_clamp(r.data, self.eps_item)
                    r.data = linf_clamp(
                        delta_x + r.data, self.eps) - delta_x

                    # print("delta's mean : ", torch.mean(delta_x).data)
                    # print("r's mean : ",torch.mean(r).data)
                    r.grad.data.zero_()

                    _, _, predicted_tokens_adv = self.asr_brain.compute_forward(
                        batch, rs.Stage.ADVTARGET)
                    predicted_words_adv = [
                        decode(utt_seq).split(" ")
                        for utt_seq in predicted_tokens_adv
                    ]

                    CER = cer_metric(batch.id, predicted_words_origin,
                                     predicted_words_adv)
                    # print(CER)
                    if CER >= CER_SUCCESS_THRESHOLD:
                        break

                # print(f'CER = {CER}')
                delta_x = linf_clamp(delta_x + r.data, self.eps)

                if delta.shape[0] <= delta_x.shape[0]:
                    delta = delta_x[:delta.shape[0]].detach()
                else:
                    delta[:delta_x.shape[0]] = delta_x.detach()

            # print(f'MAX OF INPUT WAVE IS {torch.max(wav_init).data}')
            # print(f'AVG OF INPUT WAVE IS {torch.mean(wav_init).data}')
            # print(f'MAX OF DELTA IS {torch.max(delta).data}')
            # print(f'AVG OF DELTA IS {torch.mean(delta).data}')
            print('CHECK SUCCESS RATE OVER ALL TRAINING SAMPLES')
            # TO CHECK SUCCESS RATE OVER ALL TRAINING SAMPLES
            total_sample = 0.
            fooled_sample = 0.

            cer_computer = self.asr_brain.hparams.cer_computer()

            for idx, batch in enumerate(tqdm(loader, dynamic_ncols=True)):
                batch = batch.to(self.asr_brain.device)
                wav_init, wav_lens = batch.sig

                if use_time_universal:
                    base_delta = delta[:MAXLEN_TIME]
                    delta_x = base_delta.repeat(torch.ceil(
                        wav_init.shape[1]/base_delta.shape[0]))
                    delta_x = delta_x[:wav_init.shape[1]]
                else:
                    delta_x = torch.zeros_like(wav_init[0])
                    if wav_init.shape[1] <= delta.shape[0]:
                        delta_x = delta[:wav_init.shape[1]]
                    else:
                        delta_x[:delta.shape[0]] = delta
                # if idx == 400:
                #     break
                #     raise NotImplementedError

                # CER(Xi)
                delta_batch = delta_x.unsqueeze(0).expand(wav_init.size())
                _, _, predicted_tokens_origin = self.asr_brain.compute_forward(
                    batch, rs.Stage.ADVTRUTH)

                predicted_words_origin = [
                    decode(utt_seq).split(" ")
                    for utt_seq in predicted_tokens_origin
                ]

                # CER(Xi + v)
                batch.sig = wav_init + delta_batch, wav_lens
                _, _, predicted_tokens_adv = self.asr_brain.compute_forward(
                    batch, rs.Stage.ADVTRUTH)
                predicted_words_adv = [
                    decode(utt_seq).split(" ")
                    for utt_seq in predicted_tokens_adv
                ]
                cer_computer.append(
                    batch.id, predicted_words_origin, predicted_words_adv)

                total_sample += 1.
            success_rate = cer_computer.summarize("error_rate")
            print(f'SUCCESS RATE (CER) IS {success_rate:.4f}')
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                if use_time_universal:
                    self.univ_perturb.tensor.data = base_delta.detach()
                else:
                    self.univ_perturb.tensor.data = delta.detach()
                print(
                    f"Perturbation vector with best success rate saved. Success rate:{best_success_rate:.2f}%")
        print(
            f"Training finisihed. Best success rate: {best_success_rate:.2f}%")