"""Training loop.

Two departures from the previous version, both from 02 SS5.1 and SS1.6:

* model selection is on free-running symbolic exact match on a *validation*
  split that is disjoint from test. Selecting on smoothed cross-entropy failed
  because the loss floor was 0.8778 and the run reached 0.8828 - a selection
  signal 5e-3 wide while exact match was still climbing.
* the reported number is the metric at the selected checkpoint, never a maximum
  over epochs of a test statistic.
"""

import copy
import json
import math
import os
import time

import torch
import torch.nn as nn

from ..config import PAD
from ..eval.decode import ConstraintMask, beam_search
from ..eval.metrics import clear_caches, evaluate_predictions


def build_optimizer(model, tcfg):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "norm" in name.lower() or name.endswith("bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": tcfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=tcfg.lr)


def build_scheduler(optimizer, tcfg, steps_per_epoch):
    total = max(1, tcfg.num_epochs * steps_per_epoch)
    warmup = max(1, int(tcfg.warmup_frac * total))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup     # never exactly zero on step 0
        progress = (step - warmup) / max(1, total - warmup)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), total, warmup


def run_epoch(model, loader, optimizer, scheduler, criterion, device, tcfg,
              train=True):
    model.train(train)
    total_loss, total_aux, n_batches = 0.0, 0.0, 0

    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        labels = batch["target"][:, 1:]

        with torch.set_grad_enabled(train):
            out = model(batch)
            loss = criterion(out["logits"].reshape(-1, out["logits"].size(-1)),
                             labels.reshape(-1))
            aux = out["aux_loss"]
            total = loss + aux

        if train:
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            optimizer.step()
            scheduler.step()

        total_loss += float(loss.detach())
        total_aux += float(aux.detach())
        n_batches += 1

    return total_loss / max(1, n_batches), total_aux / max(1, n_batches)


@torch.no_grad()
def evaluate_split(model, loader, dataset, vocab, tcfg, device, max_len,
                   constraint=None, beam_width=None):
    """Free-running decode over a split, scored with the full metric suite."""
    model.eval()
    predictions, references, templates = [], [], []

    for batch in loader:
        batch_on_device = {k: (v.to(device) if torch.is_tensor(v) else v)
                           for k, v in batch.items()}
        decoded = beam_search(model, batch_on_device, vocab,
                              beam_width=beam_width or tcfg.beam_width,
                              max_len=max_len,
                              length_penalty=tcfg.length_penalty,
                              constrained=tcfg.constrained_decoding,
                              constraint=constraint)
        for i, ids in enumerate(decoded):
            predictions.append(vocab.decode(ids[1:]))
            references.append(vocab.decode(batch["target"][i].tolist()[1:]))
        templates.extend(batch["template"])

    scored = evaluate_predictions(predictions, references, templates)
    clear_caches()          # sympy's global cache grows without bound otherwise
    return scored, predictions


def train_model(model, bundle, cfg, device, run_name="run", log=print):
    tcfg = cfg.train
    loaders = bundle.loaders
    vocab = bundle.target_vocab
    max_len = bundle.lengths[2] + 4
    constraint = ConstraintMask(vocab)

    model = model.to(device)
    optimizer = build_optimizer(model, tcfg)
    scheduler, total_steps, warmup = build_scheduler(
        optimizer, tcfg, len(loaders["train"]))
    criterion = nn.CrossEntropyLoss(ignore_index=PAD,
                                    label_smoothing=tcfg.label_smoothing)

    best_score, best_state, best_epoch = -1.0, None, -1
    epochs_without_improvement = 0
    history = []

    log(f"  {run_name}: {model.n_parameters():,} params, "
        f"{len(loaders['train'])} batches/epoch, {total_steps} steps "
        f"({warmup} warmup)")

    for epoch in range(1, tcfg.num_epochs + 1):
        t0 = time.time()
        train_loss, train_aux = run_epoch(model, loaders["train"], optimizer,
                                          scheduler, criterion, device, tcfg,
                                          train=True)
        val_loss, _ = run_epoch(model, loaders["val"], optimizer, scheduler,
                                criterion, device, tcfg, train=False)

        entry = {"epoch": epoch, "train_loss": train_loss,
                 "train_aux": train_aux, "val_loss": val_loss,
                 "lr": scheduler.get_last_lr()[0],
                 "seconds": round(time.time() - t0, 1)}

        due = (epoch % tcfg.eval_every == 0) or epoch == tcfg.num_epochs
        if due:
            val_metrics, _ = evaluate_split(
                model, loaders["val"], bundle.datasets["val"], vocab, tcfg,
                device, max_len, constraint,
                beam_width=tcfg.select_beam_width)
            score = val_metrics["symbolic_exact_match"]["value"]
            entry["val_symbolic_em"] = score
            entry["val_raw_em"] = val_metrics["raw_exact_match"]["value"]
            entry["val_parse_validity"] = val_metrics["parse_validity"]["value"]

            if score > best_score:
                best_score, best_epoch = score, epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
                entry["selected"] = True
            else:
                epochs_without_improvement += tcfg.eval_every

        history.append(entry)
        log(f"    epoch {epoch:>3} | {entry['seconds']:5.1f}s | "
            f"train {train_loss:.4f} val {val_loss:.4f}"
            + (f" | val symbolic EM {entry.get('val_symbolic_em', 0) * 100:5.1f}%"
               f"{' *' if entry.get('selected') else ''}" if due else ""))

        if epochs_without_improvement >= tcfg.patience:
            log(f"    early stop at epoch {epoch} "
                f"(best {best_score * 100:.1f}% at epoch {best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"history": history, "best_epoch": best_epoch,
            "best_val_symbolic_em": best_score}
