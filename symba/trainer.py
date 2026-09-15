"""Training loop and test-set evaluation."""

import copy
import math
import time

import torch
import torch.nn.functional as F

from .config import PAD, TOKEN_TYPES
from .data.dataset import collate, encode
from .data.serialize import is_well_formed
from .decode import beam_search
from .metrics import symbolically_equal


def _to(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def build_optimizer(model, tcfg):
    """AdamW; biases and norm weights (1-d parameters) are not decayed."""
    params = list(model.parameters())
    return torch.optim.AdamW(
        [{"params": [p for p in params if p.ndim >= 2], "weight_decay": tcfg.weight_decay},
         {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0}], lr=tcfg.lr)


def build_scheduler(optimizer, tcfg, steps_per_epoch):
    """Linear warmup, then cosine decay to zero."""
    total = tcfg.epochs * steps_per_epoch
    warmup = max(1, int(tcfg.warmup_frac * total))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_epoch(model, loader, device, optimizer=None, scheduler=None, grad_clip=1.0):
    """One pass over ``loader``, training if an optimizer is given.
    Returns the mean token loss and the teacher-forced token accuracy."""
    train = optimizer is not None
    model.train(train)
    loss_sum = correct = count = 0
    for batch in loader:
        batch = _to(batch, device)
        labels = batch["target"][:, 1:]
        with torch.set_grad_enabled(train):
            out = model(batch)
            loss = F.cross_entropy(out["logits"].flatten(0, 1), labels.flatten(),
                                   ignore_index=PAD)
        if train:
            optimizer.zero_grad(set_to_none=True)
            (loss + out["aux_loss"]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
        mask = labels != PAD
        n = int(mask.sum())
        loss_sum += loss.item() * n
        correct += int(((out["logits"].argmax(-1) == labels) & mask).sum())
        count += n
    return loss_sum / count, correct / count


@torch.no_grad()
def decode_split(model, loader, vocab, device, max_len, beam_width, length_penalty=0.0):
    """Free-running decode of a split: ``(predictions, references, record indices)``."""
    predictions, references, index = [], [], []
    for batch in loader:
        batch = _to(batch, device)
        decoded = beam_search(model, batch, vocab, beam_width, max_len, length_penalty)
        for ids, target in zip(decoded, batch["target"].tolist()):
            predictions.append(vocab.decode(ids[1:]))
            references.append(vocab.decode(target[1:]))
        index += batch["index"].tolist()
    return predictions, references, index


def train_model(model, data, cfg, device, log=print):
    """Train, keeping the weights with the best greedy sequence accuracy on val."""
    tcfg = cfg.train
    model.to(device)
    optimizer = build_optimizer(model, tcfg)
    scheduler = build_scheduler(optimizer, tcfg, len(data.loaders["train"]))
    history, best, best_state, best_epoch, stale = [], -1.0, None, 0, 0

    for epoch in range(1, tcfg.epochs + 1):
        start = time.time()
        train_loss, _ = run_epoch(model, data.loaders["train"], device, optimizer, scheduler,
                                  tcfg.grad_clip)
        val_loss, val_token_acc = run_epoch(model, data.loaders["val"], device)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "val_token_accuracy": val_token_acc, "lr": scheduler.get_last_lr()[0]}
        message = (f"epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} | "
                   f"val token acc {100 * val_token_acc:5.1f}%")

        if epoch % tcfg.eval_every == 0 or epoch == tcfg.epochs:
            preds, refs, _ = decode_split(model, data.loaders["val"], data.target_vocab,
                                          device, data.decode_budget, beam_width=1)
            acc = sum(p == r for p, r in zip(preds, refs)) / len(refs)
            row["val_sequence_accuracy"] = acc
            message += f" | val seq acc {100 * acc:5.1f}%"
            stale = 0 if acc > best else stale + tcfg.eval_every
            if acc >= best:     # on a tie keep the later, longer-trained weights
                best, best_epoch = acc, epoch
                best_state = copy.deepcopy(model.state_dict())

        row["seconds"] = round(time.time() - start, 1)
        history.append(row)
        log(f"{message} | {row['seconds']:.0f}s")
        if stale >= tcfg.patience:
            log(f"early stop: no val improvement for {tcfg.patience} epochs")
            break

    model.load_state_dict(best_state)
    return {"history": history, "best_epoch": best_epoch, "best_val_sequence_accuracy": best}


@torch.no_grad()
def evaluate(model, data, cfg, device):
    """Test-set accuracy, every test prediction, and two diagnostics for the figures."""
    loader, vocab = data.loaders["test"], data.target_vocab
    _, token_accuracy = run_epoch(model, loader, device)
    greedy, references, index = decode_split(model, loader, vocab, device, data.decode_budget, 1)
    beam, _, _ = decode_split(model, loader, vocab, device, data.decode_budget,
                              cfg.train.beam_width, cfg.train.length_penalty)
    correct = [symbolically_equal(p, r) for p, r in zip(beam, references)]
    n = len(references)
    test = {
        "n": n,
        "token_accuracy": token_accuracy,
        "sequence_accuracy": sum(p == r for p, r in zip(beam, references)) / n,
        "symbolic_accuracy": sum(correct) / n,
        "greedy_sequence_accuracy": sum(p == r for p, r in zip(greedy, references)) / n,
        "valid_expressions": sum(map(is_well_formed, beam)) / n,
    }

    predictions = []
    for i, pred, ref, ok in sorted(zip(index, beam, references, correct)):
        record = data.test[i]
        predictions.append({"file": record["file"], "line": record["line"],
                            "amp_tokens": len(record["amp_tokens"]),
                            "prediction": " ".join(pred), "reference": " ".join(ref),
                            "correct": ok})

    out = {"test": test, "predictions": predictions}
    if cfg.model.ffn == "moe":
        out["expert_routing"] = expert_routing(model, loader, vocab, device)
    if cfg.model.use_graph:
        out["attention_example"] = attention_example(model, data, device)
    return out


@torch.no_grad()
def expert_routing(model, loader, vocab, device):
    """How often each decoder layer sends each target-token type to each expert (top-1)."""
    model.eval()
    type_of = torch.tensor(vocab.type_ids())
    counts = torch.zeros(len(model.blocks), len(TOKEN_TYPES), model.blocks[0].ffn.n_experts,
                         dtype=torch.long)
    for batch in loader:
        batch = _to(batch, device)
        model(batch)
        inputs = batch["target"][:, :-1].reshape(-1).cpu()
        keep = inputs != PAD
        types = type_of[inputs[keep]]
        for layer, block in enumerate(model.blocks):
            experts = block.ffn.last_expert.cpu()[keep]
            counts[layer].index_put_((types, experts), torch.ones_like(types), accumulate=True)
    return {"token_types": TOKEN_TYPES, "counts": counts.tolist()}


@torch.no_grad()
def attention_example(model, data, device):
    """Last decoder layer's cross-attention onto the graph tokens (mean over
    heads) for the test record with the shortest target."""
    model.eval()
    record = min(data.test, key=lambda r: len(r["target_tokens"]))
    batch = _to(collate([encode(record, data.graph_vocab, data.amp_vocab, data.target_vocab)]),
                device)
    memory, memory_mask = model.encode(batch)
    _, weights = model.decode_step(batch["target"][:, :-1], memory, memory_mask,
                                   need_weights=True)
    on_graph = weights[0].mean(0)[:, :batch["graph"].size(1)].cpu()
    return {
        "file": record["file"], "line": record["line"],
        "graph_tokens": [data.graph_vocab.itos[i] for i in batch["graph"][0].tolist()],
        "output_tokens": [data.target_vocab.itos[i] for i in batch["target"][0, 1:].tolist()],
        # Share of the decoder's attention that goes to the graph at all.
        "graph_share": float(on_graph.sum(-1).mean()),
        # Rows renormalised over the graph tokens.
        "weights": [[round(float(w), 4) for w in row]
                    for row in on_graph / on_graph.sum(-1, keepdim=True)],
    }
