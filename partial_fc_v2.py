
import math
from typing import Callable

import torch
from torch import distributed
from torch.nn.functional import linear, normalize

from losses import AdaFaceLoss, CombinedMarginLoss


class PartialFC_V2(torch.nn.Module):
    """
    https://arxiv.org/abs/2203.15565
    A distributed sparsely updating variant of the FC layer, named Partial FC (PFC).
    When sample rate less than 1, in each iteration, positive class centers and a random subset of
    negative class centers are selected to compute the margin-based softmax loss, all class
    centers are still maintained throughout the whole training process, but only a subset is
    selected and updated in each iteration.
    .. note::
        When sample rate equal to 1, Partial FC is equal to model parallelism(default sample rate is 1).
    Example:
    --------
    >>> module_pfc = PartialFC(embedding_size=512, num_classes=8000000, sample_rate=0.2)
    >>> for img, labels in data_loader:
    >>>     embeddings = net(img)
    >>>     loss = module_pfc(embeddings, labels)
    >>>     loss.backward()
    >>>     optimizer.step()
    """
    _version = 2

    def __init__(
        self,
        margin_loss: Callable,
        embedding_size: int,
        num_classes: int,
        sample_rate: float = 1.0,
        fp16: bool = False,
        hard_neg_mining: bool = False,
        hard_neg_ratio: float = 0.2,
        hard_neg_topk: int = 50,
        hard_neg_warmup_steps: int = 0,
        hard_neg_refresh_interval: int = 2000,
        hard_neg_queue_size: int = 8192,
    ):
        """
        Paramenters:
        -----------
        embedding_size: int
            The dimension of embedding, required
        num_classes: int
            Total number of classes, required
        sample_rate: float
            The rate of negative centers participating in the calculation, default is 1.0.
        hard_neg_mining: bool
            If True, bias the per-step negative-class subsample towards classes whose
            centers are close to the positive classes in the batch, instead of pure
            uniform random. See `configs/base.py` for the related `hard_neg_*` knobs.
        hard_neg_ratio: float
            Max fraction of `num_sample` reserved for hard negatives each step.
        hard_neg_topk: int
            Number of cached nearest-neighbor classes kept per class center.
        hard_neg_warmup_steps: int
            Number of steps to keep pure-random sampling before enabling mining.
        hard_neg_refresh_interval: int
            Steps between neighbor-cache refreshes for a given class.
        hard_neg_queue_size: int
            Size of the FIFO queue of classes recently found to produce high
            "confusing" (near-margin) logits against a positive class.
        """
        super(PartialFC_V2, self).__init__()
        assert (
            distributed.is_initialized()
        ), "must initialize distributed before create this"
        self.rank = distributed.get_rank()
        self.world_size = distributed.get_world_size()

        self.dist_cross_entropy = DistCrossEntropy()
        self.embedding_size = embedding_size
        self.sample_rate: float = sample_rate
        self.fp16 = fp16
        self.num_local: int = num_classes // self.world_size + int(
            self.rank < num_classes % self.world_size
        )
        self.class_start: int = num_classes // self.world_size * self.rank + min(
            self.rank, num_classes % self.world_size
        )
        self.num_sample: int = int(self.sample_rate * self.num_local)
        self.last_batch_size: int = 0

        self.is_updated: bool = True
        self.init_weight_update: bool = True
        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (self.num_local, embedding_size)))

        # Hard-negative mining state (see class docstring / configs/base.py)
        self.hard_neg_mining = hard_neg_mining
        self.hard_neg_ratio = hard_neg_ratio
        self.hard_neg_topk = hard_neg_topk
        self.hard_neg_warmup_steps = hard_neg_warmup_steps
        self.hard_neg_refresh_interval = hard_neg_refresh_interval
        self.hard_neg_queue_size = hard_neg_queue_size
        self.global_step = 0

        if self.hard_neg_mining:
            self.register_buffer(
                "neighbor_cache",
                torch.full((self.num_local, self.hard_neg_topk), -1, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "neighbor_cache_step",
                torch.full((self.num_local,), -1, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "confusion_queue",
                torch.full((self.hard_neg_queue_size,), -1, dtype=torch.long),
                persistent=False,
            )
            self.confusion_queue_ptr = 0

        # margin_loss
        if isinstance(margin_loss, Callable):
            self.margin_softmax = margin_loss

            if isinstance(margin_loss, AdaFaceLoss):
                self.register_buffer('batch_mean', torch.ones(1)*(20))
                self.register_buffer('batch_std', torch.ones(1)*100)
        else:
            raisee

    def set_sample_rate(self, sample_rate: float):
        """Update the PartialFC sample rate mid-training (e.g. for a fine-tune schedule)."""
        self.sample_rate = sample_rate
        self.num_sample = int(self.sample_rate * self.num_local)

    def _refresh_neighbor_cache(self, class_indices: torch.Tensor):
        """Lazily (re)compute the top-k nearest other class centers for stale/missing entries."""
        if class_indices.numel() == 0:
            return
        stale = (self.neighbor_cache_step[class_indices] < 0) | (
            self.global_step - self.neighbor_cache_step[class_indices] > self.hard_neg_refresh_interval
        )
        to_refresh = class_indices[stale]
        if to_refresh.numel() == 0:
            return

        centers = normalize(self.weight)
        query = centers[to_refresh]
        sims = query @ centers.t()
        sims[torch.arange(to_refresh.size(0), device=sims.device), to_refresh] = -2.0  # exclude self
        k = min(self.hard_neg_topk, self.num_local - 1)
        topk = sims.topk(k, dim=1).indices
        self.neighbor_cache[to_refresh, :k] = topk
        self.neighbor_cache_step[to_refresh] = self.global_step

    def _update_confusion_queue(self, logits: torch.Tensor, labels: torch.Tensor, index_positive: torch.Tensor):
        """Push classes that produced the highest non-target logit for a positive sample
        (i.e. the model's current 'closest confusion') into the FIFO hard-negative queue."""
        if index_positive.numel() == 0:
            return
        sub_logits = logits[index_positive].clone()
        target_cols = labels[index_positive].view(-1, 1)
        sub_logits.scatter_(1, target_cols, float("-inf"))
        top_val, top_col = sub_logits.max(dim=1)

        k = min(top_col.numel(), max(1, self.hard_neg_queue_size // 4))
        _, sel = top_val.topk(k)
        hard_local_idx = self.weight_index[top_col[sel]]

        n = hard_local_idx.numel()
        ptr = self.confusion_queue_ptr
        slots = (torch.arange(n, device=hard_local_idx.device) + ptr) % self.hard_neg_queue_size
        self.confusion_queue[slots] = hard_local_idx
        self.confusion_queue_ptr = (ptr + n) % self.hard_neg_queue_size

    def sample(self, labels, index_positive):
        """
            This functions will change the value of labels
            Parameters:
            -----------
            labels: torch.Tensor
                pass
            index_positive: torch.Tensor
                pass
            optimizer: torch.optim.Optimizer
                pass
        """
        with torch.no_grad():
            positive = torch.unique(labels[index_positive], sorted=True).cuda()
            if self.num_sample - positive.size(0) >= 0:
                device = self.weight.device
                chosen = positive
                mining_active = self.hard_neg_mining and self.global_step >= self.hard_neg_warmup_steps

                if mining_active:
                    self._refresh_neighbor_cache(positive)
                    hard_pool = self.neighbor_cache[positive]
                    hard_pool = hard_pool[hard_pool >= 0]
                    queued = self.confusion_queue[self.confusion_queue >= 0]
                    if queued.numel() > 0:
                        hard_pool = torch.cat([hard_pool, queued])
                    if hard_pool.numel() > 0:
                        hard_pool = torch.unique(hard_pool)
                        hard_pool = hard_pool[~torch.isin(hard_pool, positive)]

                    remaining_room = self.num_sample - positive.size(0)
                    hard_budget = min(
                        hard_pool.numel(),
                        remaining_room,
                        int(self.hard_neg_ratio * self.num_sample),
                    )
                    if hard_budget > 0:
                        if hard_pool.numel() > hard_budget:
                            sel = torch.randperm(hard_pool.numel(), device=device)[:hard_budget]
                            hard_pool = hard_pool[sel]
                        chosen = torch.cat([chosen, hard_pool])

                perm = torch.rand(size=[self.num_local], device=device)
                perm[chosen] = 2.0
                index = torch.topk(perm, k=self.num_sample)[1].to(device)
                index = index.sort()[0]
            else:
                index = positive
            self.weight_index = index

            labels[index_positive] = torch.searchsorted(index, labels[index_positive])

        return self.weight[self.weight_index]

    def forward(
        self,
        local_embeddings: torch.Tensor,
        local_labels: torch.Tensor,
    ):
        """
        Parameters:
        ----------
        local_embeddings: torch.Tensor
            feature embeddings on each GPU(Rank).
        local_labels: torch.Tensor
            labels on each GPU(Rank).
        Returns:
        -------
        loss: torch.Tensor
            pass
        """
        local_embeddings = local_embeddings.float()
        local_labels.squeeze_()
        local_labels = local_labels.long()

        batch_size = local_embeddings.size(0)
        if self.last_batch_size == 0:
            self.last_batch_size = batch_size
        assert self.last_batch_size == batch_size, (
            f"last batch size do not equal current batch size: {self.last_batch_size} vs {batch_size}")

        _gather_embeddings = [
            torch.zeros((batch_size, self.embedding_size)).cuda()
            for _ in range(self.world_size)
        ]
        _gather_labels = [
            torch.zeros(batch_size).long().cuda() for _ in range(self.world_size)
        ]
        _list_embeddings = AllGather(local_embeddings, *_gather_embeddings)
        distributed.all_gather(_gather_labels, local_labels)

        embeddings = torch.cat(_list_embeddings)
        labels = torch.cat(_gather_labels)

        labels = labels.view(-1, 1)
        index_positive = (self.class_start <= labels) & (
            labels < self.class_start + self.num_local
        )
        labels[~index_positive] = -1
        labels[index_positive] -= self.class_start

        if self.sample_rate < 1:
            weight = self.sample(labels, index_positive)
        else:
            weight = self.weight

        with torch.cuda.amp.autocast(self.fp16):
            # norm_embeddings = normalize(embeddings)
            norms = embeddings.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
            norm_embeddings = embeddings / norms

            norm_weight_activated = normalize(weight)
            logits = linear(norm_embeddings, norm_weight_activated)
        if self.fp16:
            logits = logits.float()
        logits = logits.clamp(-1, 1)

        if isinstance(self.margin_softmax, CombinedMarginLoss):
            logits = self.margin_softmax(logits=logits, labels=labels)
        elif isinstance(self.margin_softmax, AdaFaceLoss):
            logits, batch_mean, batch_std = self.margin_softmax(logits=logits, labels=labels, norms=norms,
                                                                batch_mean=self.batch_mean,
                                                                batch_std=self.batch_std)
            self.batch_mean.data = batch_mean.data
            self.batch_std.data = batch_std.data
        else:
            raise ValueError('parital FC margin_softmax not supported type')

        if self.sample_rate < 1 and self.hard_neg_mining:
            with torch.no_grad():
                row_index_positive = torch.where(index_positive.view(-1))[0]
                self._update_confusion_queue(logits, labels, row_index_positive)
            self.global_step += 1

        loss = self.dist_cross_entropy(logits, labels)
        return loss


class DistCrossEntropyFunc(torch.autograd.Function):
    """
    CrossEntropy loss is calculated in parallel, allreduce denominator into single gpu and calculate softmax.
    Implemented of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor, label: torch.Tensor):
        """ """
        batch_size = logits.size(0)
        # for numerical stability
        max_logits, _ = torch.max(logits, dim=1, keepdim=True)
        # local to global
        distributed.all_reduce(max_logits, distributed.ReduceOp.MAX)
        logits.sub_(max_logits)
        logits.exp_()
        sum_logits_exp = torch.sum(logits, dim=1, keepdim=True)
        # local to global
        distributed.all_reduce(sum_logits_exp, distributed.ReduceOp.SUM)
        logits.div_(sum_logits_exp)
        index = torch.where(label != -1)[0]
        # loss
        loss = torch.zeros(batch_size, 1, device=logits.device)
        loss[index] = logits[index].gather(1, label[index])
        distributed.all_reduce(loss, distributed.ReduceOp.SUM)
        ctx.save_for_backward(index, logits, label)
        return loss.clamp_min_(1e-30).log_().mean() * (-1)

    @staticmethod
    def backward(ctx, loss_gradient):
        """
        Args:
            loss_grad (torch.Tensor): gradient backward by last layer
        Returns:
            gradients for each input in forward function
            `None` gradients for one-hot label
        """
        (
            index,
            logits,
            label,
        ) = ctx.saved_tensors
        batch_size = logits.size(0)
        one_hot = torch.zeros(
            size=[index.size(0), logits.size(1)], device=logits.device
        )
        one_hot.scatter_(1, label[index], 1)
        logits[index] -= one_hot
        logits.div_(batch_size)
        return logits * loss_gradient.item(), None


class DistCrossEntropy(torch.nn.Module):
    def __init__(self):
        super(DistCrossEntropy, self).__init__()

    def forward(self, logit_part, label_part):
        return DistCrossEntropyFunc.apply(logit_part, label_part)


class AllGatherFunc(torch.autograd.Function):
    """AllGather op with gradient backward"""

    @staticmethod
    def forward(ctx, tensor, *gather_list):
        gather_list = list(gather_list)
        distributed.all_gather(gather_list, tensor)
        return tuple(gather_list)

    @staticmethod
    def backward(ctx, *grads):
        grad_list = list(grads)
        rank = distributed.get_rank()
        grad_out = grad_list[rank]

        dist_ops = [
            distributed.reduce(grad_out, rank, distributed.ReduceOp.SUM, async_op=True)
            if i == rank
            else distributed.reduce(
                grad_list[i], i, distributed.ReduceOp.SUM, async_op=True
            )
            for i in range(distributed.get_world_size())
        ]
        for _op in dist_ops:
            _op.wait()

        grad_out *= len(grad_list)  # cooperate with distributed loss function
        return (grad_out, *[None for _ in range(len(grad_list))])


AllGather = AllGatherFunc.apply
