
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
        pco_stage: int = 1,
        pco_m1: float = 0.4,
        pco_m2: float = 0.4,
        pco_scale: float = 64.0,
        pco_update_center_stage3: bool = False,
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
        pco_stage: int
            Progressive Cluster Optimization stage (LVFace, arXiv:2501.13420):
              1 = Feature Alignment  -> plain CosFace + NCS (default, original behaviour).
              2 = Centroid Stabilization -> maintain a per-class feature-expectation bank
                  `e_i` (Eq.9-10) and use the two-anchor loss L_s (Eq.11), still with NCS.
              3 = Boundary Refinement -> same two-anchor loss L_r (Eq.12) but the caller sets
                  sample_rate=1.0 so the negative sum runs over ALL classes (NCS disabled).
            Stages 2/3 require `margin_loss` to be CosFace-style; the margins below are used
            instead of the passed-in margin_loss.
        pco_m1: float
            Cosine margin on the classifier-weight anchor (cosθ_i - m1), Eq.11/12.
        pco_m2: float
            Cosine margin on the feature-expectation anchor (cosθ_i^e - m2), Eq.11/12.
        pco_scale: float
            Feature scale s (paper uses 64).
        pco_update_center_stage3: bool
            Algorithm 1 only updates `e` inside Centroid Stabilization (stage 2). Set True to
            keep refreshing the bank during Boundary Refinement (stage 3) as well; default
            False matches the pseudo-code (bank frozen at the value reached at end of stage 2).
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

        # Progressive Cluster Optimization (PCO) state. The feature-expectation bank `e_i`
        # is sharded exactly like `weight` (one row per *local* class) and kept in fp32.
        self.pco_stage = int(pco_stage)
        self.pco_m1 = float(pco_m1)
        self.pco_m2 = float(pco_m2)
        self.pco_scale = float(pco_scale)
        self.pco_update_center = (self.pco_stage == 2) or (
            self.pco_stage == 3 and pco_update_center_stage3
        )
        if self.pco_stage >= 2:
            # persistent=True so the bank is carried in the checkpoint from stage 2 -> stage 3.
            self.register_buffer(
                "feature_center", torch.zeros(self.num_local, embedding_size, dtype=torch.float32)
            )
            self.register_buffer(
                "center_inited", torch.zeros(self.num_local, dtype=torch.bool)
            )

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
                perm = torch.rand(size=[self.num_local], device=self.weight.device)
                perm[positive] = 2.0
                index = torch.topk(perm, k=self.num_sample)[1]
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

        # PCO stage>=2 needs the per-class *local* ids before NCS (`sample`) overwrites
        # `labels` with sampled-column indices, to index/update the feature-expectation bank.
        center_labels = labels.clone() if self.pco_stage >= 2 else None

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

        if self.pco_stage >= 2:
            return self._pco_loss(norm_embeddings, logits, labels, index_positive, center_labels)

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

        loss = self.dist_cross_entropy(logits, labels)
        return loss

    def _pco_loss(self, norm_embeddings, logits, labels, index_positive, center_labels):
        """Progressive Cluster Optimization loss for stages 2 (Eq.11) and 3 (Eq.12).

        Both stages share the *same* two-anchor objective
            L = log(1 + Sneg * e^{-s(cosθ_i - m1)} + Sneg * e^{-s(cosθ_i^e - m2)})
        where Sneg = Σ_{j≠y_i} e^{s cosθ_j} is the negative sum over the columns that are
        actually present (the NCS subset for stage 2, all classes for stage 3 when the caller
        sets sample_rate=1.0). The only stage-2-vs-3 difference handled here is whether the
        feature-expectation bank keeps being updated (`self.pco_update_center`).

        Args
        ----
        norm_embeddings : [B, d] unit-norm features of the *gathered* global batch.
        logits          : [B, L] cosθ to the local (sampled) class columns, fp32, clamped.
        labels          : [B, 1] target column index into `logits` for rows owned by this
                          rank, -1 elsewhere.
        index_positive  : [B, 1] bool, rows whose true class lives on this rank's shard.
        center_labels   : [B, 1] *local* class id (pre-NCS-remap) for owned rows, -1 else.
        """
        B = logits.size(0)
        device = logits.device
        owner_mask = index_positive.view(-1)                # [B] bool
        target_col = labels.view(-1)                        # [B] long (valid where owner)
        cls_all = center_labels.view(-1)                    # [B] local class id (valid where owner)
        nf = norm_embeddings.float()                        # [B, d] fp32, unit norm
        rows = torch.where(owner_mask)[0]

        # ---- (1) maintain the feature-expectation bank e_i  (Eq.9-10) ------------------
        # Each rank owns a disjoint class shard, so a class is touched on exactly one rank.
        # Lazy init (e_i <- x_i for a never-seen class) ALWAYS runs so the prototype anchor is
        # well-defined even with a frozen bank (stage 3); the EMA *update* of already-seen
        # classes is gated by `self.pco_update_center` (Algorithm 1: only stage 2 by default).
        if rows.numel() > 0:
            with torch.no_grad():
                cls = cls_all[rows]                         # [P] local class ids of owned rows
                feats = nf[rows]                            # [P, d]
                uniq, inv = torch.unique(cls, return_inverse=True)
                d = feats.size(1)
                sum_feat = torch.zeros(uniq.size(0), d, device=device, dtype=torch.float32)
                sum_feat.index_add_(0, inv, feats)
                cnt = torch.zeros(uniq.size(0), device=device, dtype=torch.float32)
                cnt.index_add_(0, inv, torch.ones(inv.size(0), device=device, dtype=torch.float32))
                mean_feat = normalize(sum_feat / cnt.unsqueeze(1))   # [U, d] mean direction per class
                e_old = self.feature_center[uniq]                    # [U, d]
                inited = self.center_inited[uniq].unsqueeze(1)       # [U, 1]
                if self.pco_update_center:
                    # α_i = σ(cos(e_i, x_i)) (Eq.10); e_i^new = α e_i^old + (1-α) x̄_i (Eq.9).
                    cos_e = (normalize(e_old) * mean_feat).sum(1).clamp(-1, 1)
                    alpha = torch.sigmoid(cos_e).unsqueeze(1)
                    e_ema = alpha * e_old + (1.0 - alpha) * mean_feat
                    # un-seen classes start from x_i; seen classes follow the EMA.
                    self.feature_center[uniq] = torch.where(inited, e_ema, mean_feat)
                    self.center_inited[uniq] = True
                elif (~inited).any():
                    # frozen bank: only lazily initialise classes never seen in stage 2.
                    new = (~inited).squeeze(1)
                    self.feature_center[uniq[new]] = mean_feat[new]
                    self.center_inited[uniq[new]] = True

        # ---- (2) prototype cosine cosθ_i^e = cos(x_i, e_{y_i})  (2nd anchor) ------------
        # e is a frozen buffer here -> detach; gradient flows only into the feature x_i,
        # pulling it towards its (fixed) class prototype.
        proto_cos = torch.zeros(B, device=device, dtype=torch.float32)
        if rows.numel() > 0:
            e = normalize(self.feature_center[cls_all[rows]].detach())   # [P, d]
            proto_cos = proto_cos.index_put((rows,), (nf[rows] * e).sum(1))

        # ---- (3) two-anchor distributed loss (Eq.11 / Eq.12) ---------------------------
        return PCODistLossFunc.apply(
            logits.float(), proto_cos, target_col, owner_mask,
            self.pco_scale, self.pco_m1, self.pco_m2,
        )


class PCODistLossFunc(torch.autograd.Function):
    """Distributed two-anchor PCO loss (LVFace Eq.11/12), sharded over class columns.

    Per (gathered) sample i, with z_i = cosθ_i (classifier-weight anchor), p_i = cosθ_i^e
    (feature-expectation anchor) and Sneg_i = Σ_{j≠y_i} e^{s cosθ_j}:

        L_i = log(1 + Sneg_i e^{-s(z_i - m1)} + Sneg_i e^{-s(p_i - m2)}),   loss = mean_i L_i

    The class columns are split across ranks; the true class y_i (hence z_i and p_i) lives on
    exactly one rank. Forward replicates the per-row scalars via all_reduce; backward returns
    the *local* gradient w.r.t. this rank's columns, so it composes with the existing AllGather
    (which sums embedding grads across ranks) exactly like DistCrossEntropyFunc does.

    Sanity check: with only the first anchor (p term absent) this reduces *exactly* to
    DistCrossEntropy(+CosFace) — grad_neg = (s/B)·softmax_neg, grad_target = -(s/B)·(A/D).
    """

    @staticmethod
    def forward(ctx, cos_logits, proto_cos, target_col, owner_mask, s, m1, m2):
        device = cos_logits.device
        B = cos_logits.size(0)
        rows = torch.where(owner_mask)[0]

        scaled = s * cos_logits                                      # [B, L]
        # global max over all class columns (across ranks) for numerical stability.
        gmax = scaled.max(dim=1).values                             # [B]
        distributed.all_reduce(gmax, distributed.ReduceOp.MAX)
        el = torch.exp(scaled - gmax.unsqueeze(1))                  # [B, L] in (0, 1]

        full_sum = el.sum(dim=1)                                    # [B] local partial (shifted)
        distributed.all_reduce(full_sum, distributed.ReduceOp.SUM)  # Σ_all e^{s cosθ} · e^{-gmax}

        # target cosine z_i and prototype cosine p_i: each owned by one rank -> SUM to replicate.
        z = torch.zeros(B, device=device, dtype=cos_logits.dtype)
        if rows.numel() > 0:
            z[rows] = cos_logits[rows, target_col[rows]]
        distributed.all_reduce(z, distributed.ReduceOp.SUM)
        p = proto_cos.clone()
        distributed.all_reduce(p, distributed.ReduceOp.SUM)

        el_target = torch.exp(s * z - gmax)                        # [B] shifted target exp
        # Sneg excludes the true class -> subtract its (raw-cosine) contribution.
        snegsh = (full_sum - el_target).clamp_min(1e-30)           # [B] = e^{-gmax}·Sneg
        log_snegsh = torch.log(snegsh)

        # log A_i and log B_i with the same gmax shift (A = Sneg e^{-s(z-m1)}, B = Sneg e^{-s(p-m2)}).
        logA = log_snegsh + gmax - s * (z - m1)
        logB = log_snegsh + gmax - s * (p - m2)
        zeros = torch.zeros_like(logA)
        logD = torch.logsumexp(torch.stack([zeros, logA, logB], dim=0), dim=0)  # log(1 + A + B)
        loss = logD.mean()

        ratioA = torch.exp(logA - logD)                            # A/D ∈ [0,1]
        ratioB = torch.exp(logB - logD)                            # B/D ∈ [0,1]
        # negative-column gradient coefficient: (1/B)·s·el·R, with R = (A/D + B/D)/Sneg_shifted.
        R = (ratioA + ratioB) / snegsh

        ctx.save_for_backward(el, R, ratioA, ratioB, target_col, owner_mask)
        ctx.s = float(s)
        ctx.B = B
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        el, R, ratioA, ratioB, target_col, owner_mask = ctx.saved_tensors
        s, B = ctx.s, ctx.B
        # all local columns treated as negatives first ...
        grad_cos = (s / B) * el * R.unsqueeze(1)                   # [B, L]
        rows = torch.where(owner_mask)[0]
        if rows.numel() > 0:
            # ... then overwrite the true-class column on its owner rank (excluded from Sneg).
            grad_cos[rows, target_col[rows]] = (-s / B) * ratioA[rows]
        grad_cos = grad_cos * grad_out

        grad_proto = torch.zeros(B, device=el.device, dtype=el.dtype)
        if rows.numel() > 0:
            grad_proto[rows] = (-s / B) * ratioB[rows]
        grad_proto = grad_proto * grad_out

        return grad_cos, grad_proto, None, None, None, None, None


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
