import pdb

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import utils

__all__ = [
    'BigbirdBlockSpareAttention',
]


def bigbird_block_rand_mask(from_seq_length,
                            to_seq_length,
                            from_block_size,
                            to_block_size,
                            num_rand_blocks,
                            last_idx=-1):
    """Create adjacency list of random attention.
    Args:
      from_seq_length: int. length of from sequence.
      to_seq_length: int. length of to sequence.
      from_block_size: int. size of block in from sequence.
      to_block_size: int. size of block in to sequence.
      num_rand_blocks: int. Number of random chunks per row.
      last_idx: if -1 then num_rand_blocks blocks chosen anywhere in to sequence,
        if positive then num_rand_blocks blocks choosen only upto last_idx.
    Returns:
      adjacency list of size from_seq_length//from_block_size-2 by num_rand_blocks
    """
    assert from_seq_length // from_block_size == to_seq_length // to_block_size, \
        "Error the number of blocks needs to be same!"

    rand_attn = np.zeros(
        (from_seq_length // from_block_size - 2, num_rand_blocks),
        dtype=np.int32)
    # filter global attention.
    middle_seq = np.arange(
        1, to_seq_length // to_block_size - 1, dtype=np.int32)
    last = to_seq_length // to_block_size - 1
    if last_idx > (2 * to_block_size):
        last = (last_idx // to_block_size) - 1

    r = num_rand_blocks  # shorthand
    for i in range(1, from_seq_length // from_block_size - 1):
        start = i - 2
        end = i
        if i == 1:
            rand_attn[i - 1, :] = np.random.permutation(middle_seq[2:last])[:r]
        elif i == 2:
            rand_attn[i - 1, :] = np.random.permutation(middle_seq[3:last])[:r]
        elif i == from_seq_length // from_block_size - 3:
            rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            # Missing -3: should have been sliced till last-3
        elif i == from_seq_length // from_block_size - 2:
            rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            # Missing -4: should have been sliced till last-4
        else:
            if start > last:
                start = last
                rand_attn[i - 1, :] = np.random.permutation(
                    middle_seq[:start])[:r]
            elif (end + 1) == last:
                rand_attn[i - 1, :] = np.random.permutation(
                    middle_seq[:start])[:r]
            else:
                rand_attn[i - 1, :] = np.random.permutation(
                    np.concatenate((middle_seq[:start],
                                    middle_seq[end + 1:last])))[:r]
    return rand_attn


def create_rand_mask_from_inputs(from_blocked_mask, to_blocked_mask, rand_attn,
                                 num_attention_heads, num_rand_blocks,
                                 batch_size, from_seq_length, from_block_size):
    """Create 3D attention mask from a 2D tensor mask.
    Args:
      from_blocked_mask: 2D Tensor of shape [batch_size,
        from_seq_length//from_block_size, from_block_size].
      to_blocked_mask: int32 Tensor of shape [batch_size,
        to_seq_length//to_block_size, to_block_size].
      rand_attn: [batch_size, num_attention_heads,
        from_seq_length//from_block_size-2, num_rand_blocks]
      num_attention_heads: int. Number of attention heads.
      num_rand_blocks: int. Number of random chunks per row.
      batch_size: int. Batch size for computation.
      from_seq_length: int. length of from sequence.
      from_block_size: int. size of block in from sequence.

    Returns:
      float Tensor of shape [batch_size, num_attention_heads,
                             from_seq_length//from_block_size-2,
                             from_block_size, num_rand_blocks*to_block_size].
    """
    num_windows = from_seq_length // from_block_size - 2
    rand_mask = utils.torch_gather4d(to_blocked_mask, rand_attn)
    rand_mask = rand_mask.view(batch_size, num_attention_heads, num_windows,
                               num_rand_blocks * from_block_size)
    rand_mask = torch.einsum("blq,bhlk->bhlqk", from_blocked_mask[:, 1:-1],
                             rand_mask)
    return rand_mask


class BigbirdBlockSpareAttention(nn.Module):
    def __init__(self,
                 num_attention_heads,
                 size_per_head,
                 num_rand_blocks,
                 from_block_size,
                 to_block_size,
                 seed=None):
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head

        self.num_rand_blocks = num_rand_blocks
        self.from_block_size = from_block_size
        self.to_block_size = to_block_size

        self.seed = seed

    def forward(self,
                query_layer,
                key_layer,
                value_layer,
                band_mask,
                from_mask,
                to_mask,
                from_blocked_mask,
                to_blocked_mask,
                batch_size,
                from_seq_length,
                to_seq_length,
                plan_from_length=None,
                plan_num_rand_blocks=None):
        """BigBird attention sparse calculation using blocks in linear time.

        Assumes from_seq_length//from_block_size == to_seq_length//to_block_size.
        A pure function with a long argument list to allow easy use outside our
        framework.

        Args:
          query_layer: float Tensor of shape [batch_size, num_attention_heads,
            from_seq_length, size_per_head]
          key_layer: float Tensor of shape [batch_size, num_attention_heads,
            to_seq_length, size_per_head]
          value_layer: float Tensor of shape [batch_size, num_attention_heads,
            to_seq_length, size_per_head]
          band_mask: float32 Tensor of shape [batch_size, 1,
            from_seq_length//from_block_size-4, from_block_size, 3*to_block_size].
            The values should be 1 or 0. The attention scores will effectively be
            set to -infinity for any positions in the mask that are 0, and will be
            unchanged for positions that are 1.
          from_mask: float32 Tensor of shape [batch_size, 1, from_seq_length, 1].
            The values should be 1 or 0. The attention scores will effectively be set
            to -infinity for any positions in the mask that are 0, and will be
            unchanged for positions that are 1.
          to_mask: float32 Tensor of shape [batch_size, 1, 1, to_seq_length].
            The values should be 1 or 0. The attention scores will effectively be set
            to -infinity for any positions in the mask that are 0, and will be
            unchanged for positions that are 1.
          from_blocked_mask: float32 Tensor of shape [batch_size,
            from_seq_length//from_block_size, from_block_size].
            Same as from_mask, just reshaped.
          to_blocked_mask: float32 Tensor of shape [batch_size,
            to_seq_length//to_block_size, to_block_size].
            Same as to_mask, just reshaped.
          num_attention_heads: int. Number of attention heads.
          size_per_head: int. Size of each attention head.
          num_rand_blocks: int. Number of random chunks per row.
          from_seq_length: int. length of from sequence.
          to_seq_length: int. length of to sequence.
          from_block_size: int. size of block in from sequence.
          to_block_size: int. size of block in to sequence.

        Returns:
          float Tensor of shape [batch_size, from_seq_length, num_attention_heads,
            size_per_head].
        """

        if (from_seq_length // self.from_block_size !=
                to_seq_length // self.to_block_size):
            raise ValueError()

        # cast masks to float
        from_mask = from_mask.float()
        to_mask = to_mask.float()
        band_mask = band_mask.float()
        from_blocked_mask = from_blocked_mask.float()
        to_blocked_mask = to_blocked_mask.float()

        # generate random attention and corresponding masks
        np.random.seed(self.seed)
        rand_attn = [
            bigbird_block_rand_mask(  # pylint: disable=g-complex-comprehension
                from_seq_length,
                to_seq_length,
                self.from_block_size,
                self.to_block_size,
                self.num_rand_blocks,
                last_idx=1024)[:(from_seq_length // self.from_block_size - 2)]
            for _ in range(self.num_attention_heads)
        ]

        rand_attn = np.stack(rand_attn, axis=0)
        rand_attn = torch.from_numpy(rand_attn).long()
        rand_attn = torch.unsqueeze(rand_attn, 0)

        # [16, 12, 14, 3]
        rand_attn = torch.repeat_interleave(rand_attn, batch_size, 0)

        # [16, 12, 14, 64, 192]
        rand_mask = create_rand_mask_from_inputs(
            from_blocked_mask,
            to_blocked_mask,
            rand_attn,
            self.num_attention_heads,
            self.num_rand_blocks,
            batch_size,
            from_seq_length,
            self.from_block_size,
        )

        # Define shorthands
        h = self.num_attention_heads
        r = self.num_rand_blocks
        d = self.size_per_head
        b = batch_size
        m = from_seq_length
        n = to_seq_length
        wm = self.from_block_size
        wn = self.to_block_size

        # [16, 7, 16, 64, 512]
        # [batch_size, heads, seq_len // block_size, block_size, hidden_size]
        blocked_query_matrix = query_layer.view((b, h, m // wm, wm, -1))
        blocked_key_matrix = key_layer.view((b, h, n // wn, wn, -1))
        blocked_value_matrix = value_layer.view((b, h, n // wn, wn, -1))
        gathered_key = utils.torch_gather5d(blocked_key_matrix,
                                            rand_attn).view((b, h, m // wm - 2,
                                                             r * wn, -1))
        gathered_value = utils.torch_gather5d(
            blocked_value_matrix, rand_attn).view((b, h, m // wm - 2, r * wn,
                                                   -1))

        #========== component 1: First context layer ================#
        first_product = torch.einsum(
            "bhqd,bhkd->bhqk",  #
            blocked_query_matrix[:, :, 0, :, :],
            key_layer)  # [b, h, wm, -1] x [b, h, n, -1] ==> [b, h, wm, n]
        first_product = torch.mul(first_product, 1.0 / np.sqrt(d))
        first_product += (1.0 - to_mask) * -10000.0
        first_attn_weights = F.softmax(first_product, -1)  # [b, h, wm, n]
        first_context_layer = torch.einsum(
            "bhqk,bhkd->bhqd",  #
            first_attn_weights,
            value_layer)  # [b, h, wm, n] x [b, h, n, -1] ==> [b, h, wm, -1]
        first_context_layer = torch.unsqueeze(first_context_layer, 2)
        #========== First context layer ends =============================#

        #========== component 2: Second context layer =======================#
        second_key_mat = torch.cat(
            (
                blocked_key_matrix[:, :, 0, :, :],
                blocked_key_matrix[:, :, 1, :, :],
                blocked_key_matrix[:, :, 2, :, :],
                blocked_key_matrix[:, :, -1, :, :],  #
                gathered_key[:, :, 0, :, :]),
            2)  # [b, h, (4+r)*wn, -1]
        second_value_mat = torch.cat((blocked_value_matrix[:, :, 0, :, :],
                                      blocked_value_matrix[:, :, 1, :, :],
                                      blocked_value_matrix[:, :, 2, :, :],
                                      blocked_value_matrix[:, :, -1, :, :],
                                      gathered_value[:, :, 0, :, :]),
                                     2)  # [b, h, (4+r)*wn, -1]

        second_product = torch.einsum("bhqd,bhkd->bhqk",
                                      blocked_query_matrix[:, :, 1, :, :],
                                      second_key_mat)

        second_seq_pad = torch.cat(
            (to_mask[:, :, :, :3 * wn], to_mask[:, :, :, -wn:],
             torch.ones(b, 1, 1, r * wn).long()), 3)
        second_rand_pad = torch.cat(
            (torch.ones(b, h, wm, 4 * wn).long(), rand_mask[:, :, 0]), 3)
        second_product = torch.mul(second_product, 1.0 / np.sqrt(d))
        second_product += (
            1.0 - torch.minimum(second_seq_pad, second_rand_pad)) * -10000.0
        second_attn_weights = F.softmax(second_product, -1)
        second_context_layer = torch.einsum(
            "bhqk,bhkd->bhqd", second_attn_weights, second_value_mat)
        second_context_layer = torch.unsqueeze(second_context_layer, 2)
        #========== Second context layer ends ================================#

        #=================== Component 3: context layer =======================#
        exp_blocked_key_matrix = torch.cat(
            (blocked_key_matrix[:, :, 1:-3, :, :],
             blocked_key_matrix[:, :, 2:-2, :, :],
             blocked_key_matrix[:, :, 3:-1, :, :]),
            3)  # [b, h, m//wm-4, 3*wn, -1]
        exp_blocked_value_matrix = torch.cat(
            (blocked_value_matrix[:, :, 1:-3, :, :],
             blocked_value_matrix[:, :, 2:-2, :, :],
             blocked_value_matrix[:, :, 3:-1, :, :]),
            3)  # [b, h, m//wm-4, 3*wn, -1]
        middle_query_matrix = blocked_query_matrix[:, :, 2:-2, :, :]

        inner_band_product = torch.einsum(
            "bhlqd,bhlkd->bhlqk", middle_query_matrix, exp_blocked_key_matrix)
        inner_band_product = torch.mul(inner_band_product, 1.0 / np.sqrt(d))
        rand_band_product = torch.einsum("bhlqd,bhlkd->bhlqk",
                                         middle_query_matrix,
                                         gathered_key[:, :, 1:-1, :, :])

        rand_band_product = torch.mul(rand_band_product, 1.0 / np.sqrt(d))
        first_band_product = torch.einsum("bhlqd,bhkd->bhlqk",
                                          middle_query_matrix,
                                          blocked_key_matrix[:, :, 0, :, :])
        first_band_product = torch.mul(first_band_product, 1.0 / np.sqrt(d))
        last_band_product = torch.einsum("bhlqd,bhkd->bhlqk",
                                         middle_query_matrix,
                                         blocked_key_matrix[:, :, -1, :, :])

        last_band_product = torch.mul(last_band_product, 1.0 / np.sqrt(d))
        inner_band_product += (1.0 - band_mask) * -10000.0
        first_band_product += (
            1.0 - torch.unsqueeze(to_mask[:, :, :, :wn], 3)) * -10000.0
        last_band_product += (
            1.0 - torch.unsqueeze(to_mask[:, :, :, -wn:], 3)) * -10000.0
        rand_band_product += (1.0 - rand_mask[:, :, 1:-1]) * -10000.0
        band_product = torch.cat((first_band_product, inner_band_product,
                                  rand_band_product, last_band_product), -1)
        attn_weights = F.softmax(band_product, -1)
        context_layer = torch.einsum("bhlqk,bhlkd->bhlqd",
                                     attn_weights[:, :, :, :, wn:4 * wn],
                                     exp_blocked_value_matrix)
        context_layer += torch.einsum("bhlqk,bhlkd->bhlqd",
                                      attn_weights[:, :, :, :, 4 * wn:-wn],
                                      gathered_value[:, :, 1:-1, :, :])
        context_layer += torch.einsum("bhlqk,bhkd->bhlqd",
                                      attn_weights[:, :, :, :, :wn],
                                      blocked_value_matrix[:, :, 0, :, :])
        context_layer += torch.einsum("bhlqk,bhkd->bhlqd",
                                      attn_weights[:, :, :, :, -wn:],
                                      blocked_value_matrix[:, :, -1, :, :])
        #=================== context layer ends =============================#

        #=================== Component 4: second layer context layer =========#
        second_last_key_mat = torch.cat(
            (
                blocked_key_matrix[:, :, 0, :, :],
                blocked_key_matrix[:, :, -3, :, :],
                blocked_key_matrix[:, :, -2, :, :],
                blocked_key_matrix[:, :, -1, :, :],  #
                gathered_key[:, :, -1, :, :]),
            2)
        second_last_value_mat = torch.cat(
            (blocked_value_matrix[:, :, 0, :, :],
             blocked_value_matrix[:, :, -3, :, :],
             blocked_value_matrix[:, :, -2, :, :],
             blocked_value_matrix[:, :, -1, :, :],
             gathered_value[:, :, -1, :, :]), 2)
        second_last_product = torch.einsum(
            "bhqd,bhkd->bhqk",  #
            blocked_query_matrix[:, :, -2, :, :],
            second_last_key_mat)

        second_last_seq_pad = torch.cat(
            (to_mask[:, :, :, :wn], to_mask[:, :, :, -3 * wn:],
             torch.ones(b, 1, 1, r * wn).long()), 3)
        second_last_rand_pad = torch.cat(
            (torch.ones(b, h, wm, 4 * wn).long(), rand_mask[:, :, -1]), 3)
        second_last_product = torch.mul(second_last_product, 1.0 / np.sqrt(d))
        second_last_product += (1.0 - torch.minimum(
            second_last_seq_pad, second_last_rand_pad)) * -10000.0
        second_last_attn_weights = F.softmax(second_last_product, -1)
        second_last_context_layer = torch.einsum(
            "bhqk,bhkd->bhqd", second_last_attn_weights, second_last_value_mat)
        second_last_context_layer = torch.unsqueeze(second_last_context_layer,
                                                    2)
        #=================== second layer context layer ends ================#

        #================ Component 5: last context layer ====================#
        last_product = torch.einsum(
            "bhqd,bhkd->bhqk",  #
            blocked_query_matrix[:, :, -1, :, :],
            key_layer)

        last_product = torch.mul(last_product, 1.0 / np.sqrt(d))
        last_product += (1.0 - to_mask) * -10000.0
        last_attn_weights = F.softmax(last_product, -1)
        last_context_layer = torch.einsum("bhqk,bhkd->bhqd", last_attn_weights,
                                          value_layer)
        last_context_layer = torch.unsqueeze(last_context_layer, 2)
        #======================= last context layer ends ====================#

        context_layer = torch.cat(
            (first_context_layer, second_context_layer, context_layer,
             second_last_context_layer, last_context_layer), 2)
        context_layer = context_layer.view((b, h, m, -1)) * from_mask
        context_layer = context_layer.permute(0, 2, 1, 3)

        return context_layer
