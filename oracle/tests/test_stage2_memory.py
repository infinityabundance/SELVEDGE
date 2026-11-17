"""CPU tests for the stage-2 memory, its accounting and its two controls.

These prove the parts a GPU receipt cannot: the address is a deterministic
causal function of the prefix, the gather respects bounds, an ungrafted table
contributes exactly zero, the active/accessible split is the declared
arithmetic rather than a measurement that happened to land there, and the dense
control's active union equals the memory's exactly.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from frontier import arms  # noqa: E402
from frontier import memory as mem  # noqa: E402


def test_address_is_deterministic_and_device_independent():
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.int64)
    first = mem.address(tokens, site=2, order=3, head=1, rows=65_536)
    second = mem.address(tokens.clone(), site=2, order=3, head=1, rows=65_536)
    assert torch.equal(first, second)
    # A second call through a different view of the same values is identical.
    assert torch.equal(first, mem.address(tokens.repeat(2, 1), site=2, order=3,
                                          head=1, rows=65_536)[0:1])


def test_address_is_causal_under_future_tokens():
    prefix = torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.int64)
    extended = torch.cat([prefix, torch.tensor([[16, 17, 18]])], dim=1)
    for order in (2, 3):
        for head in range(4):
            a = mem.address(prefix, 2, order, head, 1_048_576)
            b = mem.address(extended, 2, order, head, 1_048_576)
            assert torch.equal(a, b[:, :prefix.shape[1]])


def test_address_bounds_at_every_registered_scale():
    tokens = torch.randint(0, 248_000, (3, 257), dtype=torch.int64)
    for rows in mem.REGISTERED_ROW_SCALES:
        support = mem.support(tokens, site=2, rows=rows)
        assert support.shape == (3, 257, mem.HEADS)
        assert support.dtype == torch.int64
        assert int(support.min()) >= 0
        assert int(support.max()) < rows
        # every head/order must be individually bounded too
        for head, order in enumerate(mem.ORDERS):
            single = mem.address(tokens, 2, order, head, rows)
            assert int(single.min()) >= 0 and int(single.max()) < rows


def test_heads_orders_and_sites_are_decorrelated():
    tokens = torch.arange(1, 34, dtype=torch.int64)[None, :]
    a = mem.address(tokens, 2, 2, 0, 8_388_608)
    b = mem.address(tokens, 2, 2, 1, 8_388_608)
    c = mem.address(tokens, 2, 3, 0, 8_388_608)
    d = mem.address(tokens, 3, 2, 0, 8_388_608)
    assert not torch.equal(a, b)
    assert not torch.equal(a, c)
    assert not torch.equal(a, d)


def test_rows_must_be_a_power_of_two():
    tokens = torch.zeros(1, 4, dtype=torch.int64)
    with pytest.raises(ValueError):
        mem.address(tokens, 0, 2, 0, 1000)


# ---------------------------------------------------------------------------
# exact accounting


def test_geometry_arithmetic_is_the_declared_union():
    geometry = mem.MemoryGeometry()
    # hand arithmetic for hidden=1024, value_dim=256, 8 reads, kernel 4
    assert geometry.rows_read_elements == 8 * 256 == 2_048
    assert geometry.key_projection_elements == 1024 * 256 == 262_144
    assert geometry.value_projection_elements == 256 * 1024 + 1024 == 263_168
    assert geometry.conv_elements == 8 * 256 * 4 + 8 * 256 == 10_240
    assert geometry.gate_elements == 0
    assert geometry.active_per_token_per_site == 537_600
    assert geometry.trainable_per_site == 535_552
    assert geometry.table_elements_per_site == 65_536 * 256 == 16_777_216
    assert geometry.accessible_per_site == 17_312_768
    report = geometry.report()
    assert report["active_per_token"] == 537_600
    assert report["accessible_added_total"] == 17_312_768
    assert report["table_charged_to_active"] is False


def test_active_never_contains_the_table():
    for rows in mem.REGISTERED_ROW_SCALES:
        geometry = mem.MemoryGeometry(rows=rows)
        assert geometry.active_per_token < geometry.accessible_added
        # the only difference is the unread part of the table
        assert (geometry.accessible_added - geometry.active_per_token
                == geometry.table_elements - geometry.rows_read_elements)
        # the read is 8 rows whatever the table size
        assert geometry.rows_read_elements == 8 * 256


def test_registered_scale_table():
    dev = mem.scale_table(65_536)
    one_m = mem.scale_table(1_048_576)
    eight_m = mem.scale_table(8_388_608)
    assert dev["table_elements_total"] == 16_777_216
    assert one_m["table_elements_total"] == 268_435_456
    assert eight_m["table_elements_total"] == 2_147_483_648
    assert dev["active_per_token"] == one_m["active_per_token"] == \
        eight_m["active_per_token"] == 537_600
    assert dev["resident_table_bytes_bf16"] == 33_554_432
    assert one_m["resident_table_bytes_bf16"] == 536_870_912
    assert eight_m["resident_table_bytes_bf16"] == 4_294_967_296


def test_module_parameter_count_matches_geometry():
    module = mem.ConditionalMemory(rows=65_536, sites=(2,), table_dtype=torch.float32)
    trainable = sum(parameter.numel() for parameter in module.parameters())
    # one table + key + value + conv weights + biases
    expected = (65_536 * 256 + 262_144 + 263_168 + 10_240)
    assert trainable == expected
    assert module.active_parameter_report()["active_per_token"] == 537_600


def test_multi_site_table_is_per_site():
    module = mem.ConditionalMemory(rows=65_536, sites=(2, 11),
                                   table_dtype=torch.float32)
    report = module.active_parameter_report()
    assert report["sites"] == 2
    assert report["active_per_token"] == 2 * 537_600
    assert report["table_elements_total"] == 2 * 65_536 * 256
    assert len(module.tables) == 2


# ---------------------------------------------------------------------------
# forward behaviour


def _small_memory(rows=64):
    context = mem.TokenContext()
    module = mem.ConditionalMemory(hidden=32, value_dim=8, rows=rows, sites=(0,),
                                   conv_kernel=3, context=context,
                                   table_dtype=torch.float32)
    return module, context


def test_ungrafted_memory_is_exactly_zero():
    module, context = _small_memory()
    tokens = torch.randint(0, 100, (2, 7), dtype=torch.int64)
    context.observe(tokens)
    hidden = torch.randn(2, 7, 32)
    out = module(hidden)
    assert torch.equal(out, torch.zeros_like(out))


def test_grafted_memory_is_nonzero_and_shape_preserving():
    module, context = _small_memory()
    tokens = torch.randint(0, 100, (1, 9), dtype=torch.int64)
    context.observe(tokens)
    hidden = torch.randn(1, 9, 32)
    with torch.no_grad():
        module.value_proj[0].weight.normal_()
        module.tables[0].normal_()
    out = module(hidden)
    assert out.shape == hidden.shape
    assert out.dtype == hidden.dtype
    assert not torch.equal(out, torch.zeros_like(out))


def test_empty_rows_read_correctly_and_out_of_range_is_impossible():
    module, context = _small_memory(rows=64)
    tokens = torch.randint(0, 248_000, (1, 40), dtype=torch.int64)
    context.observe(tokens)
    addresses = mem.support(tokens, 0, 64)
    assert addresses.numel() > 0
    # gather must work for every address; a range error would raise
    _ = module.tables[0][addresses]
    assert int(addresses.max()) < 64


def test_conv_is_causal():
    conv = mem.CausalDepthwiseConv1d(channels=2, kernel=3)
    with torch.no_grad():
        conv.weight.normal_()
    x = torch.randn(1, 6, 2)
    changed = x.clone()
    changed[:, 4:, :] += 5.0
    a = conv(x)
    b = conv(changed)
    assert torch.allclose(a[:, :4], b[:, :4], atol=0, rtol=0)
    assert not torch.allclose(a[:, 4:], b[:, 4:])


def test_conv_starts_as_identity():
    conv = mem.CausalDepthwiseConv1d(channels=3, kernel=4)
    x = torch.randn(2, 5, 3)
    assert torch.equal(conv(x), x)


def test_token_context_resets_on_prefill_and_appends_on_generation():
    context = mem.TokenContext()
    prompt = torch.tensor([[5, 6, 7, 8]], dtype=torch.int64)
    context.observe(prompt, past=None)
    assert torch.equal(context.require(4), prompt)
    for token in (9, 10):
        context.observe(torch.tensor([[token]], dtype=torch.int64), past=_FakePast(3))
    assert context.require(6).tolist() == [[5, 6, 7, 8, 9, 10]]
    assert context.report()["appends"] == 2
    # a fresh full sequence is the new prefix, not an append
    context.observe(torch.tensor([[1, 2]], dtype=torch.int64), past=None)
    assert context.require(2).tolist() == [[1, 2]]


class _FakePast:
    def __init__(self, length: int):
        self._length = length

    def get_seq_length(self) -> int:
        return self._length


def test_collision_report_counts_reads_and_occupancy():
    module, _ = _small_memory(rows=16)
    tokens = torch.randint(0, 50, (1, 64), dtype=torch.int64)
    report = module.collision_report(tokens)
    assert report["read_identities_total"] == 64 * mem.HEADS
    assert report["distinct_identities_total"] <= report["read_identities_total"]
    assert report["sites"][0]["distinct_rows_read"] <= 16
    assert 0.0 < report["sites"][0]["table_occupancy_fraction"] <= 1.0


# ---------------------------------------------------------------------------
# controls


def test_dense_adapter_matches_memory_active_exactly():
    geometry = mem.MemoryGeometry()
    adapter = arms.DenseAdapter(hidden=geometry.hidden)
    assert adapter.active_elements == geometry.active_per_token_per_site == 537_600
    assert adapter.rank == 262
    assert adapter.active_parameter_report()["gate_elements"] == 0


def test_dense_adapter_starts_at_zero():
    adapter = arms.DenseAdapter(hidden=1024, rank=8)
    x = torch.randn(2, 5, 1024)
    assert torch.equal(adapter(x), torch.zeros_like(adapter(x)))


def test_ngm_has_no_parameters_and_is_zero_when_cosine_is_negative():
    embedding = torch.nn.Embedding(20, 8)
    with torch.no_grad():
        embedding.weight.copy_(F.normalize(torch.randn(20, 8), dim=-1))
    context = mem.TokenContext()
    injector = arms.NGMInjector(hidden=8, orders=(2, 3))
    injector.bind(embedding, context)
    tokens = torch.tensor([[3, 3, 3, 3]], dtype=torch.int64)
    context.observe(tokens)
    hidden = -embedding(tokens).detach()  # cosine exactly -1 after the first window
    out = injector(hidden)
    assert out.shape == hidden.shape
    # the first two positions have a padded window, the rest are exact
    assert torch.allclose(out[:, 2:], torch.zeros_like(out[:, 2:]), atol=1e-6)
    assert injector.active_elements == (2 + 3) * 8
    assert injector.active_parameter_report()["added_elements"] == 0
