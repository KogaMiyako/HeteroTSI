import unittest

import torch


from heterodiff.model import HeterogeneityMaskGenerator

class MaskAwareAffinityTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.generator = HeterogeneityMaskGenerator(
            channels=5,
            seq_len=24,
            het_mask_mode="topk_sparse",
            top_k_ratio=0.4,
            bias_strength=5.0,
            het_gate_init=-2.0,
        )

    def test_hidden_values_do_not_change_distances_or_bias(self):
        x = torch.randn(3, 5, 24)
        mask = (torch.rand(3, 5, 24) > 0.3).float()
        changed = x + (1 - mask) * torch.randn_like(x) * 100

        raw, valid = self.generator.calculate_raw_distance(
            x, mask, return_validity=True
        )
        changed_raw, changed_valid = self.generator.calculate_raw_distance(
            changed, mask, return_validity=True
        )
        bias = self.generator(x, mask)
        changed_bias = self.generator(changed, mask)

        self.assertTrue(torch.equal(valid, changed_valid))
        self.assertTrue(torch.allclose(raw, changed_raw, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(bias, changed_bias, atol=1e-6, rtol=1e-6))

    def test_fully_observed_distance_matches_unmasked_formula(self):
        x = torch.randn(2, 5, 24)
        mask = torch.ones_like(x)
        seasonal, trend = self.generator.decomp(x)
        expected_seasonal = self.generator.linears(
            torch.abs(seasonal.unsqueeze(2) - seasonal.unsqueeze(1))
        ).squeeze(-1)
        expected_trend = self.generator.lineart(
            torch.abs(trend.unsqueeze(2) - trend.unsqueeze(1))
        ).squeeze(-1)
        expected = (
            self.generator.weight_t * expected_trend
            + self.generator.weight_s * expected_seasonal
        )

        actual = self.generator.calculate_raw_distance(x, mask)
        self.assertTrue(torch.allclose(expected, actual, atol=1e-6, rtol=1e-6))

    def test_unobserved_channel_is_finite_and_invalid(self):
        x = torch.randn(2, 5, 24)
        mask = torch.ones_like(x)
        mask[:, 0] = 0

        raw, valid = self.generator.calculate_raw_distance(
            x, mask, return_validity=True
        )
        bias = self.generator(x, mask)

        self.assertFalse(valid[:, 0].any().item())
        self.assertFalse(valid[:, :, 0].any().item())
        self.assertTrue(torch.isfinite(raw).all().item())
        self.assertTrue(torch.isfinite(bias).all().item())

    def test_affinity_parameters_receive_finite_gradients(self):
        x = torch.randn(3, 5, 24)
        mask = (torch.rand(3, 5, 24) > 0.2).float()
        raw = self.generator.calculate_raw_distance(x, mask)
        bias = self.generator(x, mask)
        (raw.square().mean() + bias.square().mean()).backward()

        for parameter in self.generator.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all().item())


if __name__ == "__main__":
    unittest.main()
