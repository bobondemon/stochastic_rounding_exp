import math
import torch
from torch.ao.quantization.observer import HistogramObserver
from torch.nn import functional as F
import warnings


class MySQNRObserver(HistogramObserver):
    r"""
    The module performs Signal-to-Quantization-Noise(SQNR) quantization scheme
    Also referred to as “TF Enhanced” in AIMET.
    """

    def __init__(
        self,
        bins: int = 2048,
        upsample_rate: int = 128,
        dtype: torch.dtype = torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        factory_kwargs=None,
    ) -> None:
        # bins: The number of bins used for histogram calculation.
        super(MySQNRObserver, self).__init__(
            bins=bins,
            upsample_rate=upsample_rate,
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            factory_kwargs=factory_kwargs,
        )
        if quant_min is not None and quant_max is not None:
            self.quant_min = quant_min
            self.quant_max = quant_max
            self.dst_nbins = 2 ** math.ceil(math.log2(quant_max - quant_min))

    def _find_range_of_histogram(self):
        min_val, max_val = self.min_val.item(), self.max_val.item()
        bin_width = (max_val - min_val) / self.bins
        histogram = self.histogram.gt(1)
        # Search for the lowest bucket which has probability > 0.
        for i in range(self.bins):
            if histogram[i]:
                min_val = min_val + bin_width * i
                break
        # Search for the highest bucket which has probability > 0.
        for i in reversed(range(self.bins)):
            if histogram[i]:
                max_val = max_val - bin_width * (self.bins - i - 1)
                break
        # Make sure we include zero in range.
        min_val = min(min_val, 0)
        max_val = max(max_val, 0)
        # Make sure we have a real range.
        max_val = max(max_val, min_val + 0.01)
        return min_val, max_val

    def _pick_test_candidates_symmetric(self, min_val, max_val, num_steps, use_unsigned_symmetric=False):
        if min_val == 0.0 and use_unsigned_symmetric:
            # Special case for symmetric encodings. If all values are positive or 0, we can treat the
            # symmetric encodings as unsigned
            delta_max = max_val / num_steps
            test_offset = 0
            # Indicates all positive values
        else:
            abs_max = max(abs(min_val), abs(max_val))
            delta_max = (2 * abs_max) / num_steps
            # Compute the offset - since we are finding symmetric candidates, offset can be computed given the delta
            test_offset = math.floor(-num_steps / 2)
        # Compute the deltas we will test.
        # We test 101 deltas, equally spaced between 1*delta_max/100 and
        # 101*delta_max/100. Note we consider one delta which is larger than delta_max.
        # The reason we do this is as follows: Due to floating point rounding errors,
        # delta_max might not be able to fully cover the whole range.
        test_candidates = []
        for f in range(1, 102):
            test_delta = delta_max * f / 100.0
            test_candidates.append((test_delta, test_offset))
        return test_candidates

    def _pick_test_candidates_asymmetric(self):
        raise NotImplementedError

    def _quant_and_sat_cost(self, delta, offset, gamma=3.0):
        min_val = delta * offset
        step_size = self.dst_nbins - 1
        max_val = min_val + delta * step_size

        pdf = F.normalize(self.histogram.clamp(min=0.0), dim=-1)
        pdf_start = self.min_val.item()
        pdf_end = self.max_val.item()
        pdf_step = (pdf_end - pdf_start) / self.bins
        min_idx = math.floor((min_val - pdf_start) / pdf_step)
        min_idx = min(max(min_idx, 0), self.bins - 1)
        max_idx = math.floor((max_val - pdf_start) / pdf_step)
        max_idx = min(max(max_idx, 0), self.bins - 1)

        # Calculate the saturation cost of the bottom part of the PDF.
        sat_cost_bottom = 0
        # Calculate the smallest value we can represent (middle of respective bucket).
        min_val_middle_of_bucket = pdf_start + (min_idx * pdf_step) + pdf_step / 2
        # Go through all buckets which go into saturation.
        for i in range(min_idx):
            # Calculate the midpoint of this bin.
            mid_val = pdf_start + i * pdf_step + pdf_step / 2
            # The saturation cost is the MSE.
            sat_cost_bottom += pdf[i] * (mid_val - min_val_middle_of_bucket) ** 2

        # Calculate the saturation cost of the top part of the PDF.
        sat_cost_top = 0
        # Calculate the largest value we can represent (middle of respective bucket).
        max_val_middle_of_bucket = pdf_start + (max_idx * pdf_step) + pdf_step / 2
        # Go through all buckets which go into saturation.
        for i in range(max_idx, self.bins):
            # Calculate the midpoint of this bin.
            mid_val = pdf_start + i * pdf_step + pdf_step / 2
            # The saturation cost is the MSE.
            sat_cost_top += pdf[i] * (mid_val - max_val_middle_of_bucket) ** 2

        # Calculate the quantization cost in the middle part of the PDF.
        quant_cost = 0
        # Go through all buckets which lie in the range we can represent.
        for i in range(min_idx, max_idx):
            # The floating point value in the middle of this bucket.
            float_val = pdf_start + i * pdf_step + pdf_step / 2
            # The quantized equivalent.
            quantized = round(float_val / delta - offset)
            # The de-quantized value: this is 'floatVal' plus the quantization error.
            dequantized = delta * (quantized + offset)
            # The quantization cost is the MSE.
            quant_cost += pdf[i] * (float_val - dequantized) ** 2

        sqnr = gamma * (sat_cost_bottom + sat_cost_top) + quant_cost
        return sqnr.item()

    def _find_best_candidate(self, test_candidates):
        best_cost = torch.inf
        best_delta = -1
        best_offset = -1
        # Go through all <delta(scale), offset> pairs and calculate the quantization and saturation cost.
        # This is a 2d grid search.
        for test_delta, test_offset in test_candidates:
            cost = self._quant_and_sat_cost(test_delta, test_offset)
            # Remember the best encoding.
            if cost < best_cost:
                best_cost = cost
                best_delta = test_delta
                best_offset = test_offset
        return best_delta, best_offset

    def _min_sqnr_param_search(self):
        # Search from all possible <scale, offset> pairs for the minimum SQNR
        num_steps = self.dst_nbins - 1
        min_val, max_val = self._find_range_of_histogram()

        if self.qscheme in [torch.per_tensor_symmetric, torch.per_channel_symmetric]:
            test_candidates = self._pick_test_candidates_symmetric(
                min_val, max_val, num_steps, use_unsigned_symmetric=(self.dtype == torch.quint8)
            )
        else:
            # [TODO]: Finish this function
            test_candidates = self._pick_test_candidates_asymmetric(min_val, max_val, num_steps)

        best_delta, best_offset = self._find_best_candidate(test_candidates)
        # Calculate the new min./max. value
        min_val = best_delta * best_offset
        max_val = min_val + best_delta * num_steps
        return min_val, max_val, best_delta, best_offset

    @torch.jit.export
    def calculate_qparams(self):
        is_uninitialized = self.min_val == float("inf") and self.max_val == float("-inf")
        if is_uninitialized:
            warnings.warn("must run observer before calling calculate_qparams. Returning default scale and zero point ")
            return torch.tensor([1.0], device=self.min_val.device.type), torch.tensor(
                [0], device=self.min_val.device.type
            )
        assert self.bins == len(self.histogram), (
            "The number of bins in histogram should be equal to the number of bins "
            "supplied while making this observer"
        )

        new_min, new_max, scale, offset = self._min_sqnr_param_search()
        new_min = torch.tensor(new_min, device=self.min_val.device)
        new_max = torch.tensor(new_max, device=self.min_val.device)

        return self._calculate_qparams(new_min, new_max)

    @torch.jit.export
    def reset_min_max_vals(self):
        """Resets the min/max values."""
        self.min_val = torch.tensor(float("inf"))
        self.max_val = torch.tensor(float("-inf"))
