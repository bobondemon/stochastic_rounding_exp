import math
import torch
from torch.ao.quantization.observer import PerChannelMinMaxObserver, HistogramObserver
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch import nn
from torch.nn import functional as F
import warnings


# https://pytorch.org/docs/stable/generated/torch.fake_quantize_per_channel_affine.html
def fake_quantize_per_channel_affine(X, scale, zero_point, ch_axis, quant_min, quant_max, rounding):
    assert ch_axis <= X.ndim and ch_axis >= 0, "[ERROR]: invalid ch_axis"
    assert scale.ndim == 1 and scale.numel() == X.shape[ch_axis], "[ERROR]: scale has wrong shape"
    assert zero_point.ndim == 1 and zero_point.numel() == X.shape[ch_axis], "[ERROR]: zero_point has wrong shape"
    rounding_op = torch.round
    if rounding == "ceil":
        rounding_op = torch.ceil
    elif rounding == "floor":
        rounding_op = torch.floor

    # e.g.: X is of shape [64, 8, 3, 3] and ch_axis = 1
    expand_shape = [1 for _ in range(X.ndim)]  # [1, 1, 1, 1]
    expand_shape[ch_axis] = X.shape[ch_axis]  # [1, 8, 1, 1]
    scale = scale.view(expand_shape)  # [1, 8, 1, 1]
    zero_point = zero_point.view(expand_shape)  # [1, 8, 1, 1]

    # [TODO]: check the stochastic rounding implementation
    # Deep Learning with Limited Numerical Precision: https://arxiv.org/pdf/1502.02551.pdf
    X_over_scale = X / scale

    floor_prob = X_over_scale.ceil() - X_over_scale
    rounding_noise = -(torch.rand(X_over_scale.size()) < floor_prob).float() + 0.5 if rounding == "stochastic" else 0.0

    round_X_div_scale = rounding_op(X_over_scale + rounding_noise)
    round_X_div_scale_ste = (X - X.detach()) / scale + round_X_div_scale.detach()
    qX = torch.clamp(round_X_div_scale_ste + zero_point, quant_min, quant_max)
    X_hat = (qX - zero_point) * scale
    return X_hat


# https://pytorch.org/docs/stable/generated/torch.fake_quantize_per_tensor_affine.html
def fake_quantize_per_tensor_affine(X, scale, zero_point, quant_min, quant_max, rounding):
    rounding_op = torch.round
    if rounding == "ceil":
        rounding_op = torch.ceil
    elif rounding == "floor":
        rounding_op = torch.floor

    # [TODO]: check the stochastic rounding implementation
    # Deep Learning with Limited Numerical Precision: https://arxiv.org/pdf/1502.02551.pdf
    X_over_scale = X / scale

    floor_prob = X_over_scale.ceil() - X_over_scale
    rounding_noise = -(torch.rand(X_over_scale.size()) < floor_prob).float() + 0.5 if rounding == "stochastic" else 0.0

    round_X_div_scale = rounding_op(X_over_scale + rounding_noise)
    round_X_div_scale_ste = (X - X.detach()) / scale + round_X_div_scale.detach()
    qX = torch.clamp(round_X_div_scale_ste + zero_point, quant_min, quant_max)
    X_hat = (qX - zero_point) * scale
    return X_hat


def check_valid_rounding_op(rounding):
    valid_op = ["round", "ceil", "floor", "stochastic"]
    assert rounding in valid_op, "[ERRO]: Only one of these values: `round`, `ceil`, `floor`, and `stochastic`"


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


class MyQParam4Weights(FakeQuantize):
    def __init__(
        self, observer=MySQNRObserver, rounding="round", quant_min=0, quant_max=255, **observer_kwargs
    ) -> None:
        super().__init__(observer, quant_min, quant_max, **observer_kwargs)
        # PyTorch version 1.10.1+cu102 has a bug here, so we need to re-create observer by passing `quant_min` and `quant_max`
        # In version 1.10.1+cu102, `FakeQuantize` doesn't pass `quant_min` and `quant_max` into it's observer object in the init function
        # The later version >=1.12 seems fixing this bug
        self.activation_post_process = observer(quant_min=quant_min, quant_max=quant_max, **observer_kwargs)
        check_valid_rounding_op(rounding)
        self.rounding = rounding

    def forward(self, w):
        self.activation_post_process.reset_min_max_vals()
        if self.observer_enabled[0] == 1:
            self.activation_post_process(w.clone().detach())
            _scale, _zero_point = self.calculate_qparams()
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point)

        if self.fake_quant_enabled[0] == 1:
            if self.is_per_channel:
                w_hat = fake_quantize_per_channel_affine(
                    w, self.scale, self.zero_point, self.ch_axis, self.quant_min, self.quant_max, self.rounding
                )
            else:
                w_hat = fake_quantize_per_tensor_affine(
                    w, self.scale, self.zero_point, self.quant_min, self.quant_max, self.rounding
                )

        return w_hat, self.scale, self.zero_point


def get_qparam4weight(qconf):
    is_symmetric = qconf["is_symmetric"]
    is_per_channel = qconf["is_per_channel"]
    is_int4 = qconf["is_int4"]  # default is quantized to int8, otherwise to int4
    rounding = qconf["rounding"]
    observer_kwargs = {}
    if is_symmetric:
        observer_kwargs.update({"dtype": torch.qint8})
        if is_per_channel:
            # Convolution kernel is of shape (out ch, in ch, ksize[0], ksize[1])
            observer_kwargs.update({"qscheme": torch.per_channel_symmetric, "ch_axis": 0})
        else:
            observer_kwargs.update({"qscheme": torch.per_tensor_symmetric})
        quant_min_max = {"quant_min": -8, "quant_max": 7} if is_int4 else {"quant_min": -128, "quant_max": 127}
    else:
        observer_kwargs.update({"dtype": torch.quint8})
        if is_per_channel:
            # Convolution kernel is of shape (out ch, in ch, ksize[0], ksize[1])
            observer_kwargs.update({"qscheme": torch.per_channel_affine, "ch_axis": 0})
        else:
            observer_kwargs.update({"qscheme": torch.per_tensor_affine})
        quant_min_max = {"quant_min": 0, "quant_max": 15} if is_int4 else {"quant_min": 0, "quant_max": 255}
    observer = PerChannelMinMaxObserver if is_per_channel else MySQNRObserver
    return MyQParam4Weights(observer=observer, rounding=rounding, **quant_min_max, **observer_kwargs)
