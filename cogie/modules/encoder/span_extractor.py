from allennlp.nn import util
from allennlp.modules.token_embedders.embedding import Embedding
import torch
from torch.nn.parameter import Parameter
from typing import List
import torch
class TimeDistributed(torch.nn.Module):
    """
    Given an input shaped like `(batch_size, time_steps, [rest])` and a `Module` that takes
    inputs like `(batch_size, [rest])`, `TimeDistributed` reshapes the input to be
    `(batch_size * time_steps, [rest])`, applies the contained `Module`, then reshapes it back.

    Note that while the above gives shapes with `batch_size` first, this `Module` also works if
    `batch_size` is second - we always just combine the first two dimensions, then split them.

    It also reshapes keyword arguments unless they are not tensors or their name is specified in
    the optional `pass_through` iterable.
    """

    def __init__(self, module):
        super().__init__()
        self._module = module

    def forward(self, *inputs, pass_through: List[str] = None, **kwargs):

        pass_through = pass_through or []

        reshaped_inputs = [self._reshape_tensor(input_tensor) for input_tensor in inputs]

        # Need some input to then get the batch_size and time_steps.
        some_input = None
        if inputs:
            some_input = inputs[-1]

        reshaped_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor) and key not in pass_through:
                if some_input is None:
                    some_input = value

                value = self._reshape_tensor(value)

            reshaped_kwargs[key] = value

        reshaped_outputs = self._module(*reshaped_inputs, **reshaped_kwargs)

        if some_input is None:
            raise RuntimeError("No input tensor to time-distribute")

        # Now get the output back into the right shape.
        # (batch_size, time_steps, **output_size)
        tuple_output = True
        if not isinstance(reshaped_outputs, tuple):
            tuple_output = False
            reshaped_outputs = (reshaped_outputs,)

        outputs = []
        for reshaped_output in reshaped_outputs:
            new_size = some_input.size()[:2] + reshaped_output.size()[1:]
            outputs.append(reshaped_output.contiguous().view(new_size))

        if not tuple_output:
            outputs = outputs[0]

        return outputs

    @staticmethod
    def _reshape_tensor(input_tensor):
        input_size = input_tensor.size()
        if len(input_size) <= 2:
            raise RuntimeError(f"No dimension to distribute: {input_size}")
        # Squash batch_size and time_steps into a single axis; result has shape
        # (batch_size * time_steps, **input_size).
        squashed_shape = [-1] + list(input_size[2:])
        return input_tensor.contiguous().view(*squashed_shape)
class SpanExtractor(torch.nn.Module):
    """
    Many NLP models deal with representations of spans inside a sentence.
    SpanExtractors define methods for extracting and representing spans
    from a sentence.

    SpanExtractors take a sequence tensor of shape (batch_size, timesteps, embedding_dim)
    and indices of shape (batch_size, num_spans, 2) and return a tensor of
    shape (batch_size, num_spans, ...), forming some representation of the
    spans.
    """

    def forward(
        self,
        sequence_tensor: torch.FloatTensor,
        span_indices: torch.LongTensor,
        sequence_mask: torch.BoolTensor = None,
        span_indices_mask: torch.BoolTensor = None,
    ):
        """
        Given a sequence tensor, extract spans and return representations of
        them. Span representation can be computed in many different ways,
        such as concatenation of the start and end spans, attention over the
        vectors contained inside the span, etc.

        # Parameters

        sequence_tensor : `torch.FloatTensor`, required.
            A tensor of shape (batch_size, sequence_length, embedding_size)
            representing an embedded sequence of words.
        span_indices : `torch.LongTensor`, required.
            A tensor of shape `(batch_size, num_spans, 2)`, where the last
            dimension represents the inclusive start and end indices of the
            span to be extracted from the `sequence_tensor`.
        sequence_mask : `torch.BoolTensor`, optional (default = `None`).
            A tensor of shape (batch_size, sequence_length) representing padded
            elements of the sequence.
        span_indices_mask : `torch.BoolTensor`, optional (default = `None`).
            A tensor of shape (batch_size, num_spans) representing the valid
            spans in the `indices` tensor. This mask is optional because
            sometimes it's easier to worry about masking after calling this
            function, rather than passing a mask directly.

        # Returns

        A tensor of shape `(batch_size, num_spans, embedded_span_size)`,
        where `embedded_span_size` depends on the way spans are represented.
        """
        raise NotImplementedError

    def get_input_dim(self) -> int:
        """
        Returns the expected final dimension of the `sequence_tensor`.
        """
        raise NotImplementedError

    def get_output_dim(self) -> int:
        """
        Returns the expected final dimension of the returned span representation.
        """
        raise NotImplementedError

class SpanExtractorWithSpanWidthEmbedding(SpanExtractor):
    """
    `SpanExtractorWithSpanWidthEmbedding` implements some common code for span
    extractors which will need to embed span width.

    Specifically, we initiate the span width embedding matrix and other
    attributes in `__init__`, leave an `_embed_spans` method that can be
    implemented to compute span embeddings in different ways, and in `forward`
    we concatenate span embeddings returned by `_embed_spans` with span width
    embeddings to form the final span representations.

    We keep SpanExtractor as a purely abstract base class, just in case someone
    wants to build a totally different span extractor.

    # Parameters

    input_dim : `int`, required.
        The final dimension of the `sequence_tensor`.
    num_width_embeddings : `int`, optional (default = `None`).
        Specifies the number of buckets to use when representing
        span width features.
    span_width_embedding_dim : `int`, optional (default = `None`).
        The embedding size for the span_width features.
    bucket_widths : `bool`, optional (default = `False`).
        Whether to bucket the span widths into log-space buckets. If `False`,
        the raw span widths are used.

    # Returns

    span_embeddings : `torch.FloatTensor`.
        A tensor of shape `(batch_size, num_spans, embedded_span_size)`,
        where `embedded_span_size` depends on the way spans are represented.
    """

    def __init__(
        self,
        input_dim: int,
        num_width_embeddings: int = None,
        span_width_embedding_dim: int = None,
        bucket_widths: bool = False,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._num_width_embeddings = num_width_embeddings
        self._bucket_widths = bucket_widths

        self._span_width_embedding= None
        if num_width_embeddings is not None and span_width_embedding_dim is not None:
            self._span_width_embedding = Embedding(
                num_embeddings=num_width_embeddings, embedding_dim=span_width_embedding_dim
            )
        elif num_width_embeddings is not None or span_width_embedding_dim is not None:
            print(
                "To use a span width embedding representation, you must"
                "specify both num_width_embeddings and span_width_embedding_dim."
            )

    def forward(
        self,
        sequence_tensor: torch.FloatTensor,
        span_indices: torch.LongTensor,
        sequence_mask: torch.BoolTensor = None,
        span_indices_mask: torch.BoolTensor = None,
    ):
        """
        Given a sequence tensor, extract spans, concatenate width embeddings
        when need and return representations of them.

        # Parameters

        sequence_tensor : `torch.FloatTensor`, required.
            A tensor of shape (batch_size, sequence_length, embedding_size)
            representing an embedded sequence of words.
        span_indices : `torch.LongTensor`, required.
            A tensor of shape `(batch_size, num_spans, 2)`, where the last
            dimension represents the inclusive start and end indices of the
            span to be extracted from the `sequence_tensor`.
        sequence_mask : `torch.BoolTensor`, optional (default = `None`).
            A tensor of shape (batch_size, sequence_length) representing padded
            elements of the sequence.
        span_indices_mask : `torch.BoolTensor`, optional (default = `None`).
            A tensor of shape (batch_size, num_spans) representing the valid
            spans in the `indices` tensor. This mask is optional because
            sometimes it's easier to worry about masking after calling this
            function, rather than passing a mask directly.

        # Returns

        A tensor of shape `(batch_size, num_spans, embedded_span_size)`,
        where `embedded_span_size` depends on the way spans are represented.
        """
        # shape (batch_size, num_spans, embedding_dim)
        span_embeddings = self._embed_spans(
            sequence_tensor, span_indices, sequence_mask, span_indices_mask
        )
        if self._span_width_embedding is not None:
            # width = end_index - start_index + 1 since `SpanField` use inclusive indices.
            # But here we do not add 1 beacuse we often initiate the span width
            # embedding matrix with `num_width_embeddings = max_span_width`
            # shape (batch_size, num_spans)
            widths_minus_one = span_indices[..., 1] - span_indices[..., 0]

            if self._bucket_widths:
                widths_minus_one = util.bucket_values(
                    widths_minus_one, num_total_buckets=self._num_width_embeddings  # type: ignore
                )

            # Embed the span widths and concatenate to the rest of the representations.
            span_width_embeddings = self._span_width_embedding(widths_minus_one)
            span_embeddings = torch.cat([span_embeddings, span_width_embeddings], -1)

        if span_indices_mask is not None:
            # Here we are masking the spans which were originally passed in as padding.
            return span_embeddings * span_indices_mask.unsqueeze(-1)

        return span_embeddings

    def get_input_dim(self) -> int:
        return self._input_dim

    def _embed_spans(
        self,
        sequence_tensor: torch.FloatTensor,
        span_indices: torch.LongTensor,
        sequence_mask: torch.BoolTensor = None,
        span_indices_mask: torch.BoolTensor = None,
    ) -> torch.Tensor:
        """
        Returns the span embeddings computed in many different ways.
        """
        raise NotImplementedError


class EndpointSpanExtractor(SpanExtractorWithSpanWidthEmbedding):
    """
    Represents spans as a combination of the embeddings of their endpoints. Additionally,
    the width of the spans can be embedded and concatenated on to the final combination.

    The following types of representation are supported, assuming that
    `x = span_start_embeddings` and `y = span_end_embeddings`.

    `x`, `y`, `x*y`, `x+y`, `x-y`, `x/y`, where each of those binary operations
    is performed elementwise.  You can list as many combinations as you want, comma separated.
    For example, you might give `x,y,x*y` as the `combination` parameter to this class.
    The computed similarity function would then be `[x; y; x*y]`, which can then be optionally
    concatenated with an embedded representation of the width of the span.

    Registered as a `SpanExtractor` with name "endpoint".

    # Parameters

    input_dim : `int`, required.
        The final dimension of the `sequence_tensor`.
    combination : `str`, optional (default = `"x,y"`).
        The method used to combine the `start_embedding` and `end_embedding`
        representations. See above for a full description.
    num_width_embeddings : `int`, optional (default = `None`).
        Specifies the number of buckets to use when representing
        span width features.
    span_width_embedding_dim : `int`, optional (default = `None`).
        The embedding size for the span_width features.
    bucket_widths : `bool`, optional (default = `False`).
        Whether to bucket the span widths into log-space buckets. If `False`,
        the raw span widths are used.
    use_exclusive_start_indices : `bool`, optional (default = `False`).
        If `True`, the start indices extracted are converted to exclusive indices. Sentinels
        are used to represent exclusive span indices for the elements in the first
        position in the sequence (as the exclusive indices for these elements are outside
        of the the sequence boundary) so that start indices can be exclusive.
        NOTE: This option can be helpful to avoid the pathological case in which you
        want span differences for length 1 spans - if you use inclusive indices, you
        will end up with an `x - x` operation for length 1 spans, which is not good.
    """

    def __init__(
        self,
        input_dim: int,
        combination: str = "x,y",
        num_width_embeddings: int = None,
        span_width_embedding_dim: int = None,
        bucket_widths: bool = False,
        use_exclusive_start_indices: bool = False,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            num_width_embeddings=num_width_embeddings,
            span_width_embedding_dim=span_width_embedding_dim,
            bucket_widths=bucket_widths,
        )
        self._combination = combination

        self._use_exclusive_start_indices = use_exclusive_start_indices
        if use_exclusive_start_indices:
            self._start_sentinel = Parameter(torch.randn([1, 1, int(input_dim)]))

    def get_output_dim(self) -> int:
        combined_dim = util.get_combined_dim(self._combination, [self._input_dim, self._input_dim])
        if self._span_width_embedding is not None:
            return combined_dim + self._span_width_embedding.get_output_dim()
        return combined_dim

    def _embed_spans(
        self,
        sequence_tensor: torch.FloatTensor,
        span_indices: torch.LongTensor,
        sequence_mask: torch.BoolTensor = None,
        span_indices_mask: torch.BoolTensor = None,
    ) -> None:
        # shape (batch_size, num_spans)
        span_starts, span_ends = [index.squeeze(-1) for index in span_indices.split(1, dim=-1)]

        if span_indices_mask is not None:
            # It's not strictly necessary to multiply the span indices by the mask here,
            # but it's possible that the span representation was padded with something other
            # than 0 (such as -1, which would be an invalid index), so we do so anyway to
            # be safe.
            span_starts = span_starts * span_indices_mask
            span_ends = span_ends * span_indices_mask

        if not self._use_exclusive_start_indices:
            if sequence_tensor.size(-1) != self._input_dim:
                raise ValueError(
                    f"Dimension mismatch expected ({sequence_tensor.size(-1)}) "
                    f"received ({self._input_dim})."
                )
            start_embeddings = util.batched_index_select(sequence_tensor, span_starts)
            end_embeddings = util.batched_index_select(sequence_tensor, span_ends)

        else:
            # We want `exclusive` span starts, so we remove 1 from the forward span starts
            # as the AllenNLP `SpanField` is inclusive.
            # shape (batch_size, num_spans)
            exclusive_span_starts = span_starts - 1
            # shape (batch_size, num_spans, 1)
            start_sentinel_mask = (exclusive_span_starts == -1).unsqueeze(-1)
            exclusive_span_starts = exclusive_span_starts * ~start_sentinel_mask.squeeze(-1)

            # We'll check the indices here at runtime, because it's difficult to debug
            # if this goes wrong and it's tricky to get right.
            if (exclusive_span_starts < 0).any():
                raise ValueError(
                    f"Adjusted span indices must lie inside the the sequence tensor, "
                    f"but found: exclusive_span_starts: {exclusive_span_starts}."
                )

            start_embeddings = util.batched_index_select(sequence_tensor, exclusive_span_starts)
            end_embeddings = util.batched_index_select(sequence_tensor, span_ends)

            # We're using sentinels, so we need to replace all the elements which were
            # outside the dimensions of the sequence_tensor with the start sentinel.
            start_embeddings = (
                start_embeddings * ~start_sentinel_mask + start_sentinel_mask * self._start_sentinel
            )

        combined_tensors = util.combine_tensors(
            self._combination, [start_embeddings, end_embeddings]
        )

        return combined_tensors


class SelfAttentiveSpanExtractor(SpanExtractorWithSpanWidthEmbedding):
    """
    Computes span representations by generating an unnormalized attention score for each
    word in the document. Spans representations are computed with respect to these
    scores by normalising the attention scores for words inside the span.

    Given these attention distributions over every span, this module weights the
    corresponding vector representations of the words in the span by this distribution,
    returning a weighted representation of each span.

    Registered as a `SpanExtractor` with name "self_attentive".

    # Parameters

    input_dim : `int`, required.
        The final dimension of the `sequence_tensor`.
    num_width_embeddings : `int`, optional (default = `None`).
        Specifies the number of buckets to use when representing
        span width features.
    span_width_embedding_dim : `int`, optional (default = `None`).
        The embedding size for the span_width features.
    bucket_widths : `bool`, optional (default = `False`).
        Whether to bucket the span widths into log-space buckets. If `False`,
        the raw span widths are used.

    # Returns

    attended_text_embeddings : `torch.FloatTensor`.
        A tensor of shape (batch_size, num_spans, input_dim), which each span representation
        is formed by locally normalising a global attention over the sequence. The only way
        in which the attention distribution differs over different spans is in the set of words
        over which they are normalized.
    """

    def __init__(
        self,
        input_dim: int,
        num_width_embeddings: int = None,
        span_width_embedding_dim: int = None,
        bucket_widths: bool = False,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            num_width_embeddings=num_width_embeddings,
            span_width_embedding_dim=span_width_embedding_dim,
            bucket_widths=bucket_widths,
        )
        self._global_attention = TimeDistributed(torch.nn.Linear(input_dim, 1))

    def get_output_dim(self) -> int:
        if self._span_width_embedding is not None:
            return self._input_dim + self._span_width_embedding.get_output_dim()
        return self._input_dim

    def _embed_spans(
        self,
        sequence_tensor: torch.FloatTensor,
        span_indices: torch.LongTensor,
        sequence_mask: torch.BoolTensor = None,
        span_indices_mask: torch.BoolTensor = None,
    ) -> torch.FloatTensor:
        # shape (batch_size, sequence_length, 1)
        global_attention_logits = self._global_attention(sequence_tensor)

        # shape (batch_size, sequence_length, embedding_dim + 1)
        concat_tensor = torch.cat([sequence_tensor, global_attention_logits], -1)

        concat_output, span_mask = util.batched_span_select(concat_tensor, span_indices)

        # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
        span_embeddings = concat_output[:, :, :, :-1]
        # Shape: (batch_size, num_spans, max_batch_span_width)
        span_attention_logits = concat_output[:, :, :, -1]

        # Shape: (batch_size, num_spans, max_batch_span_width)
        span_attention_weights = util.masked_softmax(span_attention_logits, span_mask)

        # Do a weighted sum of the embedded spans with
        # respect to the normalised attention distributions.
        # Shape: (batch_size, num_spans, embedding_dim)
        attended_text_embeddings = util.weighted_sum(span_embeddings, span_attention_weights)

        return attended_text_embeddings
