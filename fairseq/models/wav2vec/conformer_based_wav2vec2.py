import logging
from typing import List

import torch
import torch.nn as nn

from fairseq import checkpoint_utils
from fairseq.data.audio.audio_utils import get_mel_spectrogram
from fairseq.data.audio.speech_to_text_dataset import get_features_or_waveform
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    register_model,
    register_model_architecture, BaseFairseqModel,
)
from fairseq.models.speech_to_text import S2TTransformerModel, S2TTransformerEncoder, Conv1dSubsampler
from fairseq.models.wav2vec import ConvFeatureExtractionModel, Wav2Vec2Model
from fairseq.modules import (
    GradMultiply,
    ConformerEncoderLayer, LayerNorm, GumbelVectorQuantizer, FairseqDropout, PositionalEmbedding,
)
from fairseq.modules.layer_history import CreateLayerHistory
from fairseq.utils import is_xla_tensor

logger = logging.getLogger(__name__)


@register_model("conformer_based_wav2vec2")
class ConformerBasedWav2vec2Model(BaseFairseqModel):
    """Adapted Transformer model (https://arxiv.org/abs/1706.03762) for
    speech-to-text tasks. The Transformer encoder/decoder remains the same.
    A trainable input subsampler is prepended to the Transformer encoder to
    project inputs into the encoder dimension as well as downsample input
    sequence for computational efficiency."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        S2TTransformerModel.add_args(parser)

        parser.add_argument(
            "--macaron-style",
            default=False,
            type=bool,
            help="Whether to use macaron style for positionwise layer",
        )
        # Attention
        parser.add_argument(
            "--zero-triu",
            default=False,
            type=bool,
            help="If true, zero the uppper triangular part of attention matrix.",
        )
        # Relative positional encoding
        parser.add_argument(
            "--rel-pos-type",
            type=str,
            default="legacy",
            choices=["legacy", "latest"],
            help="Whether to use the latest relative positional encoding or the legacy one."
                 "The legacy relative positional encoding will be deprecated in the future."
                 "More Details can be found in https://github.com/espnet/espnet/pull/2816.",
        )
        # CNN module
        parser.add_argument(
            "--use-cnn-module",
            default=False,
            type=bool,
            help="Use convolution module or not",
        )
        parser.add_argument(
            "--cnn-module-kernel",
            default=31,
            type=int,
            help="Kernel size of convolution module.",
        )
        pass

    @classmethod
    def build_model(cls, cfg, task=None, embed_tokens=None):
        encoder = ConformerWav2vec2Encoder(cfg)
        if getattr(cfg, "load_pretrained_from", None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=cfg.load_pretrained_from, strict=False
            )
            logger.info(
                f"loaded pretrained encoder from: "
                f"{cfg.load_pretrained_encoder_from}"
            )
        return encoder




class Conv2dSubsampler(nn.Module):
    """Convolutional subsampler: a stack of 1D convolution (along temporal
    dimension) followed by non-linear activation via gated linear units
    (https://arxiv.org/abs/1911.08460)

    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = (3, 3),
    ):
        super(Conv2dSubsampler, self).__init__()
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            nn.Conv2d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride=2,
                padding=k // 2,
            )
            for i, k in enumerate(kernel_sizes)
        )

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for _ in range(self.n_layers):
            out = ((out.float() - 1) / 2 + 1).floor().long()
        return out

    def forward(self, src_tokens):
        # src_tokens  B x H x W
        x = src_tokens.unsqueeze(1).contiguous()
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
        return x


class ConformerWav2vec2Encoder(Wav2Vec2Model):
    """Speech-to-text Conformer encoder that consists of input subsampler and
    Transformer encoder."""

    def __init__(self, cfg):
        super().__init__(cfg)
        del self.encoder
        del self.feature_extractor

        self.padding_idx = 1
        self.is_mel = getattr(cfg, "is_mel_spectrograms", False)
        if self.is_mel:
            self.subsample = Conv2dSubsampler(
                cfg.input_feat_per_channel,
                cfg.conv_channels,
                cfg.encoder_embed_dim,
                [int(k) for k in cfg.conv_kernel_sizes.split(",")],
            )
            self.linear = nn.Linear(cfg.encoder_embed_dim, cfg.encoder_embed_dim)
        else:
            self.subsample = Conv1dSubsampler(
                cfg.input_feat_per_channel,
                cfg.conv_channels,
                cfg.encoder_embed_dim,
                [int(k) for k in cfg.conv_kernel_sizes.split(",")],
            )
        # embed_dim for conformerEncoder layer embedding dim
        self.layers = nn.ModuleList(
            [ConformerEncoderLayer(cfg) for _ in range(cfg.encoder_layers)]
        )
        self.attn_type = getattr(cfg, "encoder_attention_type", "selfattn")
        self.padding_idx = 1
        self.embed_positions = PositionalEmbedding(
            cfg.max_source_positions, cfg.encoder_embed_dim, self.padding_idx, pos_emb_type=self.attn_type
        )
        if getattr(cfg, "use_enc_dlcl", False):
            self.history = CreateLayerHistory(cfg, is_encoder=True)
        else:
            self.history = None
        self.softmax = nn.Softmax(dim=-1)
        self.dropout_module = FairseqDropout(
            p=cfg.dropout, module_name=self.__class__.__name__
        )



    def get_interactive_tokens_and_lengths(self, lines):
        n_frames = [p.shape[0] for p in lines]
        return lines, n_frames

    def extract_features(self, source, padding_mask, mask=False, layer=None, mel_spectrograms=None):
        res = self.forward(
            source, padding_mask=padding_mask, mask=mask, features_only=True, layer=layer, mel_spectrograms=mel_spectrograms
        )
        return res

    def forward(self,
                source,
                mel_spectrograms=None,
                padding_mask=None,
                mask=True,
                features_only=False,
                layer=None,
                mask_indices=None,
                mask_channel_indices=None,
                padding_count=None,
                ):

        if self.history is not None:
            self.history.clean()

        # todo spec  mel 899
        if self.feature_grad_mult > 0:
            features = self.subsample(mel_spectrograms if self.is_mel else source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.subsample(mel_spectrograms if self.is_mel else source)

        features_pen = features.float().pow(2).mean()

        if self.is_mel:
            # B x T x C
            features = self.linear(features.reshape(-1, features.shape[0], self.cfg.encoder_embed_dim))

        features = features.transpose(0, 1)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        num_vars = None
        code_ppl = None
        prob_ppl = None
        curr_temp = None

        if self.input_quantizer:
            q = self.input_quantizer(features, produce_targets=False)
            features = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]
            features = self.project_inp(features)

        if mask:
            x, mask_indices = self.apply_mask(
                features,
                padding_mask,
                mask_indices=mask_indices,
                mask_channel_indices=mask_channel_indices,
            )
            if not is_xla_tensor(x) and mask_indices is not None:
                # tpu-comment: reducing the size in a dynamic way causes
                # too many recompilations on xla.
                y = unmasked_features[mask_indices].view(
                    unmasked_features.size(0), -1, unmasked_features.size(-1)
                )
            else:
                y = unmasked_features
        else:
            x = features
            y = unmasked_features
            mask_indices = None

        x = self.dropout_module(x)
        input_lengths = torch.tensor([s.size(0) for s in features]).to(x.device)
        encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        positions = self.embed_positions(encoder_padding_mask)
        encoder_padding_mask = encoder_padding_mask.transpose(0, 1)
        if self.attn_type != "rel_selfattn":
            x += positions

        x = self.dropout_module(x)
        positions = self.dropout_module(positions)

        # add emb into history
        if self.history is not None:
            self.history.add(x)
        layer_results = []
        for layer in self.layers:
            if self.history is not None:
                x = self.history.pop()
            x, z = layer(x, encoder_padding_mask, pos_emb=positions)
            layer_results.append((x, z))
            if self.history is not None:
                self.history.add(x)

        if self.history is not None:
            x = self.history.pop()

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if features_only:
            return {
                "x": x,
                "padding_mask": encoder_padding_mask,
                "features": unmasked_features,
                "layer_results": layer_results
            }
        if self.quantizer:
            q = self.quantizer(y, produce_targets=False)
            y = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]

            y = self.project_q(y)

            if self.negatives_from_everywhere:
                neg_cands = self.quantizer(unmasked_features, produce_targets=False)[
                    "x"
                ]
                negs, _ = self.sample_negatives(
                    neg_cands,
                    y.size(1),
                    padding_count=padding_count,
                )
                negs = self.project_q(negs)

            else:
                negs, _ = self.sample_negatives(
                    y,
                    y.size(1),
                    padding_count=padding_count,
                )

            if self.codebook_negatives > 0:
                cb_negs = self.quantizer.sample_from_codebook(
                    y.size(0) * y.size(1), self.codebook_negatives
                )
                cb_negs = cb_negs.view(
                    self.codebook_negatives, y.size(0), y.size(1), -1
                )  # order doesnt matter
                cb_negs = self.project_q(cb_negs)
                negs = torch.cat([negs, cb_negs], dim=0)
        else:
            y = self.project_q(y)

            if self.negatives_from_everywhere:
                negs, _ = self.sample_negatives(
                    unmasked_features,
                    y.size(1),
                    padding_count=padding_count,
                )
                negs = self.project_q(negs)
            else:
                negs, _ = self.sample_negatives(
                    y,
                    y.size(1),
                    padding_count=padding_count,
                )

        if not is_xla_tensor(x):
            # tpu-comment: reducing the size in a dynamic way causes
            # too many recompilations on xla.
            x = x[mask_indices].view(x.size(0), -1, x.size(-1))

        if self.target_glu:
            y = self.target_glu(y)
            negs = self.target_glu(negs)

        x = self.final_proj(x)
        x = self.compute_preds(x, y, negs)

        result = {
            "x": x,
            "padding_mask": padding_mask,
            "features_pen": features_pen,
        }

        if prob_ppl is not None:
            result["prob_perplexity"] = prob_ppl
            result["code_perplexity"] = code_ppl
            result["num_vars"] = num_vars
            result["temp"] = curr_temp

        return result


@register_model_architecture(model_name="conformer_based_wav2vec2", arch_name="conformer_based_wav2vec2")
def base_architecture(args):
    # Convolutional subsampler
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    # Conformer
    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 80)
    args.macaron_style = getattr(args, "macaron_style", True)
    args.use_cnn_module = getattr(args, "use_cnn_module", True)
    args.cnn_module_kernel = getattr(args, "cnn_module_kernel", 31)
    # dim, k, stride
    args.conv_feature_layers = getattr(args, "conv_feature_layers",
                                       '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]')
    args.max_source_positions = getattr(args, "encoder_embed_dim", 3000)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_type = getattr(args, "encoder_attention_type", "selfattn")
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_type = getattr(args, "decoder_attention_type", "selfattn")
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)

    args.max_encoder_relative_length = getattr(args, 'max_encoder_relative_length', -1)
    args.max_decoder_relative_length = getattr(args, 'max_decoder_relative_length', -1)
    args.k_only = getattr(args, 'k_only', True)


@register_model_architecture("conformer_based_wav2vec2", "conformer_based_wav2vec2_s")
def s2t_conformer_s(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)


@register_model_architecture("conformer_based_wav2vec2", "conformer_based_wav2vec2_s_relative")
def s2t_conformer_s_relative(args):
    args.max_encoder_relative_length = 100
    args.max_decoder_relative_length = 20
    args.k_only = True
    s2t_conformer_s(args)


@register_model_architecture("conformer_based_wav2vec2", "conformer_based_wav2vec2_xs")
def s2t_conformer_xs(args):
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 4)
    args.dropout = getattr(args, "dropout", 0.3)
    s2t_conformer_s(args)


@register_model_architecture("conformer_based_wav2vec2", "conformer_based_wav2vec2_sp")
def s2t_conformer_sp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_conformer_s(args)


@register_model_architecture("conformer_based_wav2vec2", "conformer_based_wav2vec2_m")
def s2t_conformer_m(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.15)
    base_architecture(args)


@register_model_architecture("conformer_based_wav2vec2", "conformer_based_wav2vec2_mp")
def s2t_conformer_mp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_conformer_m(args)


@register_model_architecture("conformer_based_wav2vec2", "conformer_based_wav2vec2_l")
def s2t_conformer_l(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.2)
    base_architecture(args)


@register_model_architecture("conformer_based_wav2vec2", "conformer_based_wav2vec2_lp")
def s2t_conformer_lp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_conformer_l(args)
