# -*- coding: utf-8 -*-
# Start manually added imports
import copy
from typing import Optional, Union, Tuple, List

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers import add_start_docstrings, StaticCache, DynamicCache, Cache, Gemma3TextConfig, GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer, Gemma3RMSNorm, Gemma3TextScaledWordEmbedding, Gemma3RotaryEmbedding
from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
# End manually added imports

# -*- coding: utf-8 -*-

from transformers.models.gemma3.modeling_gemma3 import *
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import TokenClassifierOutput

from .config import BiLLM_START_INDEX, logger

_CONFIG_FOR_DOC = "Gemma3TextConfig"

_CHECK_BILLM_INIT = False


@add_start_docstrings(
    "The bare Gemma3 Text Model outputting raw hidden-states without any specific head on top.",
    # We'll need to define GEMMA3_START_DOCSTRING or use a generic one
)
class Gemma3TextModel(Gemma3PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Gemma3DecoderLayer`]

    Args:
        config: Gemma3TextConfig
    """

    def __init__(self, config: Gemma3TextConfig):
        try:
            temp_num_hidden_layers = config.num_hidden_layers
            temp_hidden_size = config.hidden_size
            temp_vocab_size = config.vocab_size
        except AttributeError:
            print("Warning: config attributes not found, trying replacing config with config.get_text_config()")
            config = config.get_text_config()

        super().__init__(config)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Gemma3 downcasts the below to bfloat16, causing sqrt(3072)=55.4256 to become 55.5. See https://github.com/huggingface/transformers/pull/29402
        self.embed_tokens = Gemma3TextScaledWordEmbedding(
            self.vocab_size, config.hidden_size, self.padding_idx, embed_scale=config.hidden_size ** 0.5
        )

        # BiLLM specific: track which layers should be bidirectional
        self.bidirectionas = [BiLLM_START_INDEX > -1 and layer_idx >= BiLLM_START_INDEX
                              for layer_idx in range(config.num_hidden_layers)]

        self.layers = nn.ModuleList(
            [Gemma3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # TODO: raushan fix this after RoPE refactor. For now we hack it by reassigning thetas
        # when we want to create a local RoPE layer. Config defaults should hold values for global RoPE
        config = copy.deepcopy(config)
        config.rope_theta = config.rope_local_base_freq
        config.rope_scaling = {"rope_type": "default"}
        self.rotary_emb_local = Gemma3RotaryEmbedding(config=config)

        # Initialize weights and apply final processing
        self.post_init()

        if BiLLM_START_INDEX > -1:
            logger.info(f'Here is the Bi-Gemma3TextModel! BiLLM_START_INDEX={BiLLM_START_INDEX}')
        else:
            logger.info(f'BiLLM_START_INDEX={BiLLM_START_INDEX}, BiLLM is disabled.')

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None and not self.training:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # BiLLM: Modify attention mask for bidirectional layers
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }

            # Create the masks - BiLLM modification will be in the attention layers
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        # embed positions
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings_global = self.rotary_emb(hidden_states, position_ids)
        position_embeddings_local = self.rotary_emb_local(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # BiLLM: Modify attention mask for bidirectional layers
            layer_attention_mask = causal_mask_mapping[decoder_layer.attention_type]
            if BiLLM_START_INDEX > -1 and decoder_layer.layer_idx >= BiLLM_START_INDEX:
                # For bidirectional layers, we remove the causal mask
                if layer_attention_mask is not None:
                    # Create a non-causal attention mask (allow attending to all positions)
                    batch_size, _, seq_len, _ = layer_attention_mask.shape
                    layer_attention_mask = torch.zeros_like(layer_attention_mask)

            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings_global=position_embeddings_global,
                position_embeddings_local=position_embeddings_local,
                attention_mask=layer_attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Gemma3ForCausalLM(Gemma3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    config: Gemma3TextConfig
    base_model_prefix = "model"

    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.model = Gemma3TextModel(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            **kwargs,
    ) -> CausalLMOutputWithPast:
        if self.training and self.config._attn_implementation != "eager":
            logger.warning_once(
                "It is strongly recommended to train Gemma3 models with the `eager` attention implementation "
                f"instead of `{self.config._attn_implementation}`. Use `eager` with `AutoModelForCausalLM.from_pretrained('<path-to-checkpoint>', attn_implementation='eager')`."
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Gemma3ForSequenceClassification(Gemma3PreTrainedModel):
    def __init__(self, config: Gemma3TextConfig):
        # Store custom attributes before processing config
        custom_num_labels = getattr(config, 'num_labels', 2)
        custom_id2label = getattr(config, 'id2label', {0: "LABEL_0", 1: "LABEL_1"})
        custom_label2id = getattr(config, 'label2id', {"LABEL_0": 0, "LABEL_1": 1})
        custom_problem_type = getattr(config, 'problem_type', None)
        
        try:
            temp_num_hidden_layers = config.num_hidden_layers
            temp_hidden_size = config.hidden_size
            temp_vocab_size = config.vocab_size
        except AttributeError:
            print("Warning: config attributes not found, trying replacing config with config.get_text_config()")
            config = config.get_text_config()

        # Restore custom attributes after config processing
        config.num_labels = custom_num_labels
        config.id2label = custom_id2label
        config.label2id = custom_label2id
        config.problem_type = custom_problem_type


        super().__init__(config)

        self.num_labels = config.num_labels
        self.model = Gemma3TextModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        if _CHECK_BILLM_INIT:
            self._check_initialization()

    def _check_initialization(self):
        """
        Check that the BiLLM model is correctly initialized with:
        1. Bidirectional layers configuration
        2. Correct sequence classification head
        3. Proper model architecture
        4. Attention mask handling
        """
        print("🔍 BiLLM Initialization Check:")
        
        # Check 1: Bidirectional layers configuration
        if hasattr(self.model, 'bidirectionas'):
            bidirectional_count = sum(self.model.bidirectionas)
            total_layers = len(self.model.bidirectionas)
            print(f"  ✅ Bidirectional layers: {bidirectional_count}/{total_layers} layers")
            print(f"     BiLLM_START_INDEX: {BiLLM_START_INDEX}")
            
            if BiLLM_START_INDEX >= 0:
                expected_bidirectional = total_layers - BiLLM_START_INDEX
                if bidirectional_count == expected_bidirectional:
                    print(f"  ✅ Bidirectional configuration correct")
                else:
                    print(f"  ❌ Expected {expected_bidirectional} bidirectional layers, got {bidirectional_count}")
            else:
                print(f"  ⚠️  BiLLM disabled (BiLLM_START_INDEX={BiLLM_START_INDEX})")
        else:
            print(f"  ❌ Missing bidirectional layers configuration")
        
        # Check 2: Sequence classification head
        if hasattr(self, 'score'):
            expected_output_size = self.num_labels
            actual_output_size = self.score.out_features
            print(f"  ✅ Classification head: {actual_output_size} output features")
            
            if actual_output_size == expected_output_size:
                print(f"  ✅ Output size matches num_labels ({self.num_labels})")
            else:
                print(f"  ❌ Expected {expected_output_size} outputs, got {actual_output_size}")
                
            # Check if it's different from causal LM (no lm_head)
            if not hasattr(self, 'lm_head'):
                print(f"  ✅ No lm_head (correct for sequence classification)")
            else:
                print(f"  ⚠️  Found lm_head (unexpected for sequence classification)")
        else:
            print(f"  ❌ Missing sequence classification head (score layer)")
            
        # Check 3: Model architecture
        if hasattr(self, 'model') and hasattr(self.model, 'layers'):
            num_layers = len(self.model.layers)
            print(f"  ✅ Transformer layers: {num_layers}")
            
            # Check that layers have correct attributes for BiLLM
            sample_layer = self.model.layers[0]
            if hasattr(sample_layer, 'layer_idx'):
                print(f"  ✅ Layers have layer_idx attribute")
            else:
                print(f"  ⚠️  Layers missing layer_idx attribute")
                
            if hasattr(sample_layer, 'attention_type'):
                print(f"  ✅ Layers have attention_type attribute")
            else:
                print(f"  ⚠️  Layers missing attention_type attribute")
        else:
            print(f"  ❌ Missing model or layers")
            
        # # Check 4: Attention mask handling
        # print(f"  🎭 Attention mask check:")
        # try:
        #     # Test with a small dummy input to verify attention mask behavior
        #     dummy_input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=next(self.parameters()).device)
        #     dummy_attention_mask = torch.ones_like(dummy_input_ids)
            
        #     # Check if model can handle attention mask creation
        #     with torch.no_grad():
        #         # Just check the forward pass logic - don't run full forward
        #         inputs_embeds = self.model.embed_tokens(dummy_input_ids)
        #         batch_size, seq_len = dummy_input_ids.shape
                
        #         # Simulate the mask creation logic from forward()
        #         from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
                
        #         # Create position info
        #         cache_position = torch.arange(seq_len, device=dummy_input_ids.device)
        #         position_ids = cache_position.unsqueeze(0)
                
        #         # Test mask creation
        #         mask_kwargs = {
        #             "config": self.config,
        #             "input_embeds": inputs_embeds,
        #             "attention_mask": dummy_attention_mask,
        #             "cache_position": cache_position,
        #             "past_key_values": None,
        #             "position_ids": position_ids,
        #         }
                
        #         causal_mask = create_causal_mask(**mask_kwargs)
        #         sliding_mask = create_sliding_window_causal_mask(**mask_kwargs)
                
        #         print(f"     ✅ Causal mask shape: {causal_mask.shape}")
        #         print(f"     ✅ Sliding mask shape: {sliding_mask.shape}")
                
        #         # Check bidirectional mask modification logic
        #         if BiLLM_START_INDEX >= 0 and hasattr(self.model, 'layers'):
        #             bidirectional_layers = [l for l in self.model.layers if l.layer_idx >= BiLLM_START_INDEX]
        #             if bidirectional_layers:
        #                 # Simulate the mask modification for bidirectional layers
        #                 layer_attention_mask = causal_mask.clone()
        #                 batch_size, _, seq_len, _ = layer_attention_mask.shape
        #                 modified_mask = torch.zeros_like(layer_attention_mask)
                        
        #                 print(f"     ✅ Bidirectional mask modification: {layer_attention_mask.shape} -> zeros")
        #                 print(f"     ✅ Original mask non-zero elements: {(causal_mask != 0).sum().item()}")
        #                 print(f"     ✅ Modified mask non-zero elements: {(modified_mask != 0).sum().item()}")
        #             else:
        #                 print(f"     ⚠️  No bidirectional layers found to test mask modification")
        #         else:
        #             print(f"     ⚠️  BiLLM disabled, causal masks preserved")
                    
        # except Exception as e:
        #     print(f"     ❌ Attention mask test failed: {str(e)}")
            
        # Check 5: Config consistency
        print(f"  📋 Config check:")
        print(f"     num_labels: {self.config.num_labels}")
        print(f"     id2label: {self.config.id2label}")
        print(f"     label2id: {self.config.label2id}")
        print(f"     problem_type: {getattr(self.config, 'problem_type', 'None')}")
        
        # Summary
        critical_checks = [
            hasattr(self.model, 'bidirectionas'),
            hasattr(self, 'score'),
            hasattr(self, 'model') and hasattr(self.model, 'layers')
        ]
        
        if all(critical_checks):
            print("🎉 BiLLM initialization check PASSED!")
        else:
            print("❌ BiLLM initialization check FAILED!")
            
        print("-" * 50)

    @classmethod # NEWLY ADDED
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Load the config first
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        
        # Add sequence classification specific config from kwargs
        config.id2label = kwargs.get('id2label', {0: "LABEL_0", 1: "LABEL_1"})
        config.label2id = kwargs.get('label2id', {"LABEL_0": 0, "LABEL_1": 1})
        config.num_labels = kwargs.get('num_labels', len(config.id2label))
        
        # Ensure problem_type is set correctly
        if not hasattr(config, 'problem_type'):
            config.problem_type = None
        
        print(f"Creating model with {config.num_labels} labels: {config.id2label}")
        
        # Create the BiLLM model with custom architecture (bidirectional attention, etc.)
        model = cls(config)
        
        # Now load the pretrained weights into our custom model structure
        from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM
        try:
            # Load pretrained state dict (remove sequence classification specific kwargs)
            pretrained_kwargs = {k: v for k, v in kwargs.items() 
                               if k not in ['num_labels', 'id2label', 'label2id']}
            
            pretrained_model = Gemma3ForCausalLM.from_pretrained(
                pretrained_model_name_or_path, 
                *model_args, 
                **pretrained_kwargs
            )
            
            # Copy the weights from pretrained model to our BiLLM model
            # The BiLLM model has the same structure but with custom forward logic
            model.model.load_state_dict(pretrained_model.model.state_dict(), strict=False)
            
            print("Successfully loaded pretrained weights into BiLLM custom model!")
            
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            print("Using randomly initialized weights for BiLLM model")
        
        return model
    
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
    ) -> SequenceClassifierOutputWithPast:
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = transformer_outputs.last_hidden_state
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

            # if self.config.problem_type is None:
            #     if self.num_labels == 1:
            #         self.config.problem_type = "regression"
            #     elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
            #         self.config.problem_type = "single_label_classification"
            #     else:
            #         self.config.problem_type = "multi_label_classification"

            # if self.config.problem_type == "regression":
            #     loss_fct = MSELoss()
            #     if self.num_labels == 1:
            #         loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
            #     else:
            #         loss = loss_fct(pooled_logits, labels)
            # elif self.config.problem_type == "single_label_classification":
            #     loss_fct = CrossEntropyLoss()
            #     loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            # elif self.config.problem_type == "multi_label_classification":
            #     loss_fct = BCEWithLogitsLoss()
            #     loss = loss_fct(pooled_logits, labels)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class Gemma3ForTokenClassification(Gemma3PreTrainedModel):
    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.model = Gemma3TextModel(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
    ) -> TokenClassifierOutput:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )