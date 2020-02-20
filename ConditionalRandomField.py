from typing import Dict, Optional, List, Any, Union, Tuple
import torch


class ConditionalRandomField(torch.nn.Module):
    """
    # Parameters
    num_tags : `int`, required
        The number of tags.
    constraints : `List[Tuple[int, int]]`, optional (default: None)
        An optional list of allowed transitions (from_tag_id, to_tag_id).
        These are applied to `viterbi_tags()` but do not affect `forward()`.
        These should be derived from `allowed_transitions` so that the
        start and end transitions are handled correctly for your tag type.
    include_start_end_transitions : `bool`, optional (default: True)
        Whether to include the start and end transition parameters.
    """

    def __init__(
            self,
            num_tags: int,
            constraints: List[Tuple[int, int]] = None,
            include_start_end_transitions: bool = True,
    ) -> None:
        super().__init__()
        self.num_tags = num_tags

        # transitions[i, j] is the logit for transitioning from state i to state j.
        self.transitions = torch.nn.Parameter(torch.empty(num_tags, num_tags))

        # _constraint_mask indicates valid transitions (based on supplied constraints).
        # Include special start of sequence (num_tags + 1) and end of sequence tags (num_tags + 2)
        if constraints is None:
            # All transitions are valid.
            constraint_mask = torch.ones(num_tags + 2, num_tags + 2)
        else:
            constraint_mask = torch.zeros(num_tags + 2, num_tags + 2)
            for i, j in constraints:
                constraint_mask[i, j] = 1

        self._constraint_mask = torch.nn.Parameter(constraint_mask, requires_grad=False)

        # Also need logits for transitioning from "start" state and to "end" state.
        self.include_start_end_transitions = include_start_end_transitions
        if include_start_end_transitions:
            self.start_transitions = torch.nn.Parameter(torch.empty(num_tags))
            self.end_transitions = torch.nn.Parameter(torch.empty(num_tags))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.transitions)
        if self.include_start_end_transitions:
            torch.nn.init.normal_(self.start_transitions)
            torch.nn.init.normal_(self.end_transitions)

    def _input_likelihood(self, logits: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """
        Computes the (batch_size,) denominator term for the log-likelihood, which is the
        sum of the likelihoods across all possible state sequences.
        """
        batch_size, sequence_length, num_tags = logits.size()

        # Transpose batch size and sequence dimensions
        mask = mask.transpose(0, 1).bool()
        logits = logits.transpose(0, 1)

        # Initial alpha is the (batch_size, num_tags) tensor of likelihoods combining the
        # transitions to the initial states and the logits for the first timestep.
        if self.include_start_end_transitions:
            alpha = self.start_transitions + logits[0]
        else:
            alpha = logits[0]

        # For each i we compute logits for the transitions from timestep i-1 to timestep i.
        # We do so in a (batch_size, num_tags, num_tags) tensor where the axes are
        # (instance, current_tag, next_tag)
        for i in range(1, sequence_length):
            # The emit scores are for time i ("next_tag") so we broadcast along the current_tag axis.
            emit_scores = logits[i].view(batch_size, 1, num_tags)
            # Transition scores are (current_tag, next_tag) it will be autobroadcast along the instance axis.
            # Alpha is for the current_tag, so we broadcast along the next_tag axis.
            broadcast_alpha = alpha.view(batch_size, num_tags, 1)

            # Add all the scores together and logexp over the current_tag axis.
            inner = broadcast_alpha + emit_scores + self.transitions
            # In valid positions (mask == 1) we want to take the logsumexp over the current_tag dimension
            # of `inner`. Otherwise (mask == 0) we want to retain the previous alpha.
            ##batch_size, num_tags
            # alpha = torch.logsumexp(inner, 1) * mask[i].view(batch_size, 1) + alpha * (
            #     1 - mask[i]
            # ).view(batch_size, 1)

            alpha = torch.where(mask[i].unsqueeze(1), torch.logsumexp(inner, 1), alpha)
        # Every sequence needs to end with a transition to the stop_tag.
        if self.include_start_end_transitions:
            stops = alpha + self.end_transitions
        else:
            stops = alpha

        # Finally we log_sum_exp along the num_tags dim, result is (batch_size,)
        return torch.logsumexp(stops, 1)

    def _joint_likelihood(
            self, logits: torch.Tensor, tags: torch.Tensor, mask: torch.BoolTensor
    ) -> torch.Tensor:
        """
        Computes the numerator term for the log-likelihood, which is just score(inputs, tags)
        """
        batch_size, sequence_length, _ = logits.shape

        # Transpose batch size and sequence dimensions:
        logits = logits.transpose(0, 1)
        mask = mask.transpose(0, 1).float()
        tags = tags.transpose(0, 1)

        # Start with the transition scores from start_tag to the first tag in each input
        if self.include_start_end_transitions:
            score = self.start_transitions.index_select(0, tags[0])
        else:
            score = 0.0

        # Add up the scores for the observed transitions and all the inputs but the last
        for i in range(sequence_length - 1):
            # Each is shape (batch_size,)
            current_tag, next_tag = tags[i], tags[i + 1]

            # The scores for transitioning from current_tag to next_tag
            transition_score = self.transitions[current_tag, next_tag]

            # The score for using current_tag
            emit_score = logits[i].gather(1, current_tag.view(batch_size, 1)).squeeze()

            # Include t 
            score = score + transition_score * mask[i + 1] + emit_score * mask[i]

        # Transition from last state to "stop" state. To start with, we need to find the last tag
        # for each instance.
        last_tag_index = mask.sum(0).long() - 1
        last_tags = tags.gather(0, last_tag_index.view(1, batch_size)).squeeze(0)

        # Compute score of transitioning to `stop_tag` from each "last tag".
        if self.include_start_end_transitions:
            last_transition_score = self.end_transitions.index_select(0, last_tags)
        else:
            last_transition_score = 0.0

        # Add the last input if it's not masked.
        last_inputs = logits[-1]  # (batch_size, num_tags)
        last_input_score = last_inputs.gather(1, last_tags.view(-1, 1)).squeeze()  # (batch_size)
        # last_input_score = last_input_score.squeeze()  # (batch_size,)

        score = score + last_transition_score + last_input_score * mask[-1]

        return score

    def forward(
            self, inputs: torch.Tensor, tags: torch.Tensor, mask: torch.BoolTensor = None
    ) -> torch.Tensor:
        """
        Computes the log likelihood.
        """

        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool)

        log_denominator = self._input_likelihood(inputs, mask)
        log_numerator = self._joint_likelihood(inputs, tags, mask)

        return torch.sum(log_numerator - log_denominator)

    def viterbi_tags(
            self, logits: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Uses viterbi algorithm to find most likely tags for the given inputs.
        If constraints are applied, disallows all other transitions.
        Returns a tensor with the same size as the mask (not list)
        """
        batch_size, sequence_length, num_tags = logits.size()

        if mask is None:
            mask = torch.ones(*logits.shape[:2], dtype=torch.bool, device=logits.device)

        # Augment transitions matrix with start and end transitions
        start_tag = num_tags
        end_tag = num_tags + 1

        # Apply transition constraints
        # inverse mask because torch.masked_fill will fill value when mask is True
        constraint_mask = ~(self._constraint_mask.bool())
        constrained_transitions = self.transitions.masked_fill(
            constraint_mask[:num_tags, :num_tags], -10000.0).detach()
        # transitions[:num_tags, :num_tags] = constrained_transitions

        if self.include_start_end_transitions:
            constrained_start_transitions = self.start_transitions.masked_fill(
                constraint_mask[start_tag, :num_tags],
                -10000.0
            ).detach()
            constrained_end_transitions = self.end_transitions.masked_fill(
                constraint_mask[:num_tags, end_tag],
                -10000.0
            ).detach()

        history = []
        # Transpose batch size and sequence dimensions
        mask = mask.transpose(0, 1).bool()
        logits = logits.transpose(0, 1)

        # Initial alpha is the (batch_size, num_tags) tensor of likelihoods combining the
        # transitions to the initial states and the logits for the first timestep.
        if self.include_start_end_transitions:
            score = constrained_start_transitions + logits[0]
        else:
            score = logits[0]

        # For each i we compute logits for the transitions from timestep i-1 to timestep i.
        # We do so in a (batch_size, num_tags, num_tags) tensor where the axes are
        # (instance, current_tag, next_tag)
        for i in range(1, sequence_length):
            # The emit scores are for time i ("next_tag") so we broadcast along the current_tag axis.
            emit_scores = logits[i].view(batch_size, 1, num_tags)
            broadcast_score = score.view(batch_size, num_tags, 1)
            inner = broadcast_score + emit_scores + constrained_transitions
            update_score, indices = inner.max(dim=1)
            # if mask[i] is True(mask[i] == 1 means this char is not padded) we use new_score,
            # otherwise last score.
            score = torch.where(mask[i].unsqueeze(1), update_score, score)
            history.append(indices)

        # Every sequence needs to end with a transition to the stop_tag.
        if self.include_start_end_transitions:
            score += constrained_end_transitions
        # score (batch_size, num_tgas)
        last_indices = score.argmax(-1)
        best_path = [last_indices]
        indices = last_indices
        # backtrace the path inversely
        for i in range(-1, -sequence_length, -1):
            recur_indices = history[i].gather(1, indices.view(batch_size, 1)).squeeze()
            indices = torch.where(mask[i], recur_indices, indices)
            best_path.insert(0, indices)

        best_path = torch.stack(best_path).transpose(0, 1)
        return best_path


def allowed_transitions(constraint_type: str, labels: Dict[str, int]) -> List[Tuple[int, int]]:
    """
    Given labels and a constraint type, returns the allowed transitions. It will
    additionally include transitions for the start and end states, which are used
    by the conditional random field.

    # Parameters

    constraint_type : `str`, required
        Indicates which constraint to apply. Current choices are
        "BIO", "IOB1", "BIOUL", and "BMES".
    labels : `Dict[int, str]`, required
        A mapping {label_id -> label}. Most commonly this would be the value from
        Vocabulary.get_index_to_token_vocabulary()

    # Returns

    `List[Tuple[int, int]]`
        The allowed transitions (from_label_id, to_label_id).
    """
    num_labels = len(labels)
    start_tag = num_labels
    end_tag = num_labels + 1
    labels_with_boundaries = [(k, v) for v, k in labels.items()] + \
                             [(start_tag, "START"), (end_tag, "END")]
    allowed = []
    for from_label_index, from_label in labels_with_boundaries:
        if from_label in ("START", "END"):
            from_tag = from_label
            from_entity = ""
        else:
            from_tag = from_label[0]
            from_entity = from_label[1:]
        for to_label_index, to_label in labels_with_boundaries:
            if to_label in ("START", "END"):
                to_tag = to_label
                to_entity = ""
            else:
                to_tag = to_label[0]
                to_entity = to_label[1:]
            if is_transition_allowed(constraint_type, from_tag, from_entity, to_tag, to_entity):
                allowed.append((from_label_index, to_label_index))
    return allowed


def is_transition_allowed(
        constraint_type: str, from_tag: str, from_entity: str, to_tag: str, to_entity: str
):
    """
    Given a constraint type and strings `from_tag` and `to_tag` that
    represent the origin and destination of the transition, return whether
    the transition is allowed under the given constraint type.

    # Parameters

    constraint_type : `str`, required
        Indicates which constraint to apply. Current choices are
        "BIO", BIOUL".
    from_tag : `str`, required
        The tag that the transition originates from. For example, if the
        label is `I-PER`, the `from_tag` is `I`.
    from_entity : `str`, required
        The entity corresponding to the `from_tag`. For example, if the
        label is `I-PER`, the `from_entity` is `PER`.
    to_tag : `str`, required
        The tag that the transition leads to. For example, if the
        label is `I-PER`, the `to_tag` is `I`.
    to_entity : `str`, required
        The entity corresponding to the `to_tag`. For example, if the
        label is `I-PER`, the `to_entity` is `PER`.

    # Returns

    `bool`
        Whether the transition is allowed under the given `constraint_type`.
    """

    if to_tag == "START" or from_tag == "END":
        # Cannot transition into START or from END
        return False

    if constraint_type == "BIOUL":
        if from_tag == "START":
            return to_tag in ("O", "B", "U")
        if to_tag == "END":
            return from_tag in ("O", "L", "U")
        return any(
            [
                # O can transition to O, B-* or U-*
                # L-x can transition to O, B-*, or U-*
                # U-x can transition to O, B-*, or U-*
                from_tag in ("O", "L", "U") and to_tag in ("O", "B", "U"),
                # B-x can only transition to I-x or L-x
                # I-x can only transition to I-x or L-x
                from_tag in ("B", "I") and to_tag in ("I", "L") and from_entity == to_entity,
            ]
        )

    elif constraint_type == "BIO":
        if from_tag == "START":
            return to_tag in ("O", "B")
        if to_tag == "END":
            return from_tag in ("O", "B", "I")
        return any(
            [
                # Can always transition to O or B-x
                to_tag in ("O", "B"),
                # Can only transition to I-x from B-x or I-x
                to_tag == "I" and from_tag in ("B", "I") and from_entity == to_entity,
            ]
        )
    else:
        raise KeyError(f"Unknown constraint type: {constraint_type}")
