"""CPU reference arithmetic only; this module neither loads weights nor runs a model."""
import math
from collections import defaultdict

YES_ID = 18830
NO_ID = 42192


def require(value, message):
    if not value:
        raise ValueError(message)


def logsumexp(values):
    high = max(values)
    return high + math.log(math.fsum(math.exp(v - high) for v in values))


def resolution(value, bound):
    require(math.isfinite(value), 'nonfinite score')
    if bound is None:
        return 'unqualified'
    require(math.isfinite(bound) and bound >= 0, 'invalid error bound')
    return 'positive' if value > bound else 'negative' if value < -bound else 'numerical_unresolved'


def readout(full_vocab_logits, margin_error_bound=None, qualification_ref=None):
    """Both candidates must come from one already-selected prompt-only logits vector."""
    require(len(full_vocab_logits) > NO_ID, 'full vocabulary vector missing candidate')
    values = [float(v) for v in full_vocab_logits]
    require(all(math.isfinite(v) for v in values), 'nonfinite vocabulary logits')
    require(margin_error_bound is None or qualification_ref is not None, 'bound requires qualification receipt')
    yes, no = values[YES_ID], values[NO_ID]
    margin = no - yes
    require(math.isfinite(margin), 'nonfinite margin')
    normalizer = logsumexp(values)
    log_mass = logsumexp([yes, no]) - normalizer
    if margin >= 0:
        pair_no = 1.0 / (1.0 + math.exp(-margin))
    else:
        small = math.exp(margin)
        pair_no = small / (1.0 + small)
    state = resolution(margin, margin_error_bound)
    return {'z_yes': yes, 'z_no': no, 'log_p_yes': yes - normalizer, 'log_p_no': no - normalizer,
            'm': margin, 'pair_support_no': pair_no,
            'log_legal_mass': log_mass, 'legal_mass': math.exp(log_mass),
            'raw_prediction': '无' if margin > 0 else '有' if margin < 0 else None,
            'exact_tie': margin == 0, 'margin_error_bound': margin_error_bound,
            'resolution': {'positive': 'resolved_no', 'negative': 'resolved_yes'}.get(state, state),
            'qualification_ref': qualification_ref}


def linear_effect(terms, condition_scores):
    """Coalesce by physical score ID before propagating a non-statistical error band."""
    coefficients = defaultdict(float)
    physical = {}
    for term in terms:
        coefficient = float(term['coefficient'])
        require(math.isfinite(coefficient), 'nonfinite coefficient')
        score = condition_scores[term['condition_id']]
        require(math.isfinite(score['m']), 'nonfinite input margin')
        bound = score['margin_error_bound']
        require(bound is None or (math.isfinite(bound) and bound >= 0), 'invalid input bound')
        key = score['physical_score_id']
        require(isinstance(key, str) and bool(key), 'physical score identity required')
        identity = (score['prompt_sha256'], score['m'], bound, score['qualification_ref'])
        require(key not in physical or physical[key] == identity, 'conflicting aliases of one physical score')
        physical[key] = identity
        coefficients[key] += coefficient
    coefficients = {key: value for key, value in coefficients.items() if value != 0}
    value = math.fsum(coefficient * physical[key][1] for key, coefficient in coefficients.items())
    qualified = all(physical[key][2] is not None and physical[key][3] is not None for key in coefficients)
    bound = math.fsum(abs(coefficient) * physical[key][2] for key, coefficient in coefficients.items()) if qualified else None
    return {'value': value, 'bound': bound, 'resolution': resolution(value, bound),
            'physical_terms': [{'physical_score_id': key, 'coefficient': coefficient}
                               for key, coefficient in sorted(coefficients.items())]}


def reference_status(score, reference):
    """Analysis-only helper; callers must not expose references to model scoring."""
    require(reference in ['有', '无'], 'reference must already be resolved')
    raw_correct = score['raw_prediction'] == reference
    if score['resolution'] == 'unqualified':
        conservative = None
    else:
        conservative = raw_correct and score['resolution'] in ['resolved_no', 'resolved_yes']
    return {'raw_correct': raw_correct, 'conservative_correct': conservative,
            'reference_aligned_margin': (1 if reference == '无' else -1) * score['m']}


def transition(control, treatment, reference):
    c, t = reference_status(control, reference), reference_status(treatment, reference)
    resolved = ['resolved_no', 'resolved_yes']
    if control['resolution'] not in resolved or treatment['resolution'] not in resolved:
        return 'unresolved_transition'
    if not c['raw_correct'] and t['raw_correct']:
        return 'repair'
    if c['raw_correct'] and not t['raw_correct']:
        return 'damage'
    return 'stable_correct' if c['raw_correct'] else 'stable_wrong'
