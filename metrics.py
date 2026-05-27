"""
Evaluation metrics for video captioning using pycocoevalcap
Replaces nlgeval which is hard to install
"""

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


def compute_metrics(ref_list, hyp_list):
    """
    Compute evaluation metrics for captions.
    
    Args:
        ref_list: List of reference captions (strings)
        hyp_list: List of hypothesis/predicted captions (strings)
    
    Returns:
        dict: Dictionary with metrics {BLEU_1, BLEU_2, BLEU_3, BLEU_4, METEOR, ROUGE_L, CIDEr}
    """
    
    # Convert to format expected by pycocoevalcap
    # Each reference and hypothesis needs to be in a dict with id -> list of captions
    ref_dict = {i: [caption] for i, caption in enumerate(ref_list)}
    hyp_dict = {i: [caption] for i, caption in enumerate(hyp_list)}
    
    metrics = {}
    
    try:
        # BLEU
        bleu_scorer = Bleu(4)
        bleu_scores, _ = bleu_scorer.compute_score(ref_dict, hyp_dict)
        metrics['Bleu_1'] = bleu_scores[0]
        metrics['Bleu_2'] = bleu_scores[1]
        metrics['Bleu_3'] = bleu_scores[2]
        metrics['Bleu_4'] = bleu_scores[3]
    except Exception as e:
        print(f"BLEU computation error: {e}")
        metrics['Bleu_1'] = 0.0
        metrics['Bleu_2'] = 0.0
        metrics['Bleu_3'] = 0.0
        metrics['Bleu_4'] = 0.0
    
    try:
        # ROUGE
        rouge_scorer = Rouge()
        rouge_score, _ = rouge_scorer.compute_score(ref_dict, hyp_dict)
        metrics['ROUGE_L'] = rouge_score
    except Exception as e:
        print(f"ROUGE computation error: {e}")
        metrics['ROUGE_L'] = 0.0
    
    try:
        # CIDEr
        cider_scorer = Cider()
        cider_score, _ = cider_scorer.compute_score(ref_dict, hyp_dict)
        metrics['CIDEr'] = cider_score
    except Exception as e:
        print(f"CIDEr computation error: {e}")
        metrics['CIDEr'] = 0.0
    
    return metrics