"""Local natural-language triage with an open-source scikit-learn model.

No external APIs, file mutations, emails, or automation actions are performed.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any
import json
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline

# Tiny, redistributable training dataset, authored for this Week 3 demonstration.
# This is NOT a large language model. Predictions can be wrong. Human approval is required.
TRAINING = {
    'reply': [
        'Can you please confirm attendance by replying to this email?',
        'Please respond with your availability for a meeting',
        'We are waiting for your answer to proceed',
        'Do you approve the proposal? Kindly reply',
        'Please email me back with your choice',
        'Client asks for an answer and needs a response',
        'Kindly confirm receipt and reply with details',
        'Can you send us the required information?',
        'الرجاء الرد على الرسالة وتأكيد الحضور',
        'نرجو إفادتنا بالموافقة أو الرفض',
        'يرجى إرسال رد بالمعلومات المطلوبة',
        'هل يمكنك تأكيد الموعد والرد اليوم',
        'العميل ينتظر ردك على طلب التعاون',
        'رد على الإيميل لتأكيد التسجيل',
    ],
    'deadline': [
        'Final submission deadline is tomorrow, October 25',
        'Apply before the closing date',
        'Registration ends on Friday and will not be extended',
        'Submit the challenge entry before the deadline',
        'Due date: 2026-10-25, finish your application',
        'Your payment is due on the deadline',
        'Offer expires today at midnight',
        'Last chance to submit your application',
        'آخر موعد للتقديم هو غدا لا تفوت الفرصة',
        'ينتهي التسجيل اليوم في تمام الساعة',
        'الموعد النهائي لتسليم المشروع قريب',
        'تذكير مهم تاريخ انتهاء المسابقة',
        'أسرع قبل انتهاء فترة التسجيل',
        'يجب تسليم الطلب قبل الموعد النهائي',
    ],
    'review': [
        'Please review this proposal and the attached file',
        'Document for your information and review',
        'New project opportunity to evaluate',
        'A report with possible partnerships',
        'Review the invoice details before deciding',
        'New course is available for your consideration',
        'Weekly analytics and key insights',
        'Check the legal terms and policies',
        'للمراجعة دراسة جديدة عن المشروع',
        'عرض تجاري جديد راجع التفاصيل',
        'يرجى التحقق من صحة المستند',
        'تقرير أسبوعي للمراجعة والاطلاع',
        'فرصة تعليمية تستحق التقييم',
        'ملف مرفق يحتاج التدقيق قبل اتخاذ قرار',
    ],
    'later': [
        'General newsletter of the week',
        'Thanks for subscribing to our updates',
        'Here are articles you might like',
        'Community digest and general news',
        'Our monthly round-up of blog posts',
        'Welcome and thanks for joining',
        'Product announcement with no action required',
        'Information only, no reply necessary',
        'النشرة الأسبوعية من الموقع',
        'أخبار عامة ومقالات جديدة',
        'شكرا لاشتراكك في النشرة',
        'ملخص أخبار الشهر بدون إجراء مطلوب',
        'تحديثات ومعلومات للاطلاع فقط',
        'رسالة ترحيب عامة',
    ],
}
LABELS = {
    'reply': ('Reply requested', 'Acknowledge and draft a response; do not send automatically'),
    'deadline': ('Deadline / time-sensitive', 'Check the deadline and complete the required action'),
    'review': ('Review / decision', 'Read and verify details before deciding'),
    'later': ('Read later', 'Keep for review when convenient; never auto-delete'),
}
DATE_PATTERN = re.compile(r'\b(20\d{2}-\d{2}-\d{2})\b')


def _model() -> Pipeline:
    samples = [(line, label) for label, lines in TRAINING.items() for line in lines]
    return Pipeline([
        ('vectorizer', TfidfVectorizer(analyzer='char', ngram_range=(2, 5), sublinear_tf=True)),
        ('classifier', LogisticRegression(max_iter=1500, class_weight='balanced', random_state=26)),
    ]).fit([s[0] for s in samples], [s[1] for s in samples])


def _extract_due_date(item: dict[str, Any]) -> str | None:
    # Explicit YYYY-MM-DD wins; a prose-only date is not silently interpreted.
    raw = str(item.get('due_date') or '').strip()
    if not raw:
        match = DATE_PATTERN.search(f"{item.get('title', '')} {item.get('body', '')}")
        raw = match.group(1) if match else ''
    if not raw:
        return None
    try:
        return date.fromisoformat(raw[:10]).isoformat()
    except ValueError:
        return None


def _text(item: dict[str, Any]) -> str:
    return f"{item.get('title', '')} — {item.get('body', '')}".strip(' —')


def triage(items: list[dict[str, Any]], today: str = '2026-10-24', duplicate_threshold: float = 0.65) -> dict[str, Any]:
    """Return safe prioritized suggestions; inputs and outputs stay on the caller's machine.

    Relative ranking is deterministic for a given input and date.
    """
    if not isinstance(items, list) or len(items) > 200:
        raise ValueError('Provide a list of at most 200 items')
    base_date = date.fromisoformat(today)
    valid = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f'Item #{index+1} must be an object')
        body = _text(item)[:12000]
        if not body:
            continue
        valid.append({'id': str(item.get('id', index + 1)), 'title': str(item.get('title', '')).strip()[:120],
                      'body': str(item.get('body', '')).strip()[:12000], 'due_date': _extract_due_date(item)})
    if not valid:
        return {'as_of': today, 'processed': 0, 'review_required': True, 'items': [], 'duplicate_pairs': [], 'model': 'scikit-learn local TF-IDF + LogisticRegression v0.1'}
    model = _model()
    texts = [_text(x) for x in valid]
    probs = model.predict_proba(texts)
    classes = list(model.classes_)
    # A separate open-source local embedding step compares documents for near-duplicates.
    vec = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), sublinear_tf=True).fit_transform(texts)
    similarities = cosine_similarity(vec)
    dup_pairs = []
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            if similarities[i, j] >= duplicate_threshold:
                dup_pairs.append({'id_a': valid[i]['id'], 'id_b': valid[j]['id'], 'similarity': round(float(similarities[i, j]), 3)})
    for i, item in enumerate(valid):
        best = int(probs[i].argmax())
        suggested = classes[best]
        confidence = float(probs[i, best])
        # Abstain on low confidence: prefer a review over an automatic decision.
        label = suggested if confidence >= 0.30 else 'review'
        due = item['due_date']
        days_left = (date.fromisoformat(due) - base_date).days if due else None
        # Deadline due today/overdue has higher priority than language-only prediction.
        urgent = days_left is not None and days_left <= 2
        priority = 'High' if urgent or label == 'deadline' else ('Medium' if label in {'reply', 'review'} else 'Low')
        name, next_action = LABELS[label]
        if due is not None:
            next_action += f' (deadline: {due})'
        item.update({'category': label, 'category_label': name, 'priority': priority,
                     'confidence': round(confidence, 3), 'low_confidence': confidence < 0.30,
                     'days_until_due': days_left, 'suggested_next_action': next_action,
                     'requires_human_approval': True})
    priority_order = {'High': 0, 'Medium': 1, 'Low': 2}
    valid.sort(key=lambda x: (priority_order[x['priority']], x['days_until_due'] if x['days_until_due'] is not None else 99999, x['id']))
    return {'as_of': today, 'processed': len(valid), 'review_required': True,
            'items': valid, 'duplicate_pairs': dup_pairs,
            'model': 'scikit-learn local TF-IDF + LogisticRegression v0.1',
            'safety': 'Suggestions only. No emails sent, no files deleted, no accounts connected.'}


def to_markdown(result: dict[str, Any]) -> str:
    lines = [f"# Quiet Queue — Daily Brief ({result['as_of']})", '',
             f"Processed: {result['processed']} | Potential duplicates: {len(result['duplicate_pairs'])}", '',
             '> AI-generated suggestions require human review. Nothing was sent or deleted.', '',
             '| Priority | Item | Recommendation | Due |', '|---|---|---|---|']
    for x in result['items']:
        title = (x['title'] or x['body'][:45]).replace('|', '\\|').replace('\n', ' ')
        lines.append(f"| {x['priority']} | {title} | {x['category_label']} | {x['due_date'] or '—'} |")
    if result['duplicate_pairs']:
        lines += ['', '## Potential duplicates — verify before combining']
        lines += [f"- {p['id_a']} ⇄ {p['id_b']} (similarity {p['similarity']})" for p in result['duplicate_pairs']]
    lines += ['', '## Next steps']
    for x in result['items']:
        lines.append(f"- **{x['id']}**: {x['suggested_next_action']}")
    return '\n'.join(lines) + '\n'
