# Free-Text Review Template

Use this template to assign local proxy scores for `free_text` answers.

## Inputs

- `question_id`:
- Question:
- Submission answer:
- Reference answer:
- Retrieved pages:

## Rubric

Score each criterion as `0` or `1`.

1. Correctness:
2. Completeness:
3. Grounding:
4. Confidence calibration:
5. Clarity and relevance:

## Result

- `assistant_score`: `(sum(criteria) / 5)`
- Notes:

## Export Format

```json
{
  "assistant_scores": [
    {
      "question_id": "example-question-id",
      "assistant_score": 0.8,
      "notes": "One grounding miss, otherwise acceptable."
    }
  ]
}
```
