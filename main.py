from __future__ import annotations

from src.model.model_eval import Model

if __name__ == '__main__':
    model = Model(
        wpm_threshold=0.9,
        label_type='nonormalize',
        retrain=False,
        preference_mode='exclude',
        objective='performance',
    )
    model.evaluate()
