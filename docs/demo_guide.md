# Demo Guide (90 seconds)

1. Open dashboard at http://localhost:8501
2. Attack = None -> Run. Shows clean ATTACK.
3. Attack = FGSM, eps = 0.05 -> Run. Label flips (Week 4+).
4. Attack = PGD, eps = 0.10 -> Run. Stronger flip.
5. Model = TRADES, same PGD eps=0.05 -> Run. Label stays ATTACK.
