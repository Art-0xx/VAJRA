# Threat Model

| Dimension | Specification |
|---|---|
| Attacker knowledge | White-box primary; grey-box via transfer |
| Perturbation space | Feature-space L-inf, eps in {0.01, 0.03, 0.05, 0.1} |
| Attacker goal | Evasion and induction |
| Attack families | FGSM, PGD, AutoAttack |
| Out of scope | Physical-layer, protocol-invalid, black-box queries |
