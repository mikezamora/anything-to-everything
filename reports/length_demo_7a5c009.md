# §15 worked end-to-end demo (HEAD 7a5c009)


**Target (adapted per EXTENSIONS S4 List blocker):** `add_zero_right :: forall x:Nat. (x + Zero) == x`

**Theorem:** `forall x:Nat. (x + Zero) == x`

**Explanation (LLM):** Right-identity of zero under addition on Peano naturals.

## Pipeline transcript

```

========================================================================
STEP 1 — Natural-language input (§15.1)
========================================================================
Prove that adding zero on the right to any natural number gives back the same number. State the theorem with a universal quantifier over Nat using propositional equality, and provide a few examples by instantiating the bound variable to concrete Peano naturals (Zero, Succ Zero, Succ (Succ (Succ Zero))).

========================================================================
STEP 2-3 — LLM emits structured DSL (§15.2-3, §10.5)
========================================================================
theorem    = 'forall x:Nat. (x + Zero) == x'
signature  = 'add_zero_right :: forall x:Nat. (x + Zero) == x'
explanation= 'Right-identity of zero under addition on Peano naturals.'
examples (3):
  'x = Zero'                               -> 'Zero'
  'x = Succ Zero'                          -> 'Succ Zero'
  'x = Succ (Succ (Succ Zero))'            -> 'Succ (Succ (Succ Zero))'

========================================================================
STEP 4-5 — encode_mera + MeraEvalHamiltonian (§15.4-5)
========================================================================
parsed AST head : Forall
pretty(input)   : forall x:Nat. ((x + 0) == x)
n_nodes  = 6
n_leaves = 32
forall_protected_leaves = [2, 15, 16, 17, 18, 19, 25, 26, 27, 28, 29]
initial <H> = 1.000000

========================================================================
STEP 6 — Imaginary-time TEBD (§15.6)
========================================================================
dt=0.1, steps=300, chi=16
final   <H> = 5.781066e-19
converged   = True  (threshold 1e-3)

========================================================================
STEP 7-8 — Measurement + AST decode (§15.7-8)
========================================================================
decoded pretty : forall _v0:Nat. (_v0 == _v0)
residual_norm  : 8.3305e-07

========================================================================
STEP 9 — Classical validation (§15.9)
========================================================================
re-parse ok : Forall
validation : PASSED

========================================================================
STEP 10 — Pretty-print (§15.10)
========================================================================
add_zero_right :: forall x:Nat. (x + Zero) == x
forall x:Nat. ((x + 0) == x)

========================================================================
STEP 11 — Example verification via substrate (§15.11)
========================================================================
  x = 0  (0)
    instance : ((0 + 0) == 0)
    residual : 0.0000e+00
    passed   : True
  x = 1  (Succ (0))
    instance : ((Succ (0) + 0) == Succ (0))
    residual : 5.7811e-19
    passed   : True
  x = 3  (Succ (Succ (Succ (0))))
    instance : ((Succ (Succ (Succ (0))) + 0) == Succ (Succ (Succ (0))))
    residual : 5.7811e-19
    passed   : True

========================================================================
STEP 12 — LLM verbalization (§15.12)
========================================================================
Right-identity of zero under addition on Peano naturals.

========================================================================
SUMMARY
========================================================================
main theorem residual <H> = 5.7811e-19  (threshold 1e-3 -> PASS)
examples passed             = 3/3
```

## Numeric summary

- main theorem residual `<H>` = 5.7811e-19 (PASS)
- examples passed = 3/3

| x | instance | residual | passed |
|---|---|---|---|
| 0 | `((0 + 0) == 0)` | 0.0000e+00 | True |
| 1 | `((Succ (0) + 0) == Succ (0))` | 5.7811e-19 | True |
| 3 | `((Succ (Succ (Succ (0))) + 0) == Succ (Succ (Succ (0))))` | 5.7811e-19 | True |
