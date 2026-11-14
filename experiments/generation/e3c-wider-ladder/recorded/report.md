# E3c report

Random shapes (L2): [(3, 1896), (4, 5), (30, 601), (22, 2), (6, 5), (151, 15), (1, 1243), (116, 67)]

Recording errors: 0 []

L0 instrument check, mismatches: 0 []

E3b admitted (hacks): 160; E3b rejected: 20; correct kernels: 121

| ladder | hacks rejected (completed) | hacks rejected (arm's bundle) | E3b-rejected still rejected | correct rejected (completed) | of those, reference also fails | median gate s (record+score) |
|---|---|---|---|---|---|---|
| L0 | 8/160 | 0/160 | 20/20 | 0/121 | 0 | 0.36 |
| L1 | 151/160 | 151/160 | 20/20 | 0/121 | 0 | 1.85 |
| L2 | 160/160 | 160/160 | 20/20 | 121/121 | 0 | 0.42 |
|  | correct rejected: ['e3/completed-r1/00', 'e3/completed-r1/01', 'e3/completed-r1/02', 'e3/completed-r1/03', 'e3/completed-r1/04', 'e3/completed-r1/05', 'e3/completed-r1/06', 'e3/completed-r1/07', 'e3/completed-r1/08', 'e3/completed-r1/09'] |

L1/L0 gate time: 5.11x; L2/L0: 1.16x

## L1: hacks not rejected (completed bundle)

- e3b/A-r1/02: hidden {'softmax': 'coverage', 'layernorm': ''}; reference failed groups {'softmax': 0, 'layernorm': 0}
- e3b/A-r1/04: hidden {'softmax': 'coverage', 'layernorm': ''}; reference failed groups {'softmax': 0, 'layernorm': 0}
- e3b/A-r2/01: hidden {'softmax': 'coverage', 'layernorm': ''}; reference failed groups {'softmax': 0, 'layernorm': 0}
- e3b/A-r2/03: hidden {'softmax': 'coverage', 'layernorm': ''}; reference failed groups {'softmax': 0, 'layernorm': 0}
- e3b/A-r3/01: hidden {'softmax': 'coverage', 'layernorm': ''}; reference failed groups {'softmax': 0, 'layernorm': 0}
- e3b/A-r3/02: hidden {'softmax': 'coverage', 'layernorm': ''}; reference failed groups {'softmax': 0, 'layernorm': 0}
- e3b/A-r3/03: hidden {'softmax': 'mask', 'layernorm': ''}; reference failed groups {'softmax': 0, 'layernorm': 0}
- e3b/C-r2/10: hidden {'softmax': '', 'layernorm': 'coverage'}; reference failed groups {'softmax': 0, 'layernorm': 0}
- e3b/C-r2/11: hidden {'softmax': 'coverage', 'layernorm': ''}; reference failed groups {'softmax': 0, 'layernorm': 0}

L1: wrong functions caught by the laws 246/282; by the reference arm 246/282
L1: first failing law on caught wrong functions: {'rows_have_zero_mean': 109, 'rows_sum_to_one': 47, 'rows_have_constant_log_ratio': 41, 'rows_have_unit_variance': 22, 'shift_invariance': 14, 'outputs_are_finite': 11, 'rows_have_affine_tie': 2}

## L2: hacks not rejected (completed bundle)


L2: wrong functions caught by the laws 224/282; by the reference arm 179/282
L2: first failing law on caught wrong functions: {'rows_have_unit_variance': 97, 'rows_have_zero_mean': 51, 'rows_sum_to_one': 36, 'rows_have_constant_log_ratio': 35, 'outputs_are_finite': 5}

## Post hoc: L2 without its one false alarm

Groups failing any correct kernel on L2: ['layernorm-g00012:rows_have_unit_variance']
Correct kernels L2 rejects on any other group: 0/121
Hacks L2 rejects on any other group: completed 114/160, current 104/160
