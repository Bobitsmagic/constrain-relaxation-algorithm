#let dotp(a, b) = $angle.l #a, #b angle.r$

= Formulas
$
  &"set of classes:" &C \
  &"set of samples:" &S \
  &"feature vectors:" &x_s in RR^n\
  &"weights:" &theta_c in RR^n\
  &"labels:" &y_(s, c) in {0, 1} \
  &"loss function:" &L(x, y) = - x y + ln(1 + e^x) \
  &"regularizer:" &R(theta_c) = lambda/2 ||theta_c||^2 \
  &"goal:" &min sum_(c in C) R(theta_c) + sum_(s in S) L(dotp(theta_c, x_s), y_(s, c)) \
  &"constraints:" &forall s in S: sum_(c in C) y_(s, c) = 1 \
  &&forall c in C: sum_(s in S) y_(s, c) = abs(S) / abs(C)
$

== ILP
$
  &"given" theta_c "and" x_s \
  &min sum_(c in C) R(theta_c) + sum_(s in S) L(dotp(theta_c, x_s), y_(s, c)) \
  &min sum_(c in C) sum_(s in S) - dotp(theta_c, x_s) y_(s,c)\
$

== GD
$
  &"given" y_(s, c) "and" x_s \
  &forall c in C: gradient_theta_c (R(theta_c) + sum_(s in S) L(dotp(theta_c, x_s), y_(s, c))) = lambda theta_c + sum_(s in S) x_s (-y_(s,c) + 1 / (e^(-dotp(theta_c, x_s)) + 1)) \
  \
$
