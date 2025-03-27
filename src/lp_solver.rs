use nalgebra::DVector;
use microlp::{Problem, OptimizationDirection, ComparisonOp};
use rand_distr::num_traits::Zero;

use crate::datasets::SamplePoint;

pub fn solve_linear(samples: &Vec<SamplePoint>, labels_pos: &mut DVector<f64>, weights: &DVector<f64>, linear_inequalities: &Vec<(DVector<f64>, f64)>, linear_equalities: &Vec<(DVector<f64>, f64)>) {
    let mut problem = Problem::new(OptimizationDirection::Minimize);

    let mut vars = Vec::new();
    for s in samples {
        let f = s.dot(weights);

        vars.push(problem.add_var(-f, (0.0, 1.0)));
    }

    for (normal, bias) in linear_inequalities {
        let mut constraint = Vec::new();
        for i in 0..vars.len() {
            //Skip all zero entries
            if normal[i].is_zero() {
                continue;
            }

            constraint.push((vars[i], normal[i]));
        }

        problem.add_constraint(&constraint, ComparisonOp::Le, *bias);
    }
    
    for (normal, bias) in linear_equalities {
        let mut constraint = Vec::new();
        for i in 0..vars.len() {
            //Skip all zero entries
            if normal[i].is_zero() {
                continue;
            }

            constraint.push((vars[i], normal[i]));
        }

        problem.add_constraint(&constraint, ComparisonOp::Eq, *bias);
    }

    
    let solution = problem.solve().unwrap();

    // println!("Lp obj: {}", solution.objective());

    for i in 0..vars.len() {
        labels_pos[i] = solution[vars[i]];
    }
}

pub fn solve_linear_n_valued(samples: &Vec<SamplePoint>, labels_pos: &mut DVector<f64>, weights: &DVector<f64>, linear_inequalities: &Vec<(DVector<f64>, f64)>, linear_equalities: &Vec<(DVector<f64>, f64)>) {
    let mut problem = Problem::new(OptimizationDirection::Minimize);

    let mut vars = Vec::new();
    for s in samples {
        for c in 0..3 {
            let start_weight = c * weights.len() / 3;
            let end_weight = (c + 1) * weights.len() / 3;

            let weight_vec = weights.rows_range(start_weight..end_weight);
            let f = s.dot(&weight_vec);
            vars.push(problem.add_var(-f, (0.0, 1.0)));
        }
    }

    for (normal, bias) in linear_inequalities {
        let mut constraint = Vec::new();
        for i in 0..vars.len() {
            //Skip all zero entries
            if normal[i].is_zero() {
                continue;
            }

            constraint.push((vars[i], normal[i]));
        }

        problem.add_constraint(&constraint, ComparisonOp::Le, *bias);
    }
    
    for (normal, bias) in linear_equalities {
        let mut constraint = Vec::new();
        for i in 0..vars.len() {
            //Skip all zero entries
            if normal[i].is_zero() {
                continue;
            }

            constraint.push((vars[i], normal[i]));
        }

        problem.add_constraint(&constraint, ComparisonOp::Eq, *bias);
    }

    
    let solution = problem.solve().unwrap();

    // println!("Lp obj: {}", solution.objective());

    for i in 0..vars.len() {
        labels_pos[i] = solution[vars[i]];
    }
}