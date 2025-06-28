use microlp::{OptimizationDirection, Problem};
use nalgebra::{DVector, SimdPartialOrd};

use crate::{datasets::SamplePoint, helper_functions};

pub struct LossFunction {
    loss: fn(f64, f64) -> f64,
    loss_grad: fn(f64, f64) -> f64,
}

//Takes the current weights and return the optimal classification
pub fn update_ilp(weights: &Vec<SamplePoint>, samples: &Vec<SamplePoint>, lf: &LossFunction, min_class_size: usize) -> Vec<usize> {
    let class_count = weights.len();

    let mut problem = Problem::new(OptimizationDirection::Minimize);

    let mut vars = Vec::new();
    for s in samples {
        let mut class_vars = Vec::new();
        for c in 0..class_count {
            let f = s.dot(&weights[c]);
            let zero_loss = (lf.loss)(f, 0.0);
            let one_loss = (lf.loss)(f, 1.0);

            //(1 - y) * zero_loss + y * one_loss = zero_loss + y (one_loss - zero_loss)
            class_vars.push(problem.add_binary_var(one_loss - zero_loss));
        }
        vars.push(class_vars);
    }

    //Single class constraint
    for i in 0..samples.len() {
        let mut constraint = Vec::new();
        for j in 0..class_count {
            constraint.push((vars[i][j], 1.0));
        }
        problem.add_constraint(&constraint, microlp::ComparisonOp::Eq, 1.0);
    }

    //Even distribution constraint
    for j in 0..class_count {
        let mut constraint = Vec::new();
        for i in 0..samples.len() {
            constraint.push((vars[i][j], 1.0));
        }
        problem.add_constraint(&constraint, microlp::ComparisonOp::Ge, min_class_size as f64);
    }

    
    let solution = problem.solve().unwrap();
    let mut res = vec![0; samples.len()];

    for i in 0..samples.len() {
        for j in 0..class_count {
            if solution[vars[i][j]] < 0.5 {
                continue; //Skip all zero entries
            }

            res[i] = j;
        }
    }

    return res;
}

pub fn update_lp(weights: &mut Vec<SamplePoint>, samples: &Vec<SamplePoint>, classes: &Vec<usize>, a: f64) -> f64 {
    let mut problem = Problem::new(OptimizationDirection::Minimize);
    
    let class_count = weights.len();
    let dim_count = samples[0].len();

    
    let mut weight_vars = Vec::new();
    for c in 0..class_count {
        let mut cweight_vars = Vec::new();
        for d in 0..dim_count {
            cweight_vars.push(problem.add_var(0.0, (-1.0, 1.0)));
        }
        weight_vars.push(cweight_vars);        
    }
    
    let mut cost_vars = Vec::new();
    for s in 0..samples.len() {
        let mut cvars = Vec::new();
        for c in 0..class_count {
            cvars.push(problem.add_var(1.0, (0.0, f64::INFINITY)));
        }
        cost_vars.push(cvars);
    }

    //Relu constraints
    for s in 0..samples.len() {
        for c in 0..class_count {
            let mut constraint = Vec::new();
            let sign = if classes[s] == c { 1.0 } else { -1.0 };
            for d in 0..dim_count {
                constraint.push((weight_vars[c][d], samples[s][d] * sign));
            }

            constraint.push((cost_vars[s][c], -1.0)); //Add the z term
            
            problem.add_constraint(&constraint, microlp::ComparisonOp::Le, -a);
        }
    }
    
    let solution = problem.solve().unwrap();
    
    for c in 0..class_count {
        for d in 0..dim_count {
            weights[c][d] = solution[weight_vars[c][d]];   
        }
    }
    
    return solution.objective();
}

pub fn update_lp_mu(weights: &mut Vec<SamplePoint>, samples: &Vec<SamplePoint>, classes: &Vec<usize>, a: f64) -> f64 {
    let mut problem = Problem::new(OptimizationDirection::Minimize);

    let class_count = weights.len();
    let dim_count = samples[0].len();

    
    let mut weight_vars = Vec::new();
    for c in 0..class_count {
        let mut cweight_vars = Vec::new();
        for d in 0..dim_count {
            cweight_vars.push(problem.add_var(0.0, (-1.0, 1.0)));
        }
        weight_vars.push(cweight_vars);        
    }
    
    let mut mu_pluss = Vec::new();
    let mut mu_minus = Vec::new();
    for s in 0..samples.len() {
        let mut cmu_pluss = Vec::new();
        let mut cmu_minus = Vec::new();

        for c in 0..class_count {
            cmu_pluss.push(problem.add_var(1.0, (0.0, f64::INFINITY)));
            cmu_minus.push(problem.add_var(0.0, (0.0, f64::INFINITY)));
        }

        mu_pluss.push(cmu_pluss);
        mu_minus.push(cmu_minus);   
    }

    //Relu constraints
    for s in 0..samples.len() {
        for c in 0..class_count {
            let mut constraint = Vec::new();
            let sign = if classes[s] == c { 1.0 } else { -1.0 };
            for d in 0..dim_count {
                constraint.push((weight_vars[c][d], samples[s][d] * sign));
            }

            constraint.push((mu_pluss[s][c], -1.0)); 
            constraint.push((mu_minus[s][c], 1.0)); 
            
            problem.add_constraint(&constraint, microlp::ComparisonOp::Eq, -a);
        }
    }
    
    let res =problem.solve();
    if res.is_err() {
        println!("Error solving LP: {:?}", res.err());
        return f64::MAX; //Return a large value to indicate failure
    } 
    
    let solution = res.unwrap();
    for c in 0..class_count {
        for d in 0..dim_count {
            weights[c][d] = solution[weight_vars[c][d]];   
        }
    }
    
    return solution.objective();
}

pub fn update_gd(learnrate: f64, weights: &mut Vec<SamplePoint>, samples: &Vec<SamplePoint>, classes: &Vec<usize>, lf: &LossFunction) -> f64 {
    let class_count = weights.len();

    let mut final_grad_norm = 0.0;
    for _ in 0..1000 {
        let mut squared_length = 0.0;
        for c in 0..class_count {
            let mut grad = SamplePoint::zeros(weights[c].len());

            for i in 0..samples.len() {
                let x = &samples[i];
                let f = x.dot(&weights[c]);

                let y = if classes[i] == c { 1.0 } else { 0.0 };
                let loss_grad = (lf.loss_grad)(f, y);

                grad += x * loss_grad;
            }

            grad *= learnrate;

            squared_length += grad.norm_squared();       
            
            weights[c] -= grad;

            for v in weights[c].data.as_mut_slice() {
                *v = v.clamp(-1.0, 1.0);
            }
        }

        final_grad_norm = squared_length.sqrt();
    }

    return final_grad_norm;
}

pub fn evaluate_loss(
    samples: &Vec<SamplePoint>,
    classes: &Vec<usize>,
    weights: &Vec<SamplePoint>,
    lf: &LossFunction,
) -> f64 {
    let mut loss = 0.0;

    for i in 0..samples.len() {
        for c in 0..weights.len() {
            let x = &samples[i];
            let f = x.dot(&weights[c]);

            let y = if classes[i] == c { 1.0 } else { 0.0 };
            loss += (lf.loss)(f, y);
        }
    }

    return loss;
}

pub fn solve_alternating(
    samples: &Vec<SamplePoint>,
    class_count: usize,
    min_class_size: usize,
) -> f64 {
    let mut weights = vec![SamplePoint::zeros(samples[0].len()); class_count];

    for v in weights.iter_mut() {
        for v in v.data.as_mut_slice() {
            *v = rand::random::<f64>() * 2.0 - 1.0; //Initialize weights randomly
        }
    }

    let logistic_loss = LossFunction {
        loss: helper_functions::logistic_loss,
        loss_grad: helper_functions::logistic_loss_grad,
    };

    let linear_loss = LossFunction {
        loss: helper_functions::linear_loss,
        loss_grad: helper_functions::linear_loss_grad,
    };

    
    // optimize_alternating(samples, min_class_size, &mut weights, &logistic_loss);

    optimize_alternating(samples, min_class_size, &mut weights, &linear_loss);

    let classes = update_ilp(&weights, samples, &linear_loss, min_class_size); 
    let loss = evaluate_loss(samples, &classes, &weights, &linear_loss);

    return loss;
}

pub fn optimize_alternating(
    samples: &Vec<SamplePoint>,
    min_class_size: usize,
    weights: &mut Vec<SamplePoint>,
    lf: &LossFunction) -> f64 {
    
    let mut classes = update_ilp(&weights, samples, lf, min_class_size); //Find initial solution for ilp
    let mut last_classes = classes.clone(); 

    let mut last_loss = f64::MAX;

    loop {
        // update_gd(learnrate, weights, samples, &classes, lf);
        // let loss = evaluate_loss(samples, &classes, &weights, lf);
        // println!("Before GD Loss: {:.5}", loss);
        // let lp_loss = update_lp(weights, samples, &classes, 1.0);
        let lp_loss = update_lp_mu(weights, samples, &classes, 1.0);

        // println!("Lp loss: {:.5} before Loss: {:.5} after loss: {:.5}", lp_loss, loss, evaluate_loss(samples, &classes, weights, lf));

        // println!("After LP Loss: {:.5}", loss);

        //Update classes
        classes = update_ilp(&weights, samples, lf, min_class_size); //Find initial solution for ilp
        
        
        let loss = evaluate_loss(samples, &classes, &weights, lf);
        
        if classes == last_classes || loss >= last_loss {
            return loss; //Convergence reached
        }
                
        // println!("Loss: {:.5}", loss);
        last_loss = loss;
        last_classes = classes.clone();
    }
}

pub fn generate_typst_plotter(points: &[Vec<f64>], classes: &[u8], weights: &[Vec<f64>]) {
    assert_eq!(points.len(), classes.len());

    // for w in weights{
    //     println!("weights: {:?}", w);
    // }

    println!("#let points = (");
    for (i, point) in points.iter().enumerate() {
        println!("\t(values: ({}), class: {}),", point.iter().skip(1).map(|x| format!("{:.5}", x)).collect::<Vec<String>>().join(", "), classes[i]);
    }
    println!(")\n");

    println!("#let planes = (");
    for (i, weight) in weights.iter().enumerate() {
        if weight.len() < 2 {
            continue;
        }

        let mut normal = weight.iter().skip(1).map(|&x| x).collect::<Vec<f64>>();
        let length = normal.iter().map(|&x| x * x).sum::<f64>().sqrt();
        
        normal.iter_mut().for_each(|x| *x /= length);
        let bias = weight[0] / length;

        println!("\t(normal: ({}), bias: {:0.5}, class: {}),", normal.iter().map(|x| format!("{}", x)).collect::<Vec<_>>().join(", "), bias, i);
    }
    println!(")\n");
}