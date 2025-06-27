use microlp::{OptimizationDirection, Problem};
use nalgebra::DVector;

use crate::{datasets::SamplePoint, helper_functions};

//Takes the current weights and return the optimal classification
pub fn update_ilp(weights: &Vec<SamplePoint>, samples: &Vec<SamplePoint>) -> Vec<usize> {
    let class_count = weights.len();

    let mut problem = Problem::new(OptimizationDirection::Minimize);

    let mut vars = Vec::new();
    for s in samples {
        let mut class_vars = Vec::new();
        for c in 0..class_count {
            let f = s.dot(&weights[c]);
            let zero_loss = helper_functions::logistic_loss(f, 0.0);
            let one_loss = helper_functions::logistic_loss(f, 1.0);

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
    let elements_per_class = samples.len() / class_count;
    for j in 0..class_count {
        let mut constraint = Vec::new();
        for i in 0..samples.len() {
            constraint.push((vars[i][j], 1.0));
        }
        problem.add_constraint(&constraint, microlp::ComparisonOp::Eq, elements_per_class as f64);
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

pub fn update_gd(learnrate: f64, reg_term: f64, weights: &mut Vec<SamplePoint>, samples: &Vec<SamplePoint>, classes: &Vec<usize>) -> f64 {
    let class_count = weights.len();

    let mut final_grad_norm = 0.0;
    for _ in 0..1000 {
        let mut squared_length = 0.0;
        for c in 0..class_count {
            let mut grad = weights[c].clone() * reg_term;

            for i in 0..samples.len() {
                let x = &samples[i];
                let f = x.dot(&weights[c]);

                let y = if classes[i] == c { 1.0 } else { 0.0 };
                let loss_grad = helper_functions::logistic_loss_grad(f, y);

                grad += x * loss_grad;
            }

            grad *= learnrate;

            squared_length += grad.norm_squared();
            //Minimize cost
            weights[c] -= grad;

        }

        final_grad_norm = squared_length.sqrt();
    }

    return final_grad_norm;
}

pub fn evaluate_loss(
    reg_term: f64,
    samples: &Vec<SamplePoint>,
    classes: &Vec<usize>,
    weights: &Vec<SamplePoint>,
) -> f64 {
    let mut loss = 0.0;

    for i in 0..samples.len() {
        for c in 0..weights.len() {
            let x = &samples[i];
            let f = x.dot(&weights[c]);

            let y = if classes[i] == c { 1.0 } else { 0.0 };
            loss += helper_functions::logistic_loss(f, y);
        }
    }

    //Regularization term
    for w in weights {
        loss += helper_functions::regularizer_loss(w) * reg_term;
    }

    return loss;
}

pub fn evaluate_linear_loss(
    samples: &Vec<SamplePoint>,
    classes: &Vec<usize>,
    weights: &Vec<SamplePoint>,
) -> f64 {
    let mut loss = 0.0;

    for i in 0..samples.len() {
        let x = &samples[i];
        for c in 0..weights.len() {
            let f = x.dot(&weights[c]);
            loss += if c == classes[i] {
                (1.0 + f).max(0.0) 
            } else {
                (1.0 - f).max(0.0)
            };
        }
    }

    return loss;
}

pub fn solve_alternating(
    samples: &Vec<SamplePoint>,
    class_count: usize,
) -> Vec<SamplePoint>{
    let mut weights = vec![SamplePoint::zeros(samples[0].len()); class_count];
    let mut classes = update_ilp(&weights, samples); //Find initial solution for ilp
    let mut last_classes = classes.clone(); 

    let mut reg_term = 0.01; //Regularization term
    let mut learnrate = 0.01;
    let mut last_loss = f64::MAX;

    loop {
        update_gd(learnrate, reg_term, &mut weights, samples, &classes);

        //Update classes
        classes = update_ilp(&weights, samples); //Find initial solution for ilp
        
        if classes == last_classes {
            break;
        }
        
        let loss = evaluate_loss(reg_term, samples, &classes, &weights);
        
        if loss >= last_loss {
            println!("Oh no");
            reg_term *= 1.1; //Increase regularization term
            learnrate *= 0.9; //Decrease learning rate
        } else {
            reg_term *= 0.9; //Decrease regularization term
            learnrate *= 1.1; //Increase learning rate
        }
        
        last_loss = loss;
        last_classes = classes.clone();
    //    println!("Loss: {}", loss);

    }
    // println!("Loss: {}", last_loss);

    let points = samples.iter().map(|x| x.data.as_slice().to_vec()).collect::<Vec<Vec<f64>>>();
    
    println!("Linear loss: {}", evaluate_linear_loss(samples, &classes, &weights));
    let classes = classes.iter().map(|x| *x as u8).collect::<Vec<u8>>();
    let weight_vec = weights.iter().map(|x| x.data.as_slice().to_vec()).collect::<Vec<Vec<f64>>>();
    
    // generate_typst_plotter(&points, &classes, &weight_vec);

    
    return weights;
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