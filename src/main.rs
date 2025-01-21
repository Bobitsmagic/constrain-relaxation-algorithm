use core::panic;
use std::env;
use datasets::SamplePoint;
use helper_functions::{evaluate_grad, evaluate_loss, logistic_loss, logistic_loss_grad, regularizer_loss, regularizer_loss_grad};
use nalgebra::{one, DVector};
use rand::{prelude::Distribution, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::Normal;

mod datasets;
mod lp_solver;
mod helper_functions;


fn main() {
    //enable backtrace
    env::set_var("RUST_BACKTRACE", "1");
    
    test_basic_example();
}

fn test_basic_example() {
    let mut rng = ChaCha8Rng::seed_from_u64(3);

    let samples = datasets::iris_data();

    let (mut labels, mut weights) = datasets::gen_parameter(&samples);
    
    let label_count = labels.len();

    let learnrate = 0.01; 
    let lambda = 0.01;   //Regularization parameter

    let it_count = 1000;
    let mut correct = 0;

    for _ in 0..it_count {
        randomize_weights(&mut weights, 1.0, &mut rng);
        randomize_lables(&mut labels, &mut rng);    

        let zero_var = rng.gen_range(0..50);
        let one_var = rng.gen_range(50..100);
 
        //Constraints for just 2 labels
        let mut linear_equalities = Vec::new();
        linear_equalities.push(set_label_value_zero(2, 0));
        linear_equalities.push(set_label_value_one(2, 1));
                
        let small_samples = vec![samples[zero_var].clone(), samples[one_var].clone()];
        let mut small_labels = DVector::from_vec(vec![0.0, 1.0]);
        
        gradient_descent(&small_samples, &mut small_labels, &mut weights, &Vec::new(), &linear_equalities, learnrate, lambda);
        
        //Linear constraints for all labels
        linear_equalities.clear();
        linear_equalities.push(set_label_value_zero(label_count, zero_var));
        linear_equalities.push(set_label_value_one(label_count, one_var));
        
        labels[zero_var] = 0.0;
        labels[one_var] = 1.0;
        
        println!("Initial weights:");
        print_vector(&weights);
        println!("Loss: {}", evaluate_loss(&samples, &labels, &weights, lambda));
        // println!("Constraints hit: {}", all_constraints_hit(&labels, &Vec::new(), &linear_equalities));
        
        let mut linear_inequalities = is_label_inequalities(label_count);
        linear_inequalities.remove(one_var * 2 + 1);
        linear_inequalities.remove(one_var * 2 + 0);
        linear_inequalities.remove(zero_var * 2 + 1);
        linear_inequalities.remove(zero_var * 2 + 0);
        
        lp_solver::solve_linear(&samples, &mut labels, &mut weights, &linear_inequalities, &linear_equalities);

        println!("Initial labels:");
        // print_vector(&labels);
        println!("Loss: {}", evaluate_loss(&samples, &labels, &weights, lambda));

        println!("Correct count: {}", evaluate_iris_labels(&mut labels));

        // println!("Constraints hit: {}", all_constraints_hit(&labels, &linear_inequalities, &linear_equalities));

        //descent(&samples, &mut labels, &mut weights, learnrate, lambda, &linear_inequalities, &mut rng);
        gradient_descent(&samples, &mut labels, &mut weights, &linear_inequalities, &linear_equalities, learnrate, lambda);

        let correct_count = evaluate_iris_labels(&mut labels);
        correct += (correct_count == 100) as i32;
        println!("Loss: {}", evaluate_loss(&samples, &labels, &weights, lambda));
        println!("Correct count: {}", correct_count);
        
        if evaluate_loss(&samples, &labels, &weights, lambda).abs() > 100.0 {
            println!("Final weights:");
            print_vector(&weights);
            println!("Final labels:");
            print_vector(&labels);
            println!("Constraints hit: {}", all_constraints_hit(&labels, &linear_inequalities, &linear_equalities));

            panic!();
        }

        // println!("Correct count: {}", correct_count);

        // if correct_count >= 95 {
        //     print_vector(&labels);
        // }
    }

    println!("Correct: {}/{} = {:.2}%", correct, it_count, correct as f64 / it_count as f64 * 100.0);
}

fn evaluate_iris_labels(labels: &mut DVector<f64>) -> i32 {
    let mut correct_count = 0;
    let mut int_count = 0;

    for i in 0..labels.len() {
        if labels[i] < 0.5 {
            correct_count += (i < 50) as i32;
        } else {
            correct_count += (i >= 50) as i32;
        }

        if (labels[i] - 0.5).abs() > 0.499 {
            int_count += 1;
        }
    }

    println!("Int count: {}", int_count);

    correct_count
}

//Randomizes labels to be uniformly distributed in [0, 1]
fn randomize_lables(v: &mut DVector<f64>, rng: &mut ChaCha8Rng) {
    for i in 0..v.len() {
        v[i] = rng.gen_range(0.0..1.0);
    }
}

//Randomizes weights to be normally distributed
fn randomize_weights(v: &mut DVector<f64>, deviation: f64, rng: &mut ChaCha8Rng) {
    let normal = Normal::new(0.0, deviation).unwrap();
    for i in 0..v.len() {
        v[i] = normal.sample(rng);
    }
}

//assures that labels are in range [0, 1]
fn is_label_inequalities(label_count: usize) -> Vec<(DVector<f64>, f64)> {
    let mut linear_inequalities = Vec::new();

    //Add constraints for labels to be in range [0, 1]
    //They have the form: w_1 * x_1 + ... + w_N * x_N + w_(N + 1) * l_1 + ... + w_(N + L) <= Bias

    for i in 0..label_count {
        let mut v = DVector::zeros(label_count);
        v[i] = 1.0;
        linear_inequalities.push((v.clone(), 1.0)); //l_i <= 1

        v[i] = -1.0;
        linear_inequalities.push((v, 0.0)); //-l_i <= 0
    }

    linear_inequalities
}

//assures that a label is between 0 and EPSILON
fn set_label_value_zero(label_count: usize, index: usize) -> (DVector<f64>, f64) {
    let mut v = DVector::zeros(label_count);
    v[index] = 1.0;
    return (v, 0.0); //l_i == 0
}

//assures that a label is between 0 and EPSILON
fn set_label_value_one(label_count: usize, index: usize) -> (DVector<f64>, f64) {
    let mut v = DVector::zeros(label_count);
    v[index] = 1.0;
    return (v, 1.0); //l_i == 1
}

fn gradient_descent(samples: &Vec<SamplePoint>, labels_pos: &mut DVector<f64>, weights: &mut DVector<f64>, linear_inequalities: &Vec<(DVector<f64>, f64)>, linear_equalities: &Vec<(DVector<f64>, f64)>, learnrate: f64, lambda: f64) {
    for _ in 0..1000 {
        let (weight_grad, label_grad) = evaluate_grad(&samples, &labels_pos, &weights, lambda);

        //Invert direction of gradient
        let label_step = -learnrate * label_grad;
        let eq_projected_step = project_search_direction_eq(&labels_pos, &label_step, &linear_equalities);
        let projected_step = project_search_direction_ineq(&labels_pos, &eq_projected_step, &linear_inequalities);

        let clipped_step = clip_to_constraint(&labels_pos, &projected_step, &linear_inequalities);
        if label_step.norm_squared() < clipped_step.norm_squared() {
            println!("Overshoot");
            panic!();
        }

        //Update weights and labels
        *weights -= learnrate * weight_grad;
        *labels_pos += clipped_step;
    }
}

fn all_constraints_hit(pos: &DVector<f64>, linear_inequalities: &Vec<(DVector<f64>, f64)>, linear_equalities: &Vec<(DVector<f64>, f64)>) -> bool {
    for (normal, bias) in linear_inequalities.iter().take(24) {
        if normal.dot(&pos) > *bias {
            return false;
        }
    }

    for (normal, bias) in linear_equalities {
        if (normal.dot(&pos) - *bias).abs() > 0.00001 {
            return false;
        }
    }

    return true;
}

//Walk as far as possible until a constraint is hit
fn clip_to_constraint(pos: &DVector<f64>, step: &DVector<f64>, linear_inequalities: &Vec<(DVector<f64>, f64)>) -> DVector<f64> {
    let mut min_alpha = 1.0;

    //Find first contraint that is hit
    for (normal, bias) in linear_inequalities {
        let alpha = (bias - normal.dot(&pos)) / normal.dot(&step);
        if alpha < min_alpha && step.dot(&normal) > 0.001 && alpha >= 0.0 {
            min_alpha = alpha;
        }
    }

    step * min_alpha

}

fn project_search_direction_ineq(pos: &DVector<f64>, step: &DVector<f64>, linear_inequalities: &Vec<(DVector<f64>, f64)>) -> DVector<f64> {
    let mut new_step = step.clone();
    
    //project step onto plane of inequalities
    for (normal, bias) in linear_inequalities {
        //pos is on inequality and inequality is hit or violated 
        if (normal.dot(pos) - *bias).abs() < 0.00001 && normal.dot(&new_step) > 0.0 {
            //project pos onto plane if inequality is violated
            let pos_proj = pos - (normal.dot(pos) - bias) * normal;
            //project step onto plane
            let step_proj = new_step.clone() - (normal.dot(&new_step)) * normal;

            //to correct overshoot of inequalities
            new_step = (pos_proj + step_proj) - pos;
        }   
    }

    if new_step.norm() > step.norm() * 1.0001 {
        println!("Overshoot ineq {}", new_step.norm() / step.norm());
    }

    new_step
}

fn project_search_direction_eq(pos: &DVector<f64>, step: &DVector<f64>, linear_equalities: &Vec<(DVector<f64>, f64)>) -> DVector<f64> {
    let mut new_step = step.clone();
    
    //project step onto plane of inequalities
    for (normal, bias) in linear_equalities {        
        //project pos onto plane if inequality is violated
        let pos_proj = pos - (normal.dot(pos) - bias) * normal;
        //project step onto plane
        let step_proj = new_step.clone() - (normal.dot(&new_step)) * normal;

        //to correct overshoot of inequalities
        new_step = (pos_proj + step_proj) - pos;
    }

    if new_step.norm() > step.norm() * 1.0001 {
        println!("Overshoot eq {}", new_step.norm() / step.norm());
    }

    new_step
}

fn print_vector(v: &DVector<f64>) {
    for x in v.iter() {
        print!("{:.3} ", x);
    }
    println!("");
}
